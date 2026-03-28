"""MAGI NERV Command Center — WebSocket server for real-time decision visualization."""
import asyncio
import json
import time
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from magi.core.engine import MAGI
from magi.core.node import MagiNode, Persona
from magi.core.decision import Decision
from magi.presets import get_preset, list_presets
from magi.protocols.critique import _estimate_agreement

app = FastAPI(title="MAGI NERV Command Center")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/presets")
async def api_presets():
    return {
        name: [{"name": p.name, "description": p.description} for p in personas]
        for name, personas in __import__("magi.presets", fromlist=["PRESETS"]).PRESETS.items()
    }


@app.websocket("/ws/ask")
async def ws_ask(ws: WebSocket):
    """WebSocket endpoint for real-time MAGI decisions.

    Client sends:
    {"query": "...", "mode": "adaptive", "preset": "eva",
     "melchior": "model", "balthasar": "model", "casper": "model"}

    Server streams events:
    {"event": "start", "query": "...", "mode": "...", "nodes": [...]}
    {"event": "node_start", "node": "melchior"}
    {"event": "node_done", "node": "melchior", "answer": "...", "latency_ms": ...}
    {"event": "node_error", "node": "melchior", "error": "..."}
    {"event": "agreement", "score": 0.85, "route": "vote"}
    {"event": "critique_round", "round": 1, "answers": {...}}
    {"event": "decision", "ruling": "...", "confidence": ..., ...}
    """
    await ws.accept()

    try:
        data = await ws.receive_json()
    except Exception:
        await ws.close(code=1003)
        return

    query = data.get("query", "")
    mode = data.get("mode", "adaptive")
    preset_name = data.get("preset", "eva")

    melchior_model = data.get("melchior", "openrouter/anthropic/claude-sonnet-4.6")
    balthasar_model = data.get("balthasar", "openrouter/openai/gpt-4o")
    casper_model = data.get("casper", "openrouter/google/gemini-2.5-pro")

    try:
        personas = get_preset(preset_name)
    except KeyError:
        personas = get_preset("eva")

    nodes = [
        MagiNode("melchior", melchior_model, personas[0], timeout=60),
        MagiNode("balthasar", balthasar_model, personas[1], timeout=60),
        MagiNode("casper", casper_model, personas[2], timeout=60),
    ]

    # Send start event
    await ws.send_json({
        "event": "start",
        "query": query,
        "mode": mode,
        "preset": preset_name,
        "nodes": [
            {"name": n.name, "model": n.model, "persona": n.persona.name}
            for n in nodes
        ],
    })

    try:
        # Phase 1: Parallel query all nodes
        results = {}
        failed = []
        start_time = time.monotonic()

        async def query_node(node):
            await ws.send_json({"event": "node_start", "node": node.name})
            node_start = time.monotonic()
            try:
                answer = await node.query(query)
                latency = int((time.monotonic() - node_start) * 1000)
                results[node.name] = answer
                await ws.send_json({
                    "event": "node_done",
                    "node": node.name,
                    "answer": answer[:2000],
                    "latency_ms": latency,
                })
            except Exception as e:
                failed.append(node.name)
                await ws.send_json({
                    "event": "node_error",
                    "node": node.name,
                    "error": str(e)[:200],
                })

        await asyncio.gather(*[query_node(n) for n in nodes])

        if not results:
            await ws.send_json({"event": "error", "message": "All nodes failed"})
            return

        # Phase 2: Agreement scoring
        answers = list(results.values())
        agreement = _estimate_agreement(answers)
        route = "vote" if agreement > 0.8 else ("critique" if agreement > 0.4 else "escalate")
        if mode != "adaptive":
            route = mode

        await ws.send_json({
            "event": "agreement",
            "score": round(agreement, 3),
            "route": route,
        })

        # Phase 3: Protocol execution
        if route == "vote" or len(results) < 2:
            # Just use vote result
            ruling = answers[0]
            confidence = agreement
            minority_parts = [
                f"[{name}]: {ans[:500]}"
                for name, ans in results.items()
                if ans != ruling
            ]
            protocol = "vote"
        else:
            # Critique rounds
            active_nodes = [n for n in nodes if n.name not in failed]
            current_answers = dict(results)

            for round_num in range(3 if route == "critique" else 2):
                if agreement > 0.8:
                    break

                await ws.send_json({
                    "event": "critique_start",
                    "round": round_num + 1,
                })

                critique_tasks = {}
                for node in active_nodes:
                    if node.name not in current_answers:
                        continue
                    others = {k: v for k, v in current_answers.items() if k != node.name}
                    prompt = (
                        f"Original question: {query}\n\n"
                        f"Your previous answer:\n{current_answers[node.name]}\n\n"
                        f"Other perspectives:\n"
                        + "\n\n".join(f"[{n}]: {a}" for n, a in others.items())
                        + "\n\nReview others. Where do you agree/disagree? Provide your revised answer."
                    )
                    critique_tasks[node.name] = asyncio.create_task(node.query(prompt))

                for name, task in critique_tasks.items():
                    try:
                        revised = await task
                        current_answers[name] = revised
                        await ws.send_json({
                            "event": "critique_done",
                            "round": round_num + 1,
                            "node": name,
                            "answer": revised[:2000],
                        })
                    except Exception:
                        pass

                agreement = _estimate_agreement(list(current_answers.values()))
                await ws.send_json({
                    "event": "critique_agreement",
                    "round": round_num + 1,
                    "score": round(agreement, 3),
                })

            ruling = list(current_answers.values())[0]
            confidence = agreement
            minority_parts = [
                f"[{name}]: {ans[:500]}"
                for name, ans in current_answers.items()
                if ans != ruling
            ]
            protocol = f"{route}_r{round_num + 1}"

        total_ms = int((time.monotonic() - start_time) * 1000)

        # Send final decision
        await ws.send_json({
            "event": "decision",
            "ruling": ruling[:3000],
            "confidence": round(confidence, 3),
            "minority_report": "\n\n".join(minority_parts)[:3000] if minority_parts else "",
            "protocol_used": protocol,
            "degraded": len(failed) > 0,
            "failed_nodes": failed,
            "latency_ms": total_ms,
            "votes": {k: v[:1000] for k, v in results.items()},
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"event": "error", "message": str(e)[:500]})
        except Exception:
            pass


def start_server(host: str = "0.0.0.0", port: int = 3000):
    """Start the NERV Command Center."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
