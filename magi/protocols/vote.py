"""VoteProtocol — structured voting with position extraction.

Each node is asked to state a clear POSITION first (a short tag like A/B/C/D,
YES/NO, APPROVE/REJECT), then explain. Voting is based on extracted positions,
not full-text comparison.
"""
import asyncio
import re
from collections import Counter
from magi.core.decision import Decision
import time


def _wrap_structured_prompt(query: str) -> str:
    """Wrap query to request structured position + explanation."""
    return (
        f"{query}\n\n"
        "IMPORTANT: Start your response with a clear position on the FIRST LINE.\n"
        "Format: POSITION: <your stance in a few words>\n"
        "Then explain your reasoning below."
    )


def _extract_position(response: str) -> str:
    """Extract the structured position from a response.

    Looks for 'POSITION: ...' on the first few lines.
    Falls back to the first line if no POSITION tag found.
    """
    for line in response.strip().split("\n")[:5]:
        line = line.strip()
        match = re.match(r"(?:POSITION|Position|position)\s*:\s*(.+)", line)
        if match:
            return match.group(1).strip().lower()
    # Fallback: first non-empty line, lowered and truncated
    first = response.strip().split("\n")[0].strip().lower()
    return first[:100]


def _find_majority(positions: dict[str, str]) -> tuple[str | None, list[str], list[str]]:
    """Find majority position among nodes.

    Returns (winning_position, majority_nodes, minority_nodes).
    If no majority, returns (None, [], all_nodes).
    """
    counter = Counter(positions.values())
    if not counter:
        return None, [], []

    most_common_pos, count = counter.most_common(1)[0]
    total = len(positions)

    if count > total / 2:
        majority = [n for n, p in positions.items() if p == most_common_pos]
        minority = [n for n, p in positions.items() if p != most_common_pos]
        return most_common_pos, majority, minority

    return None, [], list(positions.keys())


async def vote(query: str, nodes, timeout: float = 30.0) -> Decision:
    """Ask all nodes the same query in parallel, then majority-vote on positions.

    Each node's response is parsed for a POSITION tag. Voting is based on
    position similarity, not full-text comparison.

    If no majority exists (all 3 different positions), returns with
    protocol_used="vote_no_majority" to signal escalation to critique.
    """
    start = time.monotonic()

    structured_query = _wrap_structured_prompt(query)
    tasks = {node.name: asyncio.create_task(node.query(structured_query)) for node in nodes}

    results: dict[str, str] = {}
    failed: list[str] = []

    for name, task in tasks.items():
        try:
            results[name] = await task
        except Exception:
            failed.append(name)

    elapsed_ms = int((time.monotonic() - start) * 1000)

    if not results:
        raise RuntimeError("All MAGI nodes failed. Cannot make a decision.")

    if len(results) == 1:
        name, answer = next(iter(results.items()))
        return Decision(
            query=query,
            ruling=answer,
            confidence=0.3,
            minority_report="",
            votes=results,
            protocol_used="fallback_single",
            degraded=True,
            failed_nodes=failed,
            latency_ms=elapsed_ms,
        )

    # Extract positions
    positions = {name: _extract_position(answer) for name, answer in results.items()}
    winning_pos, majority_nodes, minority_nodes = _find_majority(positions)

    if winning_pos is None:
        # No majority — signal for escalation
        # Use first response as provisional ruling
        first_name = next(iter(results))
        minority_parts = [
            f"[{name}] (position: {positions[name]}): {results[name]}"
            for name in results if name != first_name
        ]
        return Decision(
            query=query,
            ruling=results[first_name],
            confidence=1.0 / len(results),
            minority_report="\n\n".join(minority_parts),
            votes=results,
            protocol_used="vote_no_majority",
            degraded=len(failed) > 0,
            failed_nodes=failed,
            latency_ms=elapsed_ms,
        )

    # Majority found — use first majority node's full response as ruling
    ruling_node = majority_nodes[0]
    ruling = results[ruling_node]
    confidence = len(majority_nodes) / len(results)

    minority_parts = [
        f"[{name}] (position: {positions[name]}): {results[name]}"
        for name in minority_nodes
    ]

    return Decision(
        query=query,
        ruling=ruling,
        confidence=confidence,
        minority_report="\n\n".join(minority_parts),
        votes=results,
        protocol_used="vote",
        degraded=len(failed) > 0,
        failed_nodes=failed,
        latency_ms=elapsed_ms,
    )
