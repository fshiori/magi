"""AdaptiveProtocol — dynamically selects vote, critique, or escalate.

Flow:
1. Query all 3 nodes in parallel (same as vote round 0)
2. Compute agreement_score on the 3 responses
3. Route based on score:
   - > threshold_high (0.8) → return majority vote (fast path)
   - threshold_low..threshold_high (0.4-0.8) → run CritiqueProtocol
   - < threshold_low (0.4) → run EscalateProtocol (critique with max_rounds=2, forced ruling)
"""
import asyncio
import time

from magi.core.decision import Decision
from magi.protocols.critique import critique, _estimate_agreement


async def adaptive(
    query: str,
    nodes,
    threshold_high: float = 0.8,
    threshold_low: float = 0.4,
) -> Decision:
    """Run the adaptive protocol: vote first, escalate if disagreement detected.

    Args:
        query: The user's question.
        nodes: List of MagiNode instances.
        threshold_high: Above this, just vote (fast path).
        threshold_low: Below this, escalate (forced ruling after 2 rounds).

    Returns:
        A Decision with the appropriate protocol applied.
    """
    start = time.monotonic()

    # Step 1: parallel query all nodes (same as vote round 0)
    tasks = {node.name: asyncio.create_task(node.query(query)) for node in nodes}

    results: dict[str, str] = {}
    failed: list[str] = []

    for name, task in tasks.items():
        try:
            results[name] = await task
        except Exception:
            failed.append(name)

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
            protocol_used="adaptive_fallback_single",
            degraded=True,
            failed_nodes=failed,
            latency_ms=int((time.monotonic() - start) * 1000),
        )

    # Step 2: compute agreement
    answers = list(results.values())
    agreement = _estimate_agreement(answers)

    # Step 3: route
    if agreement >= threshold_high:
        # High agreement → fast vote path, no extra calls
        elapsed_ms = int((time.monotonic() - start) * 1000)
        ruling = answers[0]
        minority_parts = [
            f"[{name}]: {answer}"
            for name, answer in results.items()
            if answer != ruling
        ]
        return Decision(
            query=query,
            ruling=ruling,
            confidence=agreement,
            minority_report="\n\n".join(minority_parts),
            votes=results,
            protocol_used="adaptive_vote",
            degraded=len(failed) > 0,
            failed_nodes=failed,
            latency_ms=elapsed_ms,
        )

    if agreement >= threshold_low:
        # Medium disagreement → critique (up to 3 rounds)
        active_nodes = [n for n in nodes if n.name not in failed]
        decision = await critique(query, active_nodes, max_rounds=3)
        decision.protocol_used = f"adaptive_critique_{decision.protocol_used}"
        decision.latency_ms = int((time.monotonic() - start) * 1000)
        return decision

    # Severe disagreement → escalate (critique with max_rounds=2, forced ruling)
    active_nodes = [n for n in nodes if n.name not in failed]
    decision = await critique(query, active_nodes, max_rounds=2)
    decision.protocol_used = f"adaptive_escalate_{decision.protocol_used}"
    decision.latency_ms = int((time.monotonic() - start) * 1000)
    return decision
