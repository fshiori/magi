"""CritiqueProtocol (ICE — Iterative Consensus Ensemble).

Flow:
1. All nodes answer the query in parallel (round 0)
2. Each node sees the other two answers and critiques them
3. Each node revises its answer based on critiques
4. Repeat until consensus (agreement_score > 0.8) or max_rounds reached
5. If no consensus, the first node synthesizes a final answer

Consensus is measured by a simple heuristic: if all revised answers
start converging (become more similar), confidence increases.
"""
import asyncio
import time

from magi.core.decision import Decision


def _build_critique_prompt(query: str, own_answer: str, other_answers: dict[str, str]) -> str:
    """Build a prompt asking a node to critique others and revise its own answer."""
    others_text = "\n\n".join(
        f"[{name}]: {answer}" for name, answer in other_answers.items()
    )
    return (
        f"Original question: {query}\n\n"
        f"Your previous answer:\n{own_answer}\n\n"
        f"Other perspectives:\n{others_text}\n\n"
        "Review the other perspectives. Where do you agree? Where do you disagree and why? "
        "Then provide your revised answer, incorporating valid points from others while "
        "maintaining your honest assessment. If your position hasn't changed, explain why."
    )


def _estimate_agreement(answers: list[str]) -> float:
    """Rough agreement score based on response length similarity and shared terms.

    Returns 0-1. This is a heuristic — a proper implementation would use
    LLM-as-judge or embedding similarity. For MVP, we use word overlap.
    """
    if len(answers) < 2:
        return 1.0

    # Tokenize into word sets
    word_sets = [set(a.lower().split()) for a in answers]

    # Pairwise Jaccard similarity
    similarities = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            intersection = word_sets[i] & word_sets[j]
            union = word_sets[i] | word_sets[j]
            if union:
                similarities.append(len(intersection) / len(union))
            else:
                similarities.append(0.0)

    return sum(similarities) / len(similarities) if similarities else 0.0


async def critique(
    query: str,
    nodes,
    max_rounds: int = 3,
    consensus_threshold: float = 0.8,
) -> Decision:
    """Run the ICE (Iterative Consensus Ensemble) protocol.

    Args:
        query: The user's question.
        nodes: List of MagiNode instances.
        max_rounds: Maximum critique rounds before forcing a result.
        consensus_threshold: agreement_score above which we stop early.

    Returns:
        A Decision with the synthesized result.
    """
    start = time.monotonic()
    failed = []
    mind_changes: list[str] = []

    # Round 0: initial parallel answers
    tasks = {node.name: asyncio.create_task(node.query(query)) for node in nodes}

    current_answers: dict[str, str] = {}
    for name, task in tasks.items():
        try:
            current_answers[name] = await task
        except Exception:
            failed.append(name)

    if not current_answers:
        raise RuntimeError("All MAGI nodes failed. Cannot make a decision.")

    if len(current_answers) == 1:
        # Single node fallback
        name, answer = next(iter(current_answers.items()))
        return Decision(
            query=query,
            ruling=answer,
            confidence=0.3,
            minority_report="",
            votes=current_answers,
            protocol_used="fallback_single",
            degraded=True,
            failed_nodes=failed,
            latency_ms=int((time.monotonic() - start) * 1000),
        )

    # Track initial answers for mind_changes detection
    initial_answers = dict(current_answers)

    # Critique rounds
    active_nodes = [n for n in nodes if n.name not in failed]
    agreement = _estimate_agreement(list(current_answers.values()))
    rounds_run = 0

    for round_num in range(max_rounds):
        if agreement >= consensus_threshold:
            break

        rounds_run = round_num + 1

        # Each node critiques and revises
        critique_tasks = {}
        for node in active_nodes:
            others = {k: v for k, v in current_answers.items() if k != node.name}
            if node.name not in current_answers:
                continue
            prompt = _build_critique_prompt(query, current_answers[node.name], others)
            critique_tasks[node.name] = asyncio.create_task(node.query(prompt))

        # Collect revised answers
        for name, task in critique_tasks.items():
            try:
                revised = await task
                if revised != current_answers.get(name):
                    current_answers[name] = revised
            except Exception:
                # Node failed mid-critique, keep its last answer
                pass

        agreement = _estimate_agreement(list(current_answers.values()))

    # Detect mind changes: nodes whose final answer differs significantly from initial
    for name in current_answers:
        if name in initial_answers:
            initial_words = set(initial_answers[name].lower().split())
            final_words = set(current_answers[name].lower().split())
            if initial_words and final_words:
                overlap = len(initial_words & final_words) / len(initial_words | final_words)
                if overlap < 0.5:  # significant change
                    mind_changes.append(name)

    elapsed_ms = int((time.monotonic() - start) * 1000)

    # Build ruling: use the first active node's final answer as ruling
    ruling_node = active_nodes[0].name
    ruling = current_answers[ruling_node]

    # Minority report: other answers that differ
    minority_parts = []
    for name, answer in current_answers.items():
        if name != ruling_node:
            minority_parts.append(f"[{name}]: {answer}")
    minority_report = "\n\n".join(minority_parts)

    return Decision(
        query=query,
        ruling=ruling,
        confidence=agreement,
        minority_report=minority_report,
        votes=current_answers,
        mind_changes=mind_changes,
        protocol_used=f"critique_ice_r{rounds_run}",
        degraded=len(failed) > 0,
        failed_nodes=failed,
        latency_ms=elapsed_ms,
    )
