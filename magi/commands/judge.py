"""magi judge — multi-model answer scoring."""


def build_judge_prompt(question: str, answer: str) -> str:
    """Build a prompt for multi-model answer evaluation."""
    return (
        "You are evaluating the quality of an answer to a question.\n\n"
        f"**Question:** {question}\n\n"
        f"**Answer:** {answer}\n\n"
        "Rate the answer on a scale of 1-10 across these dimensions:\n"
        "- **Correctness** (1-10): Is the answer factually accurate?\n"
        "- **Completeness** (1-10): Does it address all aspects of the question?\n"
        "- **Clarity** (1-10): Is it well-explained and easy to understand?\n"
        "- **Overall** (1-10): Your holistic assessment.\n\n"
        "Format your response as:\n"
        "Correctness: X/10\n"
        "Completeness: X/10\n"
        "Clarity: X/10\n"
        "Overall: X/10\n\n"
        "Then provide a brief explanation of your scoring."
    )


def format_judge_output(decision) -> str:
    """Format a judge Decision into a readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("MAGI JUDGE — Three Models Score Your Answer")
    lines.append("=" * 60)
    lines.append("")

    for node_name, review in decision.votes.items():
        lines.append(f"── {node_name.upper()} ──")
        lines.append(review)
        lines.append("")

    lines.append("=" * 60)
    lines.append("DISAGREEMENT ANALYSIS")
    lines.append("=" * 60)
    lines.append(f"Agreement: {decision.confidence:.0%}")
    lines.append(f"Protocol: {decision.protocol_used}")
    if decision.degraded:
        lines.append(f"Degraded: failed nodes = {', '.join(decision.failed_nodes)}")
    lines.append("")
    lines.append(f"Trace: {decision.trace_id} | Latency: {decision.latency_ms}ms")

    return "\n".join(lines)
