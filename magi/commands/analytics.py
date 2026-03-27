"""magi analytics — trend analysis from JSONL traces."""
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AnalyticsReport:
    total_decisions: int = 0
    by_protocol: dict[str, int] = field(default_factory=dict)
    by_node: dict[str, dict] = field(default_factory=dict)  # node -> {total, minority, failed}
    avg_confidence: float = 0.0
    avg_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    degraded_count: int = 0
    mind_change_count: int = 0


def load_traces(trace_dir: str) -> list[dict]:
    """Load all JSONL trace files from the trace directory."""
    traces = []
    trace_path = Path(trace_dir)
    if not trace_path.exists():
        return traces

    for f in sorted(trace_path.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        traces.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return traces


def analyze(traces: list[dict]) -> AnalyticsReport:
    """Analyze traces and produce an AnalyticsReport."""
    report = AnalyticsReport()
    report.total_decisions = len(traces)

    if not traces:
        return report

    total_conf = 0.0
    total_latency = 0.0

    for t in traces:
        # Protocol distribution
        protocol = t.get("protocol_used", "unknown")
        report.by_protocol[protocol] = report.by_protocol.get(protocol, 0) + 1

        # Confidence
        total_conf += t.get("confidence", 0)

        # Latency
        total_latency += t.get("latency_ms", 0)

        # Cost
        report.total_cost_usd += t.get("cost_usd", 0)

        # Degraded
        if t.get("degraded", False):
            report.degraded_count += 1

        # Mind changes
        changes = t.get("mind_changes", [])
        if changes:
            report.mind_change_count += 1

        # Per-node stats
        votes = t.get("votes", {})
        failed_nodes = t.get("failed_nodes", [])
        ruling = t.get("ruling", "")

        for node_name in votes:
            if node_name not in report.by_node:
                report.by_node[node_name] = {"total": 0, "minority": 0, "failed": 0}
            report.by_node[node_name]["total"] += 1
            # Was this node the minority? (its answer != ruling)
            if votes[node_name] != ruling:
                report.by_node[node_name]["minority"] += 1

        for node_name in failed_nodes:
            if node_name not in report.by_node:
                report.by_node[node_name] = {"total": 0, "minority": 0, "failed": 0}
            report.by_node[node_name]["failed"] += 1

    report.avg_confidence = total_conf / len(traces)
    report.avg_latency_ms = total_latency / len(traces)

    return report


def format_analytics(report: AnalyticsReport) -> str:
    """Format AnalyticsReport into readable output."""
    if report.total_decisions == 0:
        return "No trace data found. Run some queries first: magi ask \"your question\""

    lines = []
    lines.append("=" * 60)
    lines.append("MAGI ANALYTICS")
    lines.append("=" * 60)

    lines.append(f"\nTotal decisions: {report.total_decisions}")
    lines.append(f"Average confidence: {report.avg_confidence:.0%}")
    lines.append(f"Average latency: {report.avg_latency_ms:.0f}ms")
    lines.append(f"Total cost: ${report.total_cost_usd:.4f}")
    lines.append(f"Degraded decisions: {report.degraded_count}")
    lines.append(f"Decisions with mind changes: {report.mind_change_count}")

    # Protocol distribution
    lines.append(f"\n{'─' * 60}")
    lines.append("PROTOCOL DISTRIBUTION")
    lines.append(f"{'─' * 60}")
    for protocol, count in sorted(report.by_protocol.items(), key=lambda x: -x[1]):
        pct = count / report.total_decisions
        bar = "█" * int(pct * 30)
        lines.append(f"  {protocol:30s} {count:>4d} ({pct:>5.1%}) {bar}")

    # Per-node stats
    lines.append(f"\n{'─' * 60}")
    lines.append("PER-NODE STATS")
    lines.append(f"{'─' * 60}")
    lines.append(f"  {'Node':15s} {'Responded':>10s} {'Minority':>10s} {'Failed':>8s}")
    lines.append(f"  {'─'*15} {'─'*10} {'─'*10} {'─'*8}")
    for node, stats in sorted(report.by_node.items()):
        total = stats["total"]
        minority = stats["minority"]
        failed = stats["failed"]
        min_pct = minority / total if total else 0
        lines.append(f"  {node:15s} {total:>10d} {minority:>7d} ({min_pct:>4.0%}) {failed:>8d}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def format_replay(trace: dict) -> str:
    """Format a single trace for decision replay."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"DECISION REPLAY — {trace.get('trace_id', 'unknown')}")
    lines.append("=" * 60)

    lines.append(f"\nQuery: {trace.get('query', 'N/A')[:200]}")
    lines.append(f"Protocol: {trace.get('protocol_used', 'N/A')}")
    lines.append(f"Confidence: {trace.get('confidence', 0):.0%}")
    lines.append(f"Latency: {trace.get('latency_ms', 0)}ms")

    if trace.get("degraded"):
        lines.append(f"Degraded: yes (failed: {', '.join(trace.get('failed_nodes', []))})")

    lines.append(f"\n{'─' * 60}")
    lines.append("VOTES")
    for name, answer in trace.get("votes", {}).items():
        lines.append(f"\n── {name.upper()} ──")
        lines.append(answer[:500])

    if trace.get("mind_changes"):
        lines.append(f"\n{'─' * 60}")
        lines.append(f"MIND CHANGES: {', '.join(trace['mind_changes'])}")

    lines.append(f"\n{'─' * 60}")
    lines.append("RULING")
    lines.append(trace.get("ruling", "N/A")[:500])

    if trace.get("minority_report"):
        lines.append(f"\n{'─' * 60}")
        lines.append("MINORITY REPORT")
        lines.append(trace["minority_report"][:500])

    lines.append("")
    return "\n".join(lines)
