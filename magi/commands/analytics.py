"""magi analytics — trend analysis from JSONL traces."""
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class NodeStats:
    total: int = 0
    disagreed_with_ruling: int = 0
    failed: int = 0


@dataclass
class AnalyticsReport:
    total_decisions: int = 0
    skipped_lines: int = 0
    by_protocol: dict[str, int] = field(default_factory=dict)
    by_node: dict[str, NodeStats] = field(default_factory=dict)
    avg_confidence: float = 0.0
    avg_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    degraded_count: int = 0
    mind_change_count: int = 0


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value, default: str = "") -> str:
    """Safely convert a value to string."""
    if value is None:
        return default
    return str(value)


def stream_traces(trace_dir: str) -> Iterator[dict]:
    """Stream traces one by one from JSONL files. Memory-efficient."""
    trace_path = Path(trace_dir)
    if not trace_path.exists():
        return

    for f in sorted(trace_path.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    yield None  # signal a skipped line
                    continue
                if not isinstance(parsed, dict):
                    yield None
                    continue
                yield parsed


def load_traces(trace_dir: str) -> list[dict]:
    """Load all traces into memory. For small datasets or replay."""
    return [t for t in stream_traces(trace_dir) if t is not None]


def analyze_stream(trace_dir: str) -> AnalyticsReport:
    """Analyze traces using streaming. Memory-efficient for large datasets."""
    report = AnalyticsReport()
    total_conf = 0.0
    total_latency = 0.0

    for t in stream_traces(trace_dir):
        if t is None:
            report.skipped_lines += 1
            continue

        report.total_decisions += 1

        # Protocol distribution
        protocol = _safe_str(t.get("protocol_used"), "unknown")
        report.by_protocol[protocol] = report.by_protocol.get(protocol, 0) + 1

        # Confidence
        total_conf += _safe_float(t.get("confidence"))

        # Latency
        total_latency += _safe_float(t.get("latency_ms"))

        # Cost
        report.total_cost_usd += _safe_float(t.get("cost_usd"))

        # Degraded
        if t.get("degraded", False):
            report.degraded_count += 1

        # Mind changes
        changes = t.get("mind_changes", [])
        if changes:
            report.mind_change_count += 1

        # Per-node stats
        votes = t.get("votes", {})
        if not isinstance(votes, dict):
            continue
        failed_nodes = t.get("failed_nodes", [])
        if not isinstance(failed_nodes, list):
            failed_nodes = []
        ruling = _safe_str(t.get("ruling"))

        for node_name in votes:
            if node_name not in report.by_node:
                report.by_node[node_name] = NodeStats()
            report.by_node[node_name].total += 1
            # Track when this node's answer differs from the final ruling
            if _safe_str(votes[node_name]) != ruling:
                report.by_node[node_name].disagreed_with_ruling += 1

        for node_name in failed_nodes:
            if node_name not in report.by_node:
                report.by_node[node_name] = NodeStats()
            report.by_node[node_name].failed += 1

    if report.total_decisions > 0:
        report.avg_confidence = total_conf / report.total_decisions
        report.avg_latency_ms = total_latency / report.total_decisions

    return report


def analyze(traces: list[dict]) -> AnalyticsReport:
    """Analyze a list of trace dicts. For testing or pre-loaded data."""
    report = AnalyticsReport()
    total_conf = 0.0
    total_latency = 0.0

    for t in traces:
        if not isinstance(t, dict):
            report.skipped_lines += 1
            continue

        report.total_decisions += 1

        protocol = _safe_str(t.get("protocol_used"), "unknown")
        report.by_protocol[protocol] = report.by_protocol.get(protocol, 0) + 1

        total_conf += _safe_float(t.get("confidence"))
        total_latency += _safe_float(t.get("latency_ms"))
        report.total_cost_usd += _safe_float(t.get("cost_usd"))

        if t.get("degraded", False):
            report.degraded_count += 1

        changes = t.get("mind_changes", [])
        if changes:
            report.mind_change_count += 1

        votes = t.get("votes", {})
        if not isinstance(votes, dict):
            continue
        failed_nodes = t.get("failed_nodes", [])
        if not isinstance(failed_nodes, list):
            failed_nodes = []
        ruling = _safe_str(t.get("ruling"))

        for node_name in votes:
            if node_name not in report.by_node:
                report.by_node[node_name] = NodeStats()
            report.by_node[node_name].total += 1
            if _safe_str(votes[node_name]) != ruling:
                report.by_node[node_name].disagreed_with_ruling += 1

        for node_name in failed_nodes:
            if node_name not in report.by_node:
                report.by_node[node_name] = NodeStats()
            report.by_node[node_name].failed += 1

    if report.total_decisions > 0:
        report.avg_confidence = total_conf / report.total_decisions
        report.avg_latency_ms = total_latency / report.total_decisions

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
    if report.skipped_lines > 0:
        lines.append(f"Skipped lines (malformed): {report.skipped_lines}")
    lines.append(f"Average confidence: {report.avg_confidence:.0%}")
    lines.append(f"Average latency: {report.avg_latency_ms:.0f}ms")
    lines.append(f"Total cost: ${report.total_cost_usd:.4f}")
    lines.append(f"Degraded decisions: {report.degraded_count}")
    lines.append(f"Decisions with mind changes: {report.mind_change_count}")

    # Protocol distribution
    lines.append(f"\n{'-' * 60}")
    lines.append("PROTOCOL DISTRIBUTION")
    lines.append(f"{'-' * 60}")
    for protocol, count in sorted(report.by_protocol.items(), key=lambda x: -x[1]):
        pct = count / report.total_decisions
        bar = "#" * int(pct * 30)
        lines.append(f"  {protocol:30s} {count:>4d} ({pct:>5.1%}) {bar}")

    # Per-node stats
    lines.append(f"\n{'-' * 60}")
    lines.append("PER-NODE STATS")
    lines.append(f"{'-' * 60}")
    lines.append(f"  {'Node':15s} {'Responded':>10s} {'Disagreed':>10s} {'Failed':>8s}")
    lines.append(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*8}")
    for node, stats in sorted(report.by_node.items()):
        dis_pct = stats.disagreed_with_ruling / stats.total if stats.total else 0
        lines.append(
            f"  {node:15s} {stats.total:>10d} "
            f"{stats.disagreed_with_ruling:>7d} ({dis_pct:>4.0%}) "
            f"{stats.failed:>8d}"
        )

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def format_replay(trace: dict) -> str:
    """Format a single trace for decision replay."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"DECISION REPLAY — {_safe_str(trace.get('trace_id'), 'unknown')}")
    lines.append("=" * 60)

    lines.append(f"\nQuery: {_safe_str(trace.get('query'), 'N/A')[:200]}")
    lines.append(f"Protocol: {_safe_str(trace.get('protocol_used'), 'N/A')}")
    lines.append(f"Confidence: {_safe_float(trace.get('confidence')):.0%}")
    lines.append(f"Latency: {_safe_float(trace.get('latency_ms')):.0f}ms")

    if trace.get("degraded"):
        failed = trace.get("failed_nodes", [])
        if isinstance(failed, list):
            lines.append(f"Degraded: yes (failed: {', '.join(str(n) for n in failed)})")

    lines.append(f"\n{'-' * 60}")
    lines.append("VOTES")
    votes = trace.get("votes", {})
    if isinstance(votes, dict):
        for name, answer in votes.items():
            lines.append(f"\n-- {str(name).upper()} --")
            lines.append(_safe_str(answer)[:500])

    if trace.get("mind_changes"):
        changes = trace["mind_changes"]
        if isinstance(changes, list):
            lines.append(f"\n{'-' * 60}")
            lines.append(f"MIND CHANGES: {', '.join(str(c) for c in changes)}")

    lines.append(f"\n{'-' * 60}")
    lines.append("RULING")
    lines.append(_safe_str(trace.get("ruling"), "N/A")[:500])

    mr = trace.get("minority_report")
    if mr:
        lines.append(f"\n{'-' * 60}")
        lines.append("MINORITY REPORT")
        lines.append(_safe_str(mr)[:500])

    lines.append("")
    return "\n".join(lines)
