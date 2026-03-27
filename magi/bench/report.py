"""Benchmark report formatting."""
from magi.bench.runner import BenchReport


def format_report(report: BenchReport) -> str:
    """Format a BenchReport into a human-readable string."""
    lines = []
    lines.append("=" * 70)
    lines.append("MAGI BENCHMARK REPORT")
    lines.append("=" * 70)

    # Overall accuracy
    lines.append("")
    lines.append(f"Total questions: {report.total}")
    lines.append(f"MAGI accuracy:  {report.magi_accuracy:.1%} ({report.magi_correct}/{report.total})")
    lines.append("")

    # Per-node accuracy
    lines.append("Individual model accuracy:")
    for node, correct in sorted(report.single_correct.items()):
        acc = correct / report.total if report.total else 0
        lines.append(f"  {node:15s} {acc:.1%} ({correct}/{report.total})")

    # Per-category breakdown
    lines.append("")
    lines.append("-" * 70)
    lines.append("BY CATEGORY")
    lines.append("-" * 70)
    for cat, data in sorted(report.by_category.items()):
        total = data["total"]
        magi_acc = data["magi_correct"] / total if total else 0
        lines.append(f"\n  {cat}")
        lines.append(f"    MAGI: {magi_acc:.0%} ({data['magi_correct']}/{total})")
        for node, correct in sorted(data.get("single", {}).items()):
            acc = correct / total if total else 0
            lines.append(f"    {node}: {acc:.0%} ({correct}/{total})")

    # Decision quality metrics
    lines.append("")
    lines.append("=" * 70)
    lines.append("DECISION QUALITY METRICS")
    lines.append("=" * 70)

    # Calibration
    lines.append("")
    lines.append("Confidence Calibration (is high confidence = actually correct?):")
    lines.append(f"  {'Confidence':15s} {'Total':>6s} {'Correct':>8s} {'Actual %':>9s}")
    lines.append(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*9}")
    for bucket, data in report.calibration_buckets.items():
        total = data["total"]
        correct = data["correct"]
        actual = correct / total if total else 0
        lines.append(f"  {bucket:15s} {total:>6d} {correct:>8d} {actual:>8.0%}")

    # Disagreement value
    lines.append("")
    lines.append("Disagreement Value (is the minority report useful?):")
    dv = report.disagreement_value
    lines.append(f"  All models agree & correct:  {dv.get('all_agree_right', 0)}")
    lines.append(f"  All models agree & WRONG:    {dv.get('all_agree_wrong', 0)}")
    lines.append(f"  Models disagree:             {dv.get('total_disagreements', 0)}")
    lines.append(f"  Dissenter was RIGHT:         {dv.get('dissenter_right', 0)}")

    total_disagree = dv.get("total_disagreements", 0)
    dissenter_right = dv.get("dissenter_right", 0)
    if total_disagree > 0:
        value_rate = dissenter_right / total_disagree
        lines.append(f"  Disagreement value rate:     {value_rate:.0%}")
        lines.append(f"  (When models disagree, the dissenter is right {value_rate:.0%} of the time)")
    else:
        lines.append("  (No disagreements observed)")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)
