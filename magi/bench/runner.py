"""Benchmark runner — compares single model vs MAGI on multiple-choice questions."""
import asyncio
import time
from dataclasses import dataclass, field

from magi.bench.datasets import BenchQuestion
from magi.core.engine import MAGI
from magi.core.node import MagiNode, Persona


@dataclass
class BenchResult:
    question: str
    category: str
    correct_answer: str
    magi_answer: str | None
    magi_correct: bool
    magi_confidence: float
    magi_protocol: str
    single_results: dict[str, dict]  # node_name -> {answer, correct}
    latency_ms: int


@dataclass
class BenchReport:
    total: int = 0
    magi_correct: int = 0
    single_correct: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, dict] = field(default_factory=dict)
    results: list[BenchResult] = field(default_factory=list)
    # Decision quality metrics
    calibration_buckets: dict[str, dict] = field(default_factory=dict)
    disagreement_value: dict[str, int] = field(default_factory=dict)

    @property
    def magi_accuracy(self) -> float:
        return self.magi_correct / self.total if self.total else 0.0

    def single_accuracy(self, node: str) -> float:
        return self.single_correct.get(node, 0) / self.total if self.total else 0.0


def _extract_choice(response: str, choices: list[str]) -> int | None:
    """Extract the chosen answer from an LLM response.

    Looks for patterns like "A)", "(A)", "The answer is A", etc.
    Returns the index (0-3) or None if can't determine.
    """
    response_upper = response.upper().strip()
    letters = ["A", "B", "C", "D"]

    # Check for direct letter answer at the start
    for i, letter in enumerate(letters):
        if response_upper.startswith(f"{letter})") or response_upper.startswith(f"({letter})"):
            return i
        if response_upper.startswith(f"{letter}.") or response_upper.startswith(f"{letter} "):
            return i

    # Check for "the answer is X" pattern
    for i, letter in enumerate(letters):
        if f"ANSWER IS {letter}" in response_upper or f"ANSWER: {letter}" in response_upper:
            return i

    # Check for choice text appearing in the response
    for i, choice in enumerate(choices):
        if choice.lower() in response.lower():
            return i

    return None


def _build_mc_prompt(q: BenchQuestion) -> str:
    """Build a multiple-choice prompt."""
    letters = ["A", "B", "C", "D"]
    choices_str = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(q.choices))
    return (
        f"{q.question}\n\n{choices_str}\n\n"
        "Answer with just the letter (A, B, C, or D) followed by a brief explanation."
    )


async def run_benchmark(
    engine: MAGI,
    questions: list[BenchQuestion],
    mode: str = "vote",
    concurrency: int = 3,
) -> BenchReport:
    """Run a benchmark comparing MAGI vs individual models.

    Args:
        engine: MAGI engine instance.
        questions: List of benchmark questions.
        mode: MAGI decision mode.
        concurrency: Max concurrent questions (to respect rate limits).

    Returns:
        BenchReport with accuracy and decision quality metrics.
    """
    report = BenchReport()
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(q: BenchQuestion) -> BenchResult:
        async with semaphore:
            prompt = _build_mc_prompt(q)
            start = time.monotonic()

            # Run MAGI
            decision = await engine.ask(prompt, mode=mode)
            elapsed = int((time.monotonic() - start) * 1000)

            # Extract MAGI's answer
            magi_choice = _extract_choice(decision.ruling, q.choices)
            magi_correct = magi_choice == q.correct if magi_choice is not None else False

            # Extract individual node answers
            single_results = {}
            for node_name, answer in decision.votes.items():
                choice = _extract_choice(answer, q.choices)
                correct = choice == q.correct if choice is not None else False
                single_results[node_name] = {"answer": answer, "correct": correct, "choice": choice}

            return BenchResult(
                question=q.question,
                category=q.category,
                correct_answer=q.choices[q.correct],
                magi_answer=decision.ruling,
                magi_correct=magi_correct,
                magi_confidence=decision.confidence,
                magi_protocol=decision.protocol_used,
                single_results=single_results,
                latency_ms=elapsed,
            )

    # Run all questions with concurrency limit
    tasks = [run_one(q) for q in questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, Exception):
            continue
        report.total += 1
        report.results.append(r)

        if r.magi_correct:
            report.magi_correct += 1

        for node_name, sr in r.single_results.items():
            if node_name not in report.single_correct:
                report.single_correct[node_name] = 0
            if sr["correct"]:
                report.single_correct[node_name] += 1

        # Per-category tracking
        cat = r.category
        if cat not in report.by_category:
            report.by_category[cat] = {"total": 0, "magi_correct": 0, "single": {}}
        report.by_category[cat]["total"] += 1
        if r.magi_correct:
            report.by_category[cat]["magi_correct"] += 1
        for node_name, sr in r.single_results.items():
            if node_name not in report.by_category[cat]["single"]:
                report.by_category[cat]["single"][node_name] = 0
            if sr["correct"]:
                report.by_category[cat]["single"][node_name] += 1

    # Decision quality metrics
    _compute_calibration(report)
    _compute_disagreement_value(report)

    return report


def _compute_calibration(report: BenchReport) -> None:
    """Compute confidence calibration: does high confidence = actually correct?"""
    buckets = {
        "0.0-0.2": {"total": 0, "correct": 0},
        "0.2-0.4": {"total": 0, "correct": 0},
        "0.4-0.6": {"total": 0, "correct": 0},
        "0.6-0.8": {"total": 0, "correct": 0},
        "0.8-1.0": {"total": 0, "correct": 0},
    }
    for r in report.results:
        conf = r.magi_confidence
        if conf < 0.2:
            key = "0.0-0.2"
        elif conf < 0.4:
            key = "0.2-0.4"
        elif conf < 0.6:
            key = "0.4-0.6"
        elif conf < 0.8:
            key = "0.6-0.8"
        else:
            key = "0.8-1.0"
        buckets[key]["total"] += 1
        if r.magi_correct:
            buckets[key]["correct"] += 1

    report.calibration_buckets = buckets


def _compute_disagreement_value(report: BenchReport) -> None:
    """Compute disagreement value: how often is the minority report right?

    Tracks:
    - all_agree_wrong: all models agreed but were wrong
    - dissenter_right: at least one model disagreed AND was correct
    - all_agree_right: all models agreed and were correct
    """
    stats = {"all_agree_wrong": 0, "dissenter_right": 0, "all_agree_right": 0, "total_disagreements": 0}

    for r in report.results:
        node_answers = [sr["correct"] for sr in r.single_results.values()]
        if not node_answers:
            continue

        all_same_correctness = all(a == node_answers[0] for a in node_answers)

        if all_same_correctness and node_answers[0]:
            stats["all_agree_right"] += 1
        elif all_same_correctness and not node_answers[0]:
            stats["all_agree_wrong"] += 1
        else:
            stats["total_disagreements"] += 1
            # At least one disagrees — check if any dissenter was right
            if any(sr["correct"] for sr in r.single_results.values()):
                stats["dissenter_right"] += 1

    report.disagreement_value = stats
