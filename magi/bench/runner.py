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
    judge_opinion: str | None = None


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


async def _verify_with_llm(
    judge_node: MagiNode,
    question: str,
    choices: list[str],
    correct_index: int,
    response: str,
) -> tuple[bool, str]:
    """Use a strong LLM to judge if a response is correct."""
    letters = ["A", "B", "C", "D"]
    correct_letter = letters[correct_index]
    choices_str = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(choices))

    prompt = (
        f"You are a benchmark judge. A model was asked a multiple-choice question.\n\n"
        f"Question: {question}\n"
        f"Choices:\n{choices_str}\n"
        f"Correct Answer: {correct_letter}) {choices[correct_index]}\n\n"
        f"Model's Response:\n\"\"\"\n{response}\n\"\"\"\n\n"
        "Did the model choose the correct answer? "
        "Reply with 'CORRECT' or 'WRONG' followed by a one-sentence explanation of your judgment."
    )

    try:
        judgment = await judge_node.query(prompt)
        is_correct = judgment.strip().upper().startswith("CORRECT")
        return is_correct, judgment
    except Exception as e:
        return False, f"Judge failed: {e}"


def _build_mc_prompt(q: BenchQuestion) -> str:
    """Build a multiple-choice prompt."""
    letters = ["A", "B", "C", "D"]
    choices_str = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(q.choices))
    return (
        f"{q.question}\n\n{choices_str}\n\n"
        "Answer with just the letter (A, B, C, or D) followed by a brief explanation."
    )


async def run_single_benchmark(
    model: str,
    questions: list[BenchQuestion],
    concurrency: int = 5,
    use_judge: bool = False,
    judge_model: str = "openrouter/google/gemini-3.1-pro-preview",
) -> BenchReport:
    """Run a benchmark for a single LLM model (Baseline)."""
    report = BenchReport()
    semaphore = asyncio.Semaphore(concurrency)

    # Initialize nodes
    persona = Persona("Baseline", "You are a standard LLM providing a baseline assessment.")
    node = MagiNode("baseline", model, persona)

    judge_node = None
    if use_judge:
        judge_persona = Persona("Judge", "You are an impartial benchmark judge.")
        judge_node = MagiNode("judge", judge_model, judge_persona)

    async def run_one(q: BenchQuestion) -> BenchResult:
        async with semaphore:
            prompt = _build_mc_prompt(q)
            start = time.monotonic()

            # Query the single node
            try:
                answer = await node.query(prompt)
            except Exception as e:
                raise RuntimeError(f"Model failed: {e}")

            elapsed = int((time.monotonic() - start) * 1000)

            # Extract answer
            choice = _extract_choice(answer, q.choices)
            is_correct = choice == q.correct if choice is not None else False
            judge_opinion = None

            # Verify with judge if needed
            if use_judge and judge_node and (choice is None or not is_correct):
                is_correct, opinion = await _verify_with_llm(
                    judge_node, q.question, q.choices, q.correct, answer
                )
                judge_opinion = opinion

            return BenchResult(
                question=q.question,
                category=q.category,
                correct_answer=q.choices[q.correct],
                magi_answer=answer,
                magi_correct=is_correct,
                magi_confidence=1.0,  # Single node has 100% confidence in itself
                magi_protocol="single_node",
                single_results={model: {"answer": answer, "correct": is_correct, "choice": choice}},
                latency_ms=elapsed,
                judge_opinion=judge_opinion,
            )

    # Run all questions
    tasks = [run_one(q) for q in questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, Exception):
            print(f"Question failed: {r}")
            continue
        report.total += 1
        report.results.append(r)
        if r.magi_correct:
            report.magi_correct += 1
        # Individual accuracy for the single model
        report.single_correct[model] = report.single_correct.get(model, 0) + (1 if r.magi_correct else 0)

        # Category tracking
        cat = r.category
        if cat not in report.by_category:
            report.by_category[cat] = {"total": 0, "magi_correct": 0, "single": {model: 0}}
        report.by_category[cat]["total"] += 1
        if r.magi_correct:
            report.by_category[cat]["magi_correct"] += 1
            report.by_category[cat]["single"][model] += 1

    return report
    """Run a benchmark comparing MAGI vs individual models.

    Args:
        engine: MAGI engine instance.
        questions: List of benchmark questions.
        mode: MAGI decision mode.
        concurrency: Max concurrent questions (to respect rate limits).
        use_judge: Whether to use an LLM judge for verification.
        judge_model: The model to use as a judge.

    Returns:
        BenchReport with accuracy and decision quality metrics.
    """
    report = BenchReport()
    semaphore = asyncio.Semaphore(concurrency)

    # Initialize judge node if needed
    judge_node = None
    if use_judge:
        judge_persona = Persona("Judge", "You are an impartial benchmark judge. You evaluate if answers are correct.")
        judge_node = MagiNode("judge", judge_model, judge_persona)

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
            judge_opinion = None

            # If simple extraction fails or is wrong, and we have a judge, double check
            if use_judge and judge_node and (magi_choice is None or not magi_correct):
                is_correct, opinion = await _verify_with_llm(
                    judge_node, q.question, q.choices, q.correct, decision.ruling
                )
                if is_correct:
                    magi_correct = True
                judge_opinion = opinion

            # Extract individual node answers
            single_results = {}
            for node_name, answer in decision.votes.items():
                choice = _extract_choice(answer, q.choices)
                correct = choice == q.correct if choice is not None else False

                # Also judge individual nodes if they failed simple extraction
                if use_judge and judge_node and (choice is None or not correct):
                    is_correct, _ = await _verify_with_llm(
                        judge_node, q.question, q.choices, q.correct, answer
                    )
                    if is_correct:
                        correct = True

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
                judge_opinion=judge_opinion,
            )

    # Run all questions with concurrency limit
    tasks = [run_one(q) for q in questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, Exception):
            import traceback
            print(f"Question failed with error: {r}")
            traceback.print_exception(type(r), r, r.__traceback__)
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
