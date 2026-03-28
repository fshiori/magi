"""Experiment: 三個臭皮匠 vs 諸葛亮
Can three free models using MAGI critique beat one strong closed-source model?

Test groups:
1. Single strong model (Claude Sonnet 4.6) — baseline
2. Three free models with MAGI vote
3. Three free models with MAGI critique

All run on the same 25 built-in benchmark questions.
"""
import asyncio
import sys
import time

import litellm

from magi.core.engine import MAGI
from magi.core.node import MagiNode, Persona, MELCHIOR, BALTHASAR, CASPER
from magi.bench.datasets import get_dataset
from magi.bench.runner import _build_mc_prompt, _extract_choice, BenchResult


# Models
STRONG_MODEL = "openrouter/anthropic/claude-sonnet-4.6"
FREE_MODELS = (
    "openrouter/xiaomi/mimo-v2-pro",
    "openrouter/minimax/minimax-m2.7",
    "openrouter/deepseek/deepseek-v3.2",
)


async def run_single_model(model: str, questions):
    """Run a single model on all questions."""
    correct = 0
    total = 0
    errors = 0

    for q in questions:
        prompt = _build_mc_prompt(q)
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                num_retries=2,
                timeout=60,
            )
            answer = response.choices[0].message.content
            choice = _extract_choice(answer, q.choices)
            if choice == q.correct:
                correct += 1
            total += 1
            status = "OK" if choice == q.correct else f"WRONG (got {choice}, expected {q.correct})"
            print(f"  [{total:2d}/{len(questions)}] {status} — {q.question[:50]}")
        except Exception as e:
            errors += 1
            total += 1
            print(f"  [{total:2d}/{len(questions)}] ERROR: {type(e).__name__} — {q.question[:50]}")

    return correct, total, errors


async def run_magi_group(models, questions, mode="vote"):
    """Run MAGI with given models on all questions."""
    engine = MAGI(
        melchior=models[0],
        balthasar=models[1],
        casper=models[2],
        timeout=60,
    )

    correct = 0
    total = 0
    errors = 0
    protocols = []

    for q in questions:
        prompt = _build_mc_prompt(q)
        try:
            decision = await engine.ask(prompt, mode=mode)
            choice = _extract_choice(decision.ruling, q.choices)
            if choice == q.correct:
                correct += 1
            total += 1
            status = "OK" if choice == q.correct else f"WRONG (got {choice}, expected {q.correct})"
            proto = decision.protocol_used
            protocols.append(proto)
            deg = " [DEGRADED]" if decision.degraded else ""
            print(f"  [{total:2d}/{len(questions)}] {status} — {proto}{deg} — {q.question[:40]}")
        except Exception as e:
            errors += 1
            total += 1
            print(f"  [{total:2d}/{len(questions)}] ERROR: {type(e).__name__} — {q.question[:40]}")

    return correct, total, errors, protocols


async def main():
    questions = get_dataset("builtin")
    print(f"Benchmark: {len(questions)} questions\n")

    # --- Group 1: Single strong model ---
    print("=" * 60)
    print(f"GROUP 1: Single strong model ({STRONG_MODEL.split('/')[-1]})")
    print("=" * 60)
    start = time.monotonic()
    g1_correct, g1_total, g1_errors = await run_single_model(STRONG_MODEL, questions)
    g1_time = time.monotonic() - start
    g1_acc = g1_correct / g1_total if g1_total else 0
    print(f"\nResult: {g1_correct}/{g1_total} ({g1_acc:.0%}) in {g1_time:.0f}s, {g1_errors} errors\n")

    # --- Group 2: Three free models — vote ---
    print("=" * 60)
    print("GROUP 2: Three free models — MAGI VOTE")
    print(f"  {' / '.join(m.split('/')[-1] for m in FREE_MODELS)}")
    print("=" * 60)
    start = time.monotonic()
    g2_correct, g2_total, g2_errors, g2_protos = await run_magi_group(FREE_MODELS, questions, "vote")
    g2_time = time.monotonic() - start
    g2_acc = g2_correct / g2_total if g2_total else 0
    print(f"\nResult: {g2_correct}/{g2_total} ({g2_acc:.0%}) in {g2_time:.0f}s, {g2_errors} errors\n")

    # --- Group 3: Three free models — critique ---
    print("=" * 60)
    print("GROUP 3: Three free models — MAGI CRITIQUE")
    print(f"  {' / '.join(m.split('/')[-1] for m in FREE_MODELS)}")
    print("=" * 60)
    start = time.monotonic()
    g3_correct, g3_total, g3_errors, g3_protos = await run_magi_group(FREE_MODELS, questions, "critique")
    g3_time = time.monotonic() - start
    g3_acc = g3_correct / g3_total if g3_total else 0
    print(f"\nResult: {g3_correct}/{g3_total} ({g3_acc:.0%}) in {g3_time:.0f}s, {g3_errors} errors\n")

    # --- Summary ---
    print("=" * 60)
    print("EXPERIMENT RESULTS: 三個臭皮匠 vs 諸葛亮")
    print("=" * 60)
    print(f"")
    print(f"  {'Group':40s} {'Accuracy':>10s} {'Time':>8s} {'Errors':>8s}")
    print(f"  {'-'*40} {'-'*10} {'-'*8} {'-'*8}")
    print(f"  {'Single Claude Sonnet 4.6':40s} {g1_acc:>9.0%} {g1_time:>7.0f}s {g1_errors:>8d}")
    print(f"  {'3x Free Models (vote)':40s} {g2_acc:>9.0%} {g2_time:>7.0f}s {g2_errors:>8d}")
    print(f"  {'3x Free Models (critique)':40s} {g3_acc:>9.0%} {g3_time:>7.0f}s {g3_errors:>8d}")
    print()

    if g3_acc > g1_acc:
        print("  >>> 三個臭皮匠 WIN! Critique mode beat the single strong model.")
    elif g3_acc == g1_acc:
        print("  >>> TIE. Critique mode matched the single strong model.")
    else:
        diff = g1_acc - g3_acc
        print(f"  >>> 諸葛亮 wins by {diff:.0%}. But check: did critique reduce errors vs vote?")
        if g3_acc > g2_acc:
            print(f"  >>> Critique improved over vote: {g2_acc:.0%} → {g3_acc:.0%} (+{g3_acc-g2_acc:.0%})")

    print()


if __name__ == "__main__":
    asyncio.run(main())
