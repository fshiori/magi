"""MAGI CLI — Disagreement OS for LLMs."""
import asyncio
import sys

import click

from magi.core.engine import MAGI
from magi.core.node import MELCHIOR, BALTHASAR, CASPER, AuthenticationError
from magi.commands.diff import (
    get_git_diff,
    check_diff_size,
    build_review_prompt,
    format_review_output,
)


@click.group()
@click.version_option(package_name="magi-system")
def main():
    """MAGI — Disagreement OS for LLMs. Three models, one decision."""
    pass


@main.command()
@click.argument("query")
@click.option("--mode", default="vote", type=click.Choice(["vote", "critique", "adaptive", "escalate"]))
@click.option("--melchior", default="claude-sonnet-4-6", help="Model for Melchior node")
@click.option("--balthasar", default="gpt-4o", help="Model for Balthasar node")
@click.option("--casper", default="gemini/gemini-2.5-pro", help="Model for Casper node")
def ask(query: str, mode: str, melchior: str, balthasar: str, casper: str):
    """Ask MAGI a question. Three models deliberate, one decision emerges."""
    engine = MAGI(melchior=melchior, balthasar=balthasar, casper=casper)
    try:
        decision = asyncio.run(engine.ask(query, mode=mode))
    except AuthenticationError as e:
        click.echo(f"Authentication error: {e}", err=True)
        sys.exit(1)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"\n{'=' * 60}")
    click.echo("MAGI DECISION")
    click.echo(f"{'=' * 60}")
    click.echo(f"\nRuling:\n{decision.ruling}")
    click.echo(f"\nConfidence: {decision.confidence:.0%}")
    click.echo(f"Protocol: {decision.protocol_used}")

    if decision.minority_report:
        click.echo(f"\nMinority Report:\n{decision.minority_report}")

    if decision.degraded:
        click.echo(f"\n⚠ Degraded: failed nodes = {', '.join(decision.failed_nodes)}")

    click.echo(f"\nTrace: {decision.trace_id} | Latency: {decision.latency_ms}ms")


@main.command()
@click.argument("file", required=False)
@click.option("--staged", is_flag=True, help="Review staged changes (git add first)")
@click.option("--melchior", default="claude-sonnet-4-6", help="Model for Melchior node")
@click.option("--balthasar", default="gpt-4o", help="Model for Balthasar node")
@click.option("--casper", default="gemini/gemini-2.5-pro", help="Model for Casper node")
def diff(file: str | None, staged: bool, melchior: str, balthasar: str, casper: str):
    """Multi-model code review. Three models review your diff independently."""
    if not file and not staged:
        staged = True  # default to --staged

    try:
        diff_text = get_git_diff(staged=staged, file_path=file)
        check_diff_size(diff_text)
    except (RuntimeError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Use code-review personas
    from magi.core.node import Persona
    code_personas = (
        Persona("Security Analyst", "You focus on security vulnerabilities, injection risks, auth issues, and data exposure."),
        Persona("Performance Engineer", "You focus on performance bottlenecks, N+1 queries, memory leaks, and scalability."),
        Persona("Code Quality Reviewer", "You focus on readability, maintainability, DRY violations, and best practices."),
    )

    engine = MAGI(
        melchior=melchior,
        balthasar=balthasar,
        casper=casper,
        personas=code_personas,
    )

    prompt = build_review_prompt(diff_text)

    try:
        decision = asyncio.run(engine.ask(prompt, mode="vote"))
    except AuthenticationError as e:
        click.echo(f"Authentication error: {e}", err=True)
        sys.exit(1)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(format_review_output(decision))


@main.command()
@click.option("--question", "-q", required=True, help="The question that was asked")
@click.option("--answer", "-a", required=True, help="The answer to evaluate")
@click.option("--melchior", default="claude-sonnet-4-6", help="Model for Melchior node")
@click.option("--balthasar", default="gpt-4o", help="Model for Balthasar node")
@click.option("--casper", default="gemini/gemini-2.5-pro", help="Model for Casper node")
def judge(question: str, answer: str, melchior: str, balthasar: str, casper: str):
    """Multi-model answer scoring. Three models rate a Q&A pair."""
    from magi.commands.judge import build_judge_prompt, format_judge_output

    engine = MAGI(melchior=melchior, balthasar=balthasar, casper=casper)
    prompt = build_judge_prompt(question, answer)

    try:
        decision = asyncio.run(engine.ask(prompt, mode="vote"))
    except AuthenticationError as e:
        click.echo(f"Authentication error: {e}", err=True)
        sys.exit(1)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(format_judge_output(decision))


if __name__ == "__main__":
    main()
