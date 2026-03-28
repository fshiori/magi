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
@click.option("--preset", default="eva", help="Persona preset (eva, code-review, research, writing, strategy)")
@click.option("--melchior", default="claude-sonnet-4-6", help="Model for Melchior node")
@click.option("--balthasar", default="gpt-4o", help="Model for Balthasar node")
@click.option("--casper", default="gemini/gemini-2.5-pro", help="Model for Casper node")
def ask(query: str, mode: str, preset: str, melchior: str, balthasar: str, casper: str):
    """Ask MAGI a question. Three models deliberate, one decision emerges."""
    from magi.presets import get_preset
    try:
        personas = get_preset(preset)
    except KeyError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    engine = MAGI(melchior=melchior, balthasar=balthasar, casper=casper, personas=personas)
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

    from magi.presets import get_preset
    engine = MAGI(
        melchior=melchior,
        balthasar=balthasar,
        casper=casper,
        personas=get_preset("code-review"),
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


@main.command()
def presets():
    """List available persona presets."""
    from magi.presets import PRESETS
    for name, personas in sorted(PRESETS.items()):
        names = " / ".join(p.name for p in personas)
        click.echo(f"  {name:15s} {names}")


@main.command()
@click.option("--dataset", default="builtin", help="Dataset to benchmark against")
@click.option("--mode", default="vote", type=click.Choice(["vote", "critique", "adaptive"]))
@click.option("--concurrency", default=3, help="Max concurrent questions")
@click.option("--melchior", default="claude-sonnet-4-6", help="Model for Melchior node")
@click.option("--balthasar", default="gpt-4o", help="Model for Balthasar node")
@click.option("--casper", default="gemini/gemini-2.5-pro", help="Model for Casper node")
def bench(dataset: str, mode: str, concurrency: int, melchior: str, balthasar: str, casper: str):
    """Run benchmark: MAGI vs individual models on multiple-choice questions."""
    from magi.bench.datasets import get_dataset
    from magi.bench.runner import run_benchmark
    from magi.bench.report import format_report

    try:
        questions = get_dataset(dataset)
    except ValueError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    engine = MAGI(melchior=melchior, balthasar=balthasar, casper=casper)
    click.echo(f"Running benchmark: {len(questions)} questions, mode={mode}, concurrency={concurrency}")
    click.echo(f"Models: {melchior} / {balthasar} / {casper}")
    click.echo("")

    try:
        report = asyncio.run(run_benchmark(engine, questions, mode=mode, concurrency=concurrency))
    except Exception as e:
        click.echo(f"Benchmark failed: {e}", err=True)
        sys.exit(1)

    click.echo(format_report(report))


@main.command()
@click.option("--trace-dir", default=None, help="Trace directory (default: ~/.magi/traces)")
def analytics(trace_dir: str | None):
    """Analyze decision history from traces."""
    import os
    from magi.commands.analytics import analyze_stream, format_analytics

    trace_dir = trace_dir or os.path.expanduser("~/.magi/traces")
    report = analyze_stream(trace_dir)
    click.echo(format_analytics(report))


@main.command()
@click.argument("trace_id")
@click.option("--trace-dir", default=None, help="Trace directory (default: ~/.magi/traces)")
def replay(trace_id: str, trace_dir: str | None):
    """Replay a specific decision by trace ID."""
    import os
    from magi.commands.analytics import load_traces, format_replay

    trace_dir = trace_dir or os.path.expanduser("~/.magi/traces")
    traces = load_traces(trace_dir)

    match = [t for t in traces if t.get("trace_id", "").startswith(trace_id)]
    if not match:
        click.echo(f"No decision found with trace ID starting with '{trace_id}'", err=True)
        sys.exit(1)
    if len(match) > 1:
        click.echo(f"Multiple matches for '{trace_id}'. Be more specific.", err=True)
        for t in match:
            click.echo(f"  {t['trace_id']} — {t.get('query', 'N/A')[:60]}", err=True)
        sys.exit(1)

    click.echo(format_replay(match[0]))


@main.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=3000, help="Server port")
def dashboard(host: str, port: int):
    """Launch NERV Command Center — real-time MAGI visualization."""
    try:
        from magi.web.server import start_server
    except ImportError:
        click.echo("Web dependencies not installed. Run: pip install magi-system[web]", err=True)
        sys.exit(1)
    click.echo(f"NERV Command Center starting at http://localhost:{port}")
    start_server(host=host, port=port)


if __name__ == "__main__":
    main()
