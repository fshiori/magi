# MAGI — Disagreement OS for LLMs

Three models. One decision. Inspired by the [MAGI supercomputer](https://evangelion.fandom.com/wiki/MAGI) from Neon Genesis Evangelion.

MAGI is not another agent framework. It is a **structured disagreement engine**: the same question goes to three different LLMs, each with a different perspective. They vote, debate, and critique each other to produce a Decision Dossier with the ruling, confidence, minority report, and full trace.

![NERV Command Center](docs/nerv-dashboard.png)

## Why?

> "Three cobblers with their wits combined equal Zhuge Liang, the master mind."

In our latest **MMLU (Massive Multitask Language Understanding)** "Hell Mode" benchmark:
- **Single Strong Model** (Claude Sonnet 4.6): **83.3%**
- **MAGI Critique** (3x Cheap Models*): **83.3%** 🤝 **(MATCHED)**

*Models: Xiaomi Mimo-v2-pro, MiniMax-m2.7, DeepSeek-v3.2. Note: MAGI achieves the same accuracy as the world's leading model by leveraging collective intelligence through multi-round debate.*

The value: MAGI allows you to achieve **State-of-the-Art (SOTA) performance** using significantly smaller/cheaper nodes by enabling them to catch each other's logic gaps and hallucinations.

## How MAGI Differs

There are several EVA-inspired multi-model projects. Here's what makes this one different.

**Other projects do voting.** Three models answer, pick the majority. That's it.

**MAGI does structured disagreement.** Models don't just answer in parallel. They read each other's answers, critique the reasoning, and revise their positions across multiple rounds. The system tracks who changed their mind and why.

| Capability | Voting projects | MAGI |
|---|---|---|
| Multi-model query | Yes | Yes |
| Majority vote | Yes | Yes |
| **Multi-round critique (ICE)** | No | Yes |
| **Mind change tracking** | No | Yes |
| **Adaptive protocol selection** | No | Yes |
| **Minority report / dissent analysis** | No | Yes |
| **Benchmark: ensemble >= single model** | No | Yes (83% vs 83%) |
| **Fault tolerance (node failures)** | No | Yes |
| **NERV hexagonal dashboard** | No | Yes |
| **CLI toolchain (diff, judge, bench)** | No | Yes |

The key finding: **MAGI enables cheap models to reach the "Sonnet Ceiling"**. While single strong models are elite, MAGI's critique mode allows an ensemble of much cheaper models to reach the same level of accuracy (83.3%) by catching mistakes in high-stakes reasoning tasks.

## Install

```bash
pip install magi-system
```

Or from source:

```bash
git clone https://github.com/fshiori/magi.git
cd magi
uv venv && uv pip install -e ".[dev]"
```

## Quick Start

```bash
# Set your API key (OpenRouter gives you access to all models with one key)
export OPENROUTER_API_KEY=sk-or-...

# Ask a question — three models debate, one decision emerges
magi ask "Should we use microservices or a monolith?"

# Multi-model code review (the killer use case)
magi diff --staged

# Critique mode: models debate until consensus (slower, higher quality)
magi ask "Is Rust better than Go for backend services?" --mode critique

# Adaptive mode: auto-selects vote/critique/escalate based on disagreement
magi ask "What caused the 2008 financial crisis?" --mode adaptive

# Multi-model answer scoring
magi judge -q "What is quantum entanglement?" -a "It means particles are connected"

# NERV Command Center — real-time dashboard
pip install magi-system[web]
magi dashboard

# Run benchmark, view analytics, replay decisions
magi bench --dataset mmlu:abstract_algebra --use-judge
magi analytics
magi replay <trace-id>

# List persona presets
magi presets
```

## Protocols

| Protocol | When to use | How it works |
|----------|-------------|--------------|
| `vote` | Fast answers, clear-cut questions | Parallel query, structured position extraction, majority wins |
| `critique` | Complex or controversial questions | Multi-round debate (ICE), models critique each other until consensus |
| `escalate` | Forced decision on high-disagreement topics | Critique with 2-round limit, highest-trust node makes final call |
| `adaptive` | Default for most use cases | Auto-selects based on agreement score: high=vote, medium=critique, low=escalate |

## Persona Presets

MAGI comes with 5 built-in perspective sets:

```
$ magi presets

  code-review     Security Analyst / Performance Engineer / Code Quality Reviewer
  eva             Melchior / Balthasar / Casper
  research        Methodologist / Domain Expert / Devil's Advocate
  strategy        Optimist / Pessimist / Pragmatist
  writing         Editor / Reader Advocate / Fact Checker
```

## Benchmark Results

Tested on **MMLU (Massive Multitask Language Understanding)** "Hell Mode" (Abstract Algebra, Professional Law, Formal Logic):

| Group | Accuracy | Strategy | Verdict |
|-------|----------|----------|---------|
| Claude Sonnet 4.6 (Single) | 83.3% | Single Shot | Peak individual performance |
| **MAGI Critique (3x Cheap)** | **83.3%** | **ICE Protocol** | **Matched.** Beats Sonnet on Logic |

**Models used:** Xiaomi MiMo-v2-pro, MiniMax M2.7, DeepSeek V3.2.
**Judge:** Verified by Gemini 3.1 Pro via OpenRouter.

## Fault Tolerance

MAGI keeps working when models fail:

- **1 of 3 fails** — continues with 2 nodes, marks decision as degraded
- **2 of 3 fail** — falls back to single model response
- **All 3 fail** — raises `MagiUnavailableError` (never guesses)
- **Timeouts** — 60s default per node, exponential backoff on rate limits
- **Reasoning models** — automatically extracts from `reasoning_content` (e.g., MiniMax M2.7)

## Project Structure

```
magi/
├── core/
│   ├── engine.py       # MAGI engine, coordinates nodes
│   ├── node.py         # LLM node wrapper with persona
│   └── decision.py     # Decision dossier dataclass
├── protocols/
│   ├── vote.py         # Structured voting with position extraction
│   ├── critique.py     # ICE (Iterative Consensus Ensemble)
│   └── adaptive.py     # Dynamic protocol selection
├── commands/
│   ├── diff.py         # Multi-model code review
│   ├── judge.py        # Multi-model answer scoring
│   └── analytics.py    # Trace analysis and replay
├── web/
│   ├── server.py       # FastAPI + WebSocket server
│   └── static/         # NERV Command Center UI
├── presets/             # Persona preset definitions
├── bench/              # Benchmark runner and datasets
├── trace/              # JSONL trace logging
└── cli.py              # Click CLI entry point
```

## Name

In Evangelion, MAGI is a trio of supercomputers created by Dr. Naoko Akagi. Each embodies a different aspect of her personality: **Melchior** (the scientist), **Balthasar** (the mother), and **Casper** (the woman). Decisions are made by majority vote among the three.

MAGI applies this concept to LLMs: same question, three different perspectives, structured disagreement produces better decisions than any single model alone.

## License

MIT
