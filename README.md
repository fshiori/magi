# MAGI — Disagreement OS for LLMs

Three models. One decision. Inspired by the [MAGI supercomputer](https://evangelion.fandom.com/wiki/MAGI) from Neon Genesis Evangelion.

MAGI is not another agent framework. It is a **structured disagreement engine**: the same question goes to three different LLMs, each with a different perspective. They vote, debate, and critique each other to produce a Decision Dossier with the ruling, confidence, minority report, and full trace.

![NERV Command Center](docs/nerv-dashboard.png)

## Why?

> "Three cobblers with their wits combined equal Zhuge Liang, the master mind."

In our latest **MMLU (Massive Multitask Language Understanding)** "Hell Mode" benchmark:
- **Single Strong Model** (Claude 3.5 Sonnet): **66.7%**
- **MAGI Critique** (3x Cheap Models*): **80.0%** 🏆

*Models: Xiaomi Mimo-v2-pro, MiniMax-m2.7, DeepSeek-v3.2. Note: While individual models are cheaper per token, Critique Mode uses more rounds and context to achieve this higher reliability.*

The value is not just "accuracy." It is **better decision quality**: MAGI's critique mode allows specialized models to catch each other's hallucinations and logic gaps, outperforming a single "God-model" in complex domains like Abstract Algebra and Professional Law.

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
| **Benchmark: ensemble > single model** | No | Yes (80% > 66%) |
| **Fault tolerance (node failures)** | No | Yes |
| **NERV hexagonal dashboard** | No | Yes |
| **CLI toolchain (diff, judge, bench)** | No | Yes |

The key finding: **vote alone often hits a logic ceiling**. MAGI's critique mode breaks through it (80% vs 66%) by letting models catch each other's mistakes in high-stakes reasoning tasks.

A NeurIPS 2025 paper ([Debate or Vote](https://arxiv.org/abs/2508.17536)) found that "debate doesn't systematically improve beliefs." But their debate asks models to persuade humans. MAGI's ICE protocol asks models to find errors in each other's reasoning. Different mechanism, different result.

## Install

```bash
pip install magi-system
```

Or from source:

```bash
git clone https://github.com/fshiori/magi.git
cd magi
pip install -e ".[dev]"
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
magi bench
magi analytics
magi replay <trace-id>

# List persona presets
magi presets
```

## How It Works

```
  You ──▶ MAGI Engine ──▶ 3 LLMs in parallel ──▶ Protocol ──▶ Decision Dossier
                              │                       │
                         Melchior               Vote (fast)
                         Balthasar          Critique (debate)
                         Casper            Adaptive (auto)
```

Each Decision Dossier contains:

- **Ruling** — the final answer
- **Confidence** — how much the models agreed (0-100%)
- **Minority Report** — dissenting opinions and why they disagree
- **Mind Changes** — which models changed position during debate
- **Trace** — full JSONL history for replay and analytics

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

```bash
# Use a specific preset
magi ask "Should we expand to the EU market?" --preset strategy

# magi diff always uses code-review preset automatically
```

## Python API

```python
import asyncio
from magi import MAGI

engine = MAGI(
    melchior="openrouter/deepseek/deepseek-v3.2",
    balthasar="openrouter/xiaomi/mimo-v2-pro",
    casper="openrouter/minimax/minimax-m2.7",
)

decision = asyncio.run(engine.ask(
    "What are the security implications of this API design?",
    mode="adaptive",
))

print(decision.ruling)          # The final answer
print(decision.confidence)      # 0.0 - 1.0
print(decision.minority_report) # Dissenting views
print(decision.mind_changes)    # Who changed their mind
print(decision.protocol_used)   # Which protocol was selected
```

## Configuration

MAGI uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood, so it supports 100+ LLM providers.

### API Keys

```bash
# Direct providers
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=AI...

# Or use OpenRouter for all models with one key
export OPENROUTER_API_KEY=sk-or-...
```

### Using OpenRouter

```bash
magi ask "your question" \
  --melchior openrouter/anthropic/claude-sonnet-4.6 \
  --balthasar openrouter/openai/gpt-4o \
  --casper openrouter/google/gemini-2.5-pro
```

## Benchmark Results

Tested on **MMLU (Massive Multitask Language Understanding)** "Hell Mode" (Abstract Algebra, Professional Law, Formal Logic):

| Group | Accuracy | Strategy | Verdict |
|-------|----------|----------|---------|
| Claude 3.5 Sonnet (Single) | 66.7% | Single Shot | Fast, but misses subtle details |
| **MAGI Critique (3x Cheap)** | **80.0%** | **ICE Protocol** | **Winner.** Peer review fixes errors |

**Models used:** Xiaomi MiMo-v2-pro, MiniMax M2.7, DeepSeek V3.2.
**Judge:** Verified by Gemini 3.1 Pro via OpenRouter.

**Key finding:** Single strong models often fail on subtle multi-step reasoning. MAGI's **Critique Mode** breaks through this by letting models find errors in each other's reasoning, achieving higher accuracy at a fraction of the cost.

## Fault Tolerance

MAGI keeps working when models fail:

- **1 of 3 fails** — continues with 2 nodes, marks decision as degraded
- **2 of 3 fail** — falls back to single model response
- **All 3 fail** — raises `MagiUnavailableError` (never guesses)
- **Timeouts** — 30s default per node, exponential backoff on rate limits
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

## Development

```bash
git clone https://github.com/fshiori/magi.git
cd magi
uv venv && uv pip install -e ".[dev]"
python -m pytest tests/ -v
```

83 tests covering all protocols, degradation modes, and edge cases.

Published on [PyPI](https://pypi.org/project/magi-system/).

## NERV Command Center

Real-time dashboard showing the three MAGI nodes thinking, debating, and reaching a decision. EVA-accurate hexagonal layout with vote status lamps (承認/否決/膠着).

```bash
pip install magi-system[web]
magi dashboard
# Open http://localhost:3000
```

Features:
- Live WebSocket streaming of node responses
- Critique round tracking with agreement score
- EVA-style verdict display: 承認 (approve), 否決 (reject), 膠着 (deadlock)
- Click any hexagon to see the full response
- Auto-popup when nodes complete
- Markdown rendering for LLM output

## Roadmap

- [ ] MAGI-as-API-Gateway — OpenAI-compatible proxy, any app just changes `base_url`
- [ ] LLM-as-judge agreement scoring (replace word-overlap heuristic)
- [ ] Scorecard weighted voting (after sufficient data collection)
- [ ] Streaming token output in NERV UI

## Name

In Evangelion, MAGI is a trio of supercomputers created by Dr. Naoko Akagi. Each embodies a different aspect of her personality: **Melchior** (the scientist), **Balthasar** (the mother), and **Casper** (the woman). Decisions are made by majority vote among the three.

MAGI applies this concept to LLMs: same question, three different perspectives, structured disagreement produces better decisions than any single model alone.

## License

MIT
