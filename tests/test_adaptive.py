"""Tests for AdaptiveProtocol."""
import pytest
from unittest.mock import AsyncMock
from magi.core.node import MagiNode, MELCHIOR, BALTHASAR, CASPER
from magi.protocols.adaptive import adaptive


def make_nodes():
    return [
        MagiNode("melchior", "mock", MELCHIOR),
        MagiNode("balthasar", "mock", BALTHASAR),
        MagiNode("casper", "mock", CASPER),
    ]


@pytest.mark.asyncio
async def test_adaptive_high_agreement_fast_path():
    """High agreement → vote path, no extra LLM calls."""
    nodes = make_nodes()
    # All give nearly identical answers (high word overlap needed for >0.8)
    nodes[0].query = AsyncMock(return_value="the answer is yes because of the strong evidence")
    nodes[1].query = AsyncMock(return_value="the answer is yes because of the strong evidence presented")
    nodes[2].query = AsyncMock(return_value="the answer is yes because of the strong evidence here")

    decision = await adaptive("test", nodes)
    assert "adaptive_vote" in decision.protocol_used
    # Each node called only once (no critique rounds)
    for n in nodes:
        assert n.query.call_count == 1


@pytest.mark.asyncio
async def test_adaptive_medium_disagreement_critique():
    """Medium disagreement → routes to critique."""
    nodes = make_nodes()
    call_counts = {"melchior": 0, "balthasar": 0, "casper": 0}

    for n in nodes:
        name = n.name
        async def make_mock(node_name=name):
            async def _query(prompt):
                call_counts[node_name] += 1
                if call_counts[node_name] == 1:
                    # Moderate disagreement
                    return {
                        "melchior": "the primary reason is economic growth and market dynamics",
                        "balthasar": "the primary concern is social welfare and human impact",
                        "casper": "the practical consideration is resource allocation efficiency",
                    }[node_name]
                else:
                    return "after review we agree on balanced economic and social approach"
            return _query
        n.query = await make_mock()

    decision = await adaptive("test", nodes, threshold_high=0.9, threshold_low=0.1)
    assert "adaptive_critique" in decision.protocol_used


@pytest.mark.asyncio
async def test_adaptive_severe_disagreement_escalate():
    """Severe disagreement → escalate (critique with max 2 rounds)."""
    nodes = make_nodes()
    call_counts = {"melchior": 0, "balthasar": 0, "casper": 0}

    for n in nodes:
        name = n.name
        async def make_mock(node_name=name):
            async def _query(prompt):
                call_counts[node_name] += 1
                # Always completely different
                return f"unique perspective {node_name} call {call_counts[node_name]} with entirely different vocabulary"
            return _query
        n.query = await make_mock()

    decision = await adaptive("test", nodes, threshold_high=0.99, threshold_low=0.98)
    assert "adaptive_escalate" in decision.protocol_used


@pytest.mark.asyncio
async def test_adaptive_all_nodes_fail():
    nodes = make_nodes()
    for n in nodes:
        n.query = AsyncMock(side_effect=RuntimeError("dead"))
    with pytest.raises(RuntimeError, match="All MAGI nodes failed"):
        await adaptive("test", nodes)


@pytest.mark.asyncio
async def test_adaptive_single_node_fallback():
    nodes = make_nodes()
    nodes[0].query = AsyncMock(return_value="only survivor")
    nodes[1].query = AsyncMock(side_effect=TimeoutError())
    nodes[2].query = AsyncMock(side_effect=TimeoutError())

    decision = await adaptive("test", nodes)
    assert decision.degraded is True
    assert "fallback_single" in decision.protocol_used
