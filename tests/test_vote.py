"""Tests for VoteProtocol with structured positions."""
import pytest
from unittest.mock import AsyncMock
from magi.core.node import MagiNode, MELCHIOR, BALTHASAR, CASPER
from magi.protocols.vote import vote, _extract_position, _find_majority


def make_nodes():
    return [
        MagiNode("melchior", "mock-model", MELCHIOR),
        MagiNode("balthasar", "mock-model", BALTHASAR),
        MagiNode("casper", "mock-model", CASPER),
    ]


# --- Position extraction tests ---

class TestExtractPosition:
    def test_standard_format(self):
        assert _extract_position("POSITION: Yes, strongly agree\nBecause...") == "yes, strongly agree"

    def test_lowercase(self):
        assert _extract_position("position: no\nExplanation here") == "no"

    def test_mixed_case(self):
        assert _extract_position("Position: It depends on context\nDetails...") == "it depends on context"

    def test_no_tag_fallback(self):
        pos = _extract_position("The answer is definitely yes.\nMore details...")
        assert "the answer is definitely yes" in pos

    def test_position_not_on_first_line(self):
        text = "## Analysis\n\nPOSITION: Approve with conditions\nDetails..."
        assert _extract_position(text) == "approve with conditions"

    def test_empty_response(self):
        pos = _extract_position("")
        assert pos == ""


# --- Majority finding tests ---

class TestFindMajority:
    def test_clear_majority(self):
        positions = {"a": "yes", "b": "yes", "c": "no"}
        pos, majority, minority = _find_majority(positions)
        assert pos == "yes"
        assert set(majority) == {"a", "b"}
        assert minority == ["c"]

    def test_unanimous(self):
        positions = {"a": "yes", "b": "yes", "c": "yes"}
        pos, majority, minority = _find_majority(positions)
        assert pos == "yes"
        assert len(majority) == 3
        assert minority == []

    def test_no_majority(self):
        positions = {"a": "yes", "b": "no", "c": "maybe"}
        pos, majority, minority = _find_majority(positions)
        assert pos is None
        assert majority == []

    def test_two_nodes(self):
        positions = {"a": "yes", "b": "no"}
        pos, majority, minority = _find_majority(positions)
        assert pos is None  # 1:1 is not a majority

    def test_two_agree_out_of_two(self):
        positions = {"a": "yes", "b": "yes"}
        pos, majority, minority = _find_majority(positions)
        assert pos == "yes"

    def test_empty(self):
        pos, majority, minority = _find_majority({})
        assert pos is None


# --- Vote protocol tests ---

@pytest.mark.asyncio
async def test_vote_majority_found():
    """Two nodes agree on position, one disagrees."""
    nodes = make_nodes()
    nodes[0].query = AsyncMock(return_value="POSITION: Yes\nBecause of evidence A")
    nodes[1].query = AsyncMock(return_value="POSITION: Yes\nBecause of evidence B")
    nodes[2].query = AsyncMock(return_value="POSITION: No\nBecause of concern C")

    decision = await vote("Should we?", nodes)
    assert decision.protocol_used == "vote"
    assert decision.confidence == pytest.approx(2 / 3, abs=0.01)
    assert "No" in decision.minority_report or "concern C" in decision.minority_report


@pytest.mark.asyncio
async def test_vote_unanimous():
    nodes = make_nodes()
    for n in nodes:
        n.query = AsyncMock(return_value="POSITION: Approve\nLooks good to me.")

    decision = await vote("Review this?", nodes)
    assert decision.confidence == 1.0
    assert decision.minority_report == ""


@pytest.mark.asyncio
async def test_vote_no_majority_three_way_split():
    nodes = make_nodes()
    nodes[0].query = AsyncMock(return_value="POSITION: Yes\nReason A")
    nodes[1].query = AsyncMock(return_value="POSITION: No\nReason B")
    nodes[2].query = AsyncMock(return_value="POSITION: Maybe\nReason C")

    decision = await vote("Complex question", nodes)
    assert decision.protocol_used == "vote_no_majority"
    assert decision.confidence == pytest.approx(1 / 3, abs=0.01)


@pytest.mark.asyncio
async def test_vote_one_node_fails():
    nodes = make_nodes()
    nodes[0].query = AsyncMock(return_value="POSITION: Yes\nAnswer A")
    nodes[1].query = AsyncMock(return_value="POSITION: Yes\nAnswer B")
    nodes[2].query = AsyncMock(side_effect=TimeoutError("timeout"))

    decision = await vote("test", nodes)
    assert decision.degraded is True
    assert "casper" in decision.failed_nodes
    assert decision.confidence == 1.0  # 2/2 agree


@pytest.mark.asyncio
async def test_vote_two_nodes_fail():
    nodes = make_nodes()
    nodes[0].query = AsyncMock(return_value="POSITION: Yes\nOnly answer")
    nodes[1].query = AsyncMock(side_effect=TimeoutError("timeout"))
    nodes[2].query = AsyncMock(side_effect=RuntimeError("API error"))

    decision = await vote("test", nodes)
    assert decision.degraded is True
    assert decision.protocol_used == "fallback_single"
    assert decision.confidence == 0.3


@pytest.mark.asyncio
async def test_vote_all_nodes_fail():
    nodes = make_nodes()
    for n in nodes:
        n.query = AsyncMock(side_effect=TimeoutError("timeout"))

    with pytest.raises(RuntimeError, match="All MAGI nodes failed"):
        await vote("test", nodes)


@pytest.mark.asyncio
async def test_vote_without_position_tag():
    """Models that don't follow POSITION format still get parsed."""
    nodes = make_nodes()
    nodes[0].query = AsyncMock(return_value="I think yes, because of X")
    nodes[1].query = AsyncMock(return_value="I think yes, because of Y")
    nodes[2].query = AsyncMock(return_value="I think no, because of Z")

    decision = await vote("test", nodes)
    # First two have same first-line prefix "i think yes", should group
    assert len(decision.votes) == 3
