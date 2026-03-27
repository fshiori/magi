"""Tests for magi analytics."""
import json
import os
import tempfile
import pytest
from magi.commands.analytics import load_traces, analyze, format_analytics, format_replay


SAMPLE_TRACES = [
    {
        "query": "What is 1+1?",
        "ruling": "2",
        "confidence": 0.9,
        "minority_report": "",
        "votes": {"melchior": "2", "balthasar": "2", "casper": "2"},
        "mind_changes": [],
        "protocol_used": "vote",
        "degraded": False,
        "failed_nodes": [],
        "latency_ms": 1500,
        "cost_usd": 0.01,
        "trace_id": "abc12345",
    },
    {
        "query": "Should we use microservices?",
        "ruling": "It depends on scale",
        "confidence": 0.5,
        "minority_report": "[casper]: No, monolith first",
        "votes": {
            "melchior": "It depends on scale",
            "balthasar": "It depends on the team",
            "casper": "No, monolith first",
        },
        "mind_changes": ["balthasar"],
        "protocol_used": "critique_ice_r2",
        "degraded": False,
        "failed_nodes": [],
        "latency_ms": 5000,
        "cost_usd": 0.05,
        "trace_id": "def67890",
    },
    {
        "query": "Is this code secure?",
        "ruling": "Mostly yes",
        "confidence": 0.7,
        "minority_report": "",
        "votes": {"melchior": "Mostly yes", "balthasar": "Mostly yes"},
        "mind_changes": [],
        "protocol_used": "vote",
        "degraded": True,
        "failed_nodes": ["casper"],
        "latency_ms": 2000,
        "cost_usd": 0.008,
        "trace_id": "ghi11111",
    },
]


@pytest.fixture
def trace_dir():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "2026-03-27.jsonl")
        with open(path, "w") as f:
            for t in SAMPLE_TRACES:
                f.write(json.dumps(t) + "\n")
        yield d


def test_load_traces(trace_dir):
    traces = load_traces(trace_dir)
    assert len(traces) == 3


def test_load_traces_empty():
    with tempfile.TemporaryDirectory() as d:
        traces = load_traces(d)
        assert traces == []


def test_load_traces_nonexistent():
    traces = load_traces("/nonexistent/path")
    assert traces == []


def test_analyze(trace_dir):
    traces = load_traces(trace_dir)
    report = analyze(traces)
    assert report.total_decisions == 3
    assert report.by_protocol["vote"] == 2
    assert report.by_protocol["critique_ice_r2"] == 1
    assert report.degraded_count == 1
    assert report.mind_change_count == 1
    assert report.avg_confidence == pytest.approx(0.7, abs=0.01)
    assert report.total_cost_usd == pytest.approx(0.068, abs=0.001)


def test_analyze_empty():
    report = analyze([])
    assert report.total_decisions == 0
    assert report.avg_confidence == 0.0


def test_analyze_per_node(trace_dir):
    traces = load_traces(trace_dir)
    report = analyze(traces)
    assert "melchior" in report.by_node
    assert report.by_node["melchior"]["total"] == 3
    assert report.by_node["casper"]["failed"] == 1


def test_format_analytics(trace_dir):
    traces = load_traces(trace_dir)
    report = analyze(traces)
    output = format_analytics(report)
    assert "MAGI ANALYTICS" in output
    assert "Total decisions: 3" in output
    assert "PROTOCOL DISTRIBUTION" in output
    assert "PER-NODE STATS" in output


def test_format_analytics_empty():
    report = analyze([])
    output = format_analytics(report)
    assert "No trace data" in output


def test_format_replay():
    output = format_replay(SAMPLE_TRACES[1])
    assert "DECISION REPLAY" in output
    assert "def67890" in output
    assert "microservices" in output
    assert "MIND CHANGES" in output
    assert "balthasar" in output
