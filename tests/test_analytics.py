"""Tests for magi analytics."""
import json
import os
import tempfile
import pytest
from magi.commands.analytics import (
    load_traces, analyze, analyze_stream, format_analytics, format_replay,
    NodeStats, _safe_float, _safe_str,
)


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


# --- Safe helpers ---

def test_safe_float_normal():
    assert _safe_float(0.5) == 0.5
    assert _safe_float(42) == 42.0


def test_safe_float_string():
    assert _safe_float("0.8") == 0.8


def test_safe_float_invalid():
    assert _safe_float("not a number") == 0.0
    assert _safe_float(None) == 0.0
    assert _safe_float([1, 2]) == 0.0


def test_safe_str():
    assert _safe_str("hello") == "hello"
    assert _safe_str(None) == ""
    assert _safe_str(42) == "42"


# --- Load traces ---

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


def test_load_traces_skips_non_dict():
    """Non-dict JSON lines should be skipped."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.jsonl")
        with open(path, "w") as f:
            f.write('"just a string"\n')
            f.write("[1, 2, 3]\n")
            f.write(json.dumps(SAMPLE_TRACES[0]) + "\n")
            f.write("not json at all\n")
        traces = load_traces(d)
        assert len(traces) == 1  # only the valid dict


def test_load_traces_skips_malformed_json():
    """Malformed JSON lines should be skipped."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.jsonl")
        with open(path, "w") as f:
            f.write("{broken json\n")
            f.write(json.dumps(SAMPLE_TRACES[0]) + "\n")
        traces = load_traces(d)
        assert len(traces) == 1


# --- Analyze ---

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
    assert report.by_node["melchior"].total == 3
    assert report.by_node["casper"].failed == 1


def test_analyze_disagreed_with_ruling(trace_dir):
    """Nodes whose answer differs from ruling are counted as disagreed."""
    traces = load_traces(trace_dir)
    report = analyze(traces)
    # In trace 2, balthasar and casper disagree with ruling
    assert report.by_node["balthasar"].disagreed_with_ruling >= 1
    assert report.by_node["casper"].disagreed_with_ruling >= 1


def test_analyze_skips_non_dict():
    """Non-dict entries in traces list are skipped."""
    report = analyze(["not a dict", 42, None, SAMPLE_TRACES[0]])
    assert report.total_decisions == 1
    assert report.skipped_lines == 3


def test_analyze_handles_string_confidence():
    """String confidence values should be safely converted."""
    trace = dict(SAMPLE_TRACES[0])
    trace["confidence"] = "0.8"
    report = analyze([trace])
    assert report.avg_confidence == pytest.approx(0.8)


# --- Streaming analyze ---

def test_analyze_stream(trace_dir):
    report = analyze_stream(trace_dir)
    assert report.total_decisions == 3
    assert report.by_protocol["vote"] == 2


def test_analyze_stream_with_malformed(trace_dir):
    """Streaming analysis counts skipped lines."""
    path = os.path.join(trace_dir, "bad.jsonl")
    with open(path, "w") as f:
        f.write("{broken\n")
        f.write('"just a string"\n')
    report = analyze_stream(trace_dir)
    assert report.total_decisions == 3  # from the good file
    assert report.skipped_lines == 2


# --- Formatting ---

def test_format_analytics(trace_dir):
    traces = load_traces(trace_dir)
    report = analyze(traces)
    output = format_analytics(report)
    assert "MAGI ANALYTICS" in output
    assert "Total decisions: 3" in output
    assert "PROTOCOL DISTRIBUTION" in output
    assert "PER-NODE STATS" in output


def test_format_analytics_shows_skipped():
    report = analyze(["bad", SAMPLE_TRACES[0]])
    output = format_analytics(report)
    assert "Skipped" in output or "skipped" in output.lower()


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


def test_format_replay_handles_none_values():
    """Replay should not crash on None or missing fields."""
    trace = {"trace_id": "test123"}
    output = format_replay(trace)
    assert "test123" in output
    assert "N/A" in output
