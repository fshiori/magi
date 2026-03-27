"""Tests for magi judge command."""
from magi.commands.judge import build_judge_prompt, format_judge_output
from magi.core.decision import Decision


def test_build_judge_prompt():
    prompt = build_judge_prompt("What is 2+2?", "4")
    assert "What is 2+2?" in prompt
    assert "4" in prompt
    assert "Correctness" in prompt
    assert "1-10" in prompt


def test_build_judge_prompt_long_answer():
    long_answer = "x" * 5000
    prompt = build_judge_prompt("Q?", long_answer)
    assert long_answer in prompt


def test_format_judge_output():
    d = Decision(
        query="judge prompt",
        ruling="Correctness: 8/10\nOverall: 7/10",
        confidence=0.75,
        minority_report="[balthasar]: Overall: 6/10",
        votes={
            "melchior": "Correctness: 8/10\nOverall: 7/10",
            "balthasar": "Correctness: 7/10\nOverall: 6/10",
            "casper": "Correctness: 9/10\nOverall: 8/10",
        },
    )
    output = format_judge_output(d)
    assert "MAGI JUDGE" in output
    assert "MELCHIOR" in output
    assert "BALTHASAR" in output
    assert "CASPER" in output
    assert "75%" in output
