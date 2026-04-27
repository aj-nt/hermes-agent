"""Tests that [PIPELINE-DEAD-CODE] markers exist at known dead code boundaries.

Phase 6B-1: When USE_NEW_PIPELINE=True, the AIAgent methods switch_model(),
interrupt(), and run_conversation() all delegate to the CompatShim. The code
after the delegation point in each method is unreachable when the flag is on.

These tests verify that every dead code block has a [PIPELINE-DEAD-CODE]
marker comment, so Phase 6B-2 (full removal) is a simple grep operation.
"""

import re


RUN_AGENT = "run_agent.py"


def _read_run_agent():
    with open(RUN_AGENT) as f:
        return f.read()


def test_switch_model_has_dead_code_marker():
    """The old switch_model logic after the delegation point should be marked."""
    content = _read_run_agent()
    # Find the delegation block in switch_model
    lines = content.split("\n")
    in_delegation = False
    delegation_end = None
    for i, line in enumerate(lines):
        if "# --- Phase 6: New pipeline delegation ---" in line:
            # Check if we're in the switch_model method
            # Look back for "def switch_model"
            for j in range(i - 1, max(i - 30, 0), -1):
                if "def switch_model" in lines[j]:
                    in_delegation = True
                    break
        if in_delegation and "# --- End Phase 6 delegation ---" in line:
            delegation_end = i
            break

    assert delegation_end is not None, "Could not find end of switch_model delegation block"

    # The lines after delegation_end should contain a dead code marker
    after = "\n".join(lines[delegation_end + 1:delegation_end + 5])
    assert "[PIPELINE-DEAD-CODE]" in after, (
        f"Expected [PIPELINE-DEAD-CODE] marker after switch_model delegation. "
        f"Found: {after[:200]}"
    )


def test_interrupt_has_dead_code_marker():
    """The old interrupt logic after the delegation point should be marked."""
    content = _read_run_agent()
    lines = content.split("\n")
    delegation_end = None
    in_interrupt = False

    for i, line in enumerate(lines):
        if "def interrupt(" in line:
            in_interrupt = True
        if in_interrupt and "# --- End Phase 6 delegation ---" in line:
            delegation_end = i
            break

    assert delegation_end is not None, "Could not find end of interrupt delegation block"

    after = "\n".join(lines[delegation_end + 1:delegation_end + 5])
    assert "[PIPELINE-DEAD-CODE]" in after, (
        f"Expected [PIPELINE-DEAD-CODE] marker after interrupt delegation. "
        f"Found: {after[:200]}"
    )


def test_run_conversation_has_dead_code_marker():
    """The entire old agent loop after delegation should be marked."""
    content = _read_run_agent()
    lines = content.split("\n")
    delegation_end = None
    in_run_conversation = False

    for i, line in enumerate(lines):
        if "def run_conversation(" in line:
            in_run_conversation = True
        if in_run_conversation and "# --- End Phase 6 delegation ---" in line:
            delegation_end = i
            break

    assert delegation_end is not None, "Could not find end of run_conversation delegation block"

    after = "\n".join(lines[delegation_end + 1:delegation_end + 10])
    assert "[PIPELINE-DEAD-CODE]" in after, (
        f"Expected [PIPELINE-DEAD-CODE] marker after run_conversation delegation. "
        f"Found: {after[:200]}"
    )


def test_dead_code_ranges_constant_exists():
    """run_agent.py should document dead code ranges for Phase 6B-2 removal."""
    content = _read_run_agent()
    assert "DEAD_CODE_RANGES_WHEN_PIPELINE_ACTIVE" in content, (
        "Expected DEAD_CODE_RANGES_WHEN_PIPELINE_ACTIVE constant or comment "
        "documenting the line ranges for Phase 6B-2 removal."
    )


def test_dead_code_range_values_are_current():
    """DEAD_CODE_RANGES should reference valid line numbers and method names."""
    content = _read_run_agent()
    lines = content.split("\n")

    # Find the constant/comment block
    ranges_text = None
    for i, line in enumerate(lines):
        if "DEAD_CODE_RANGES_WHEN_PIPELINE_ACTIVE" in line:
            # Collect surrounding context (might be a dict literal or comment block)
            start = i
            end = min(i + 20, len(lines))
            ranges_text = "\n".join(lines[start:end])
            break

    assert ranges_text is not None, "DEAD_CODE_RANGES not found"

    # Should mention the three dead methods
    for method_name in ["switch_model", "interrupt", "run_conversation"]:
        assert method_name in ranges_text, (
            f"Expected '{method_name}' in DEAD_CODE_RANGES. Got: {ranges_text[:300]}"
        )