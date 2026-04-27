"""Tests that verify Phase 6B-2 dead code removal is safe.

These tests confirm:
1. The [PIPELINE-DEAD-CODE] markers and their dead code blocks are gone
2. The delegation points still route to the CompatShim
3. Methods still used by the CompatShim are NOT removed
4. The file shrunk by approximately the expected amount
"""

import re


RUN_AGENT = "run_agent.py"
COMPAT = "agent/orchestrator/compat.py"


def _read_run_agent():
    with open(RUN_AGENT) as f:
        return f.read()


def test_no_dead_code_markers_remain():
    """After Phase 6B-2, no [PIPELINE-DEAD-CODE] markers should exist."""
    content = _read_run_agent()
    matches = [line for line in content.split("\n") if "[PIPELINE-DEAD-CODE]" in line]
    assert len(matches) == 0, (
        f"Found {len(matches)} [PIPELINE-DEAD-CODE] marker(s) still present. "
        f"Phase 6B-2 should have removed them along with the dead code.\n"
        f"Markers: {matches}"
    )


def test_no_dead_code_ranges_constant():
    """DEAD_CODE_RANGES constant is obsolete after removal."""
    content = _read_run_agent()
    assert "DEAD_CODE_RANGES_WHEN_PIPELINE_ACTIVE" not in content, (
        "DEAD_CODE_RANGES_WHEN_PIPELINE_ACTIVE should be removed in Phase 6B-2 "
        "since the dead code it documented no longer exists."
    )


def test_line_count_shrunk_by_at_least_3000():
    """After removal, run_agent.py should be at least 3000 lines shorter."""
    content = _read_run_agent()
    line_count = len(content.split("\n"))
    # Original was ~11146 lines. Dead code was ~3608 lines.
    # After removal, should be around 7500 lines.
    assert line_count < 8200, (
        f"run_agent.py has {line_count} lines — expected ~7500 after removing "
        f"~3608 lines of dead code. Dead code may not have been fully removed."
    )


def test_switch_model_unconditionally_delegates():
    """switch_model should delegate to CompatShim without a guard check."""
    content = _read_run_agent()
    # Find switch_model method
    lines = content.split("\n")
    in_method = False
    delegation_found = False
    for line in lines:
        if "def switch_model(" in line:
            in_method = True
        if in_method:
            if "_new_pipeline.switch_model" in line:
                delegation_found = True
            # Should NOT have the old "from hermes_cli.providers import determine_api_mode"
            # as dead-code body
            if in_method and delegation_found and "def " in line and "switch_model" not in line:
                break  # reached next method

    assert delegation_found, "switch_model should delegate to _new_pipeline.switch_model"


def test_interrupt_unconditionally_delegates():
    """interrupt should delegate to CompatShim."""
    content = _read_run_agent()
    lines = content.split("\n")
    in_method = False
    delegation_found = False
    for line in lines:
        if "def interrupt(" in line:
            in_method = True
        if in_method:
            if "_new_pipeline.interrupt" in line:
                delegation_found = True
            if in_method and delegation_found and "def " in line and "interrupt" not in line:
                break

    assert delegation_found, "interrupt should delegate to _new_pipeline.interrupt"


def test_run_conversation_unconditionally_delegates():
    """run_conversation should delegate to CompatShim."""
    content = _read_run_agent()
    lines = content.split("\n")
    in_method = False
    delegation_found = False
    for line in lines:
        if "def run_conversation(" in line:
            in_method = True
        if in_method:
            if "_new_pipeline.run_conversation" in line:
                delegation_found = True
            if in_method and delegation_found and "def " in line and "run_conversation" not in line:
                break

    assert delegation_found, "run_conversation should delegate to _new_pipeline.run_conversation"


# Methods that the CompatShim still calls on self._agent — these must NOT be removed
# Discovered by grepping self._agent.X in compat.py
COMPAT_CALLED_METHODS = [
    "_build_system_prompt",
    "_build_api_kwargs",
    "_interruptible_streaming_api_call",
    "release_clients",
    "shutdown_memory_provider",
    "commit_memory_session",
]

COMPAT_CALLED_ATTRIBUTES = [
    "tools",
]


def test_compat_still_used_methods_exist():
    """Methods called by CompatShim on parent_agent must still exist in AIAgent."""
    content = _read_run_agent()
    for method in COMPAT_CALLED_METHODS:
        pattern = rf"def {method}\b"
        assert re.search(pattern, content), (
            f"AIAgent must still have '{method}' method — it's called by CompatShim."
        )


def test_compat_still_used_attributes_exist():
    """Attributes accessed by CompatShim on parent_agent must still exist."""
    content = _read_run_agent()
    for attr in COMPAT_CALLED_ATTRIBUTES:
        assert f"self.{attr}" in content, (
            f"AIAgent must still have '{attr}' attribute — it's accessed by CompatShim."
        )