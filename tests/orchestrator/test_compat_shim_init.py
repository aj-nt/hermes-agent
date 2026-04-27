"""Test that CompatShim is created at the correct indentation level.

This test catches the bug where removing the if USE_NEW_PIPELINE guard
left the CompatShim creation code inside the preceding if block,
causing it to only run for anthropic_messages mode.
"""
from __future__ import annotations

import pytest


def test_compat_shim_created_for_chat_completions_mode():
    """CompatShim must be created when api_mode is chat_completions."""
    from run_agent import AIAgent
    agent = AIAgent(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="glm-5.1:cloud",
        api_mode="chat_completions",
        enabled_toolsets=[],
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assert hasattr(agent, "_new_pipeline"), (
        "AIAgent must have _new_pipeline attribute after __init__. "
        "If this fails, the CompatShim creation code is inside an if block "
        "that only runs for anthropic_messages mode."
    )
    agent.release_clients()


def test_compat_shim_created_for_anthropic_mode():
    """CompatShim must also be created for anthropic_messages mode."""
    from run_agent import AIAgent
    agent = AIAgent(
        base_url="http://localhost:11434/v1",
        api_key="test",
        model="claude-3-5-sonnet-20241022",
        api_mode="anthropic_messages",
        enabled_toolsets=[],
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assert hasattr(agent, "_new_pipeline"), (
        "AIAgent must have _new_pipeline for anthropic_messages mode too."
    )
    agent.release_clients()