# Kore: Agent Core Replacement
#
# This package will eventually contain the modular agent loop and
# provider adapters. For now, it holds the configuration dataclasses
# that group AIAgent's 61 __init__ parameters into coherent objects.

from agent.kore.error_utils import (
    summarize_api_error,
    mask_api_key_for_logs,
    clean_error_message,
    extract_api_error_context,
    clean_session_content,
    dump_api_request_debug,
    usage_summary_for_api_request_hook,
)

from agent.kore.tool_calls import (
    get_tool_call_id_static,
    sanitize_api_messages,
    cap_delegate_task_calls,
    deduplicate_tool_calls,
    deterministic_call_id,
    split_responses_tool_id,
    sanitize_tool_calls_for_strict_api,
    sanitize_tool_call_arguments,
    VALID_API_ROLES,
    TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER,
)
