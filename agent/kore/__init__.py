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
    repair_tool_call,
    VALID_API_ROLES,
    TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER,
)

from agent.kore.responses_api import (
    model_requires_responses_api,
    provider_model_requires_responses_api,
)

from agent.kore.vision_utils import (
    content_has_image_parts,
    materialize_data_url_for_vision,
)

from agent.kore.client_lifecycle import (
    is_openai_client_closed,
    build_keepalive_http_client,
    force_close_tcp_sockets,
    cleanup_dead_connections,
)

from agent.kore.display_utils import (
    summarize_background_review_actions,
    wrap_verbose,
    normalize_interim_visible_text,
)

from agent.kore.reasoning import (
    resolved_api_call_stale_timeout_base,
    supports_reasoning_extra_body,
    github_models_reasoning_extra_body,
    needs_kimi_tool_reasoning,
    needs_deepseek_tool_reasoning,
    copy_reasoning_content_for_api,
)

from agent.kore.url_helpers import (
    is_direct_openai_url,
    is_azure_openai_url,
    is_openrouter_url,
    is_qwen_portal,
    max_tokens_param,
    anthropic_preserve_dots,
    should_sanitize_tool_calls,
    anthropic_prompt_cache_policy,
)

from agent.kore.think_blocks import (
    looks_like_codex_intermediate_ack,
)

from agent.kore.message_prep import (
    qwen_prepare_chat_messages,
    qwen_prepare_chat_messages_inplace,
)

from agent.kore.reasoning import (
    extract_reasoning,
    get_messages_up_to_last_assistant,
)


from agent.kore.tdd_gate import (
    TddGateTracker,
    is_tdd_gated_path,
    build_nudge,
)

from agent.kore.steer import (
    apply_steer_to_tool_results,
)
