# Battle-Test Wiring: Phase 6C Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Wire the CompatShim to delegate real work (API calls, tool dispatch, streaming, system prompt building) to the existing AIAgent, so the new pipeline can run a real conversation end-to-end.

**Architecture:** "Wrap first, redesign later." The CompatShim receives a reference to the parent AIAgent at construction time. When the feature flag flips, the Orchestrator pipeline drives the loop, but each stage's hard work delegates to proven AIAgent methods via injected callables. This tests the pipeline coordination without reimplementing providers.

**Tech Stack:** Python 3.11+, hermes-agent dogfood branch, pytest

---

## Key Insight

The current CompatShim creates an Orchestrator with empty/stub components. The ProviderRegistry has no providers, the ToolExecutor has no real tools, and StreamingState has no real emitter. We need to inject real callables that bridge to AIAgent's existing methods:

1. **Provider call** → `AIAgent._interruptible_streaming_api_call()`  
2. **Tool dispatch** → `model_tools.handle_function_call()`  
3. **System prompt** → `AIAgent._build_system_prompt()`  
4. **Streaming emission** → `AIAgent.stream_delta_callback` / `_fire_stream_delta`

The CompatShim currently gets constructed in `AIAgent.__init__` (line ~1738) with only model/provider/session info. It needs the parent AIAgent reference to access those methods.

---

### Task 1: Add parent agent reference to AIAgentCompatShim

**Objective:** Pass the parent AIAgent instance to CompatShim so it can delegate real work.

**Files:**
- Modify: `agent/orchestrator/compat.py` (constructor + `_agent` attribute)
- Modify: `run_agent.py` (line ~1738, CompatShim construction)

**Step 1: Update CompatShim.__init__ to accept parent_agent**

In `agent/orchestrator/compat.py`, add `parent_agent` parameter:

```python
def __init__(
    self,
    parent_agent: Any = None,  # AIAgent instance — injected for delegation
    model: str = "",
    provider: str = "",
    base_url: str = "",
    api_key: str = "",
    max_iterations: Optional[int] = None,
    api_mode: str = "",
    session_id: str = "",
    registry: Optional[ProviderRegistry] = None,
    event_bus: Optional[EventBus] = None,
    tool_executor: Optional[ToolExecutor] = None,
    **kwargs,
) -> None:
    # Store parent for delegation
    self._agent = parent_agent
    ...
```

**Step 2: Update construction in run_agent.py**

In `run_agent.py` line ~1738-1748, pass `self`:

```python
if USE_NEW_PIPELINE:
    from agent.orchestrator.compat import AIAgentCompatShim
    self._new_pipeline = AIAgentCompatShim(
        parent_agent=self,
        model=model,
        provider=provider_name or provider or "",
        base_url=base_url or "",
        api_key=api_key or "",
        max_iterations=max_iterations,
        api_mode=getattr(self, "api_mode", "chat_completions"),
        session_id=session_id or "",
    )
```

**Step 3: Run orchestrator tests**

Run: `pytest tests/orchestrator/test_compat.py -v`
Expected: All existing tests pass (new param is optional, defaults to None)

**Step 4: Update compat tests to verify parent_agent wiring**

Add test that parent_agent is stored and accessible:

```python
def test_parent_agent_stored(self):
    mock_agent = MagicMock()
    shim = AIAgentCompatShim(parent_agent=mock_agent, model="test")
    assert shim._agent is mock_agent
```

**Step 5: Commit**

```bash
git add agent/orchestrator/compat.py run_agent.py tests/orchestrator/test_compat.py
git commit -m "orchestrator: pass parent AIAgent to CompatShim for delegation wiring"
```

---

### Task 2: Wire provider call through to AIAgent._interruptible_streaming_api_call

**Objective:** OpenAICompatibleProvider in the pipeline should call the real AIAgent method to make API calls.

**Files:**
- Modify: `agent/orchestrator/compat.py` (new method `_make_provider_call`)
- Modify: `agent/orchestrator/provider_adapters.py` (OpenAICompatibleProvider.execute with real call_fn)

**Step 1: Add _make_provider_call to CompatShim**

In `compat.py`, add a method that bridges to AIAgent's API call:

```python
def _make_provider_call(self, api_kwargs: dict) -> dict:
    """Delegate the API call to the parent AIAgent's proven path.
    
    This bridges ProviderCallStage.process() to the real 
    AIAgent._interruptible_streaming_api_call() or _interruptible_api_call().
    """
    if self._agent is None:
        raise RuntimeError("No parent AIAgent for provider call delegation")
    
    # Use the streaming path — it handles all api_modes.
    # The result is a SimpleNamespace that mimics OpenAI response shape.
    result = self._agent._interruptible_streaming_api_call(
        api_kwargs,
        on_first_delta=None,  # Streaming delta wiring comes in Task 4
    )
    return result
```

**Step 2: Wire this callable into OpenAICompatibleProvider**

The CompatShim's `_resolve_provider_name` already gets set on the orchestrator. Extend this pattern: after constructing the Orchestrator, inject a real call_fn into the OpenAICompatibleProvider that wraps `_make_provider_call`:

```python
# In CompatShim.__init__ or a wiring method:
from agent.orchestrator.provider_adapters import OpenAICompatibleProvider

# Register a provider that delegates to the parent AIAgent
provider = OpenAICompatibleProvider(
    base_url=self.base_url,
    model_name=self.model,
    call_fn=self._make_provider_call,
)
self._registry.register("openai_compatible", provider)
```

**Step 3: Update test_compat to verify provider registration**

```python
def test_provider_registered_on_init(self):
    shim = AIAgentCompatShim(parent_agent=MagicMock(), model="gpt-4", base_url="http://localhost:11434/v1")
    assert "openai_compatible" in shim._registry.list_names()

def test_provider_call_fn_wired(self):
    mock_agent = MagicMock()
    mock_agent._interruptible_streaming_api_call.return_value = MagicMock(choices=[])
    shim = AIAgentCompatShim(parent_agent=mock_agent, model="gpt-4")
    result = shim._make_provider_call({"model": "gpt-4", "messages": []})
    mock_agent._interruptible_streaming_api_call.assert_called_once()
```

**Step 4: Run tests**

Run: `pytest tests/orchestrator/test_compat.py tests/orchestrator/test_provider_adapters.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add agent/orchestrator/compat.py tests/orchestrator/test_compat.py
git commit -m "orchestrator: wire provider call to AIAgent._interruptible_streaming_api_call"
```

---

### Task 3: Wire tool dispatch to model_tools.handle_function_call

**Objective:** ToolExecutor in the pipeline dispatches tool calls through the real tool registry.

**Files:**
- Modify: `agent/orchestrator/compat.py` (new method `_dispatch_tool`)
- Modify: `agent/orchestrator/tools.py` (already has ToolExecutor.dispatch, accepts handler)

**Step 1: Add _dispatch_tool to CompatShim**

```python
def _dispatch_tool(self, name: str, args: dict) -> str:
    """Delegate tool dispatch to model_tools.handle_function_call.
    
    This bridges ToolExecutor to the real Hermes tool registry.
    """
    from model_tools import handle_function_call
    
    # Pass through the parent agent's session context
    task_id = getattr(self._agent, "task_id", None) if self._agent else None
    session_id = self.session_id
    
    result = handle_function_call(
        function_name=name,
        function_args=args,
        task_id=task_id,
        session_id=session_id,
    )
    return result
```

**Step 2: Wire into ToolExecutor**

Override the tool handler in the Orchestrator's ToolDispatchStage and ToolExecutor:

In `compat.py.__init__`, after creating the orchestrator, register the dispatch function.

The ToolExecutor already accepts a handler function at registration time. In the CompatShim, we need to register all tools from the parent AIAgent's tool definitions and set the dispatch handler:

```python
def _wire_tools(self) -> None:
    """Wire parent AIAgent's tool definitions into the ToolExecutor."""
    if self._agent is None or not hasattr(self._agent, "tools"):
        return
    
    # Register each tool name with the executor
    for tool_def in self._agent.tools:
        name = tool_def.get("function", {}).get("name", "")
        if name:
            self._tool_executor.register(name, handler=self._dispatch_tool)
```

Call `_wire_tools()` at the end of `__init__`.

**Step 3: Write tests**

```python
def test_tool_dispatch_delegates_to_handle_function_call(self):
    shim = AIAgentCompatShim(parent_agent=MagicMock(), model="test")
    # Mock handle_function_call at module level
    with patch("agent.orchestrator.compat.handle_function_call", return_value='{"result": "ok"}'):
        result = shim._dispatch_tool("terminal", {"command": "echo hi"})
        assert result == '{"result": "ok"}'
```

**Step 4: Run tests**

Run: `pytest tests/orchestrator/test_compat.py tests/orchestrator/test_tools.py -v`

**Step 5: Commit**

```bash
git add agent/orchestrator/compat.py tests/orchestrator/test_compat.py
git commit -m "orchestrator: wire tool dispatch to model_tools.handle_function_call"
```

---

### Task 4: Wire system prompt building to AIAgent._build_system_prompt

**Objective:** The pipeline's RequestPrepStage should use the real system prompt (with memory injection, vault, checkpoint, etc.) instead of a blank one.

**Files:**
- Modify: `agent/orchestrator/compat.py` (_sync_state_to_ctx, system prompt)
- Modify: `agent/orchestrator/stages.py` (RequestPrepStage — allow override)

**Step 1: Update _sync_state_to_ctx to use parent's system prompt**

```python
def _sync_state_to_ctx(self) -> None:
    """Copy SessionState fields + parent's system prompt into ConversationContext."""
    self._ctx.messages = self._state.messages
    self._ctx.session_id = self._state.session_id
    self._ctx.max_iterations = self.max_iterations
    self._ctx.iteration = 0
    
    # Build system prompt from the parent AIAgent's proven path
    if self._agent is not None and hasattr(self._agent, "_build_system_prompt"):
        system_prompt = self._agent._build_system_prompt()
        self._ctx.system_prompt = system_prompt or ""
    elif self._state.system_prompt:
        self._ctx.system_prompt = self._state.system_prompt
    else:
        self._ctx.system_prompt = ""
    
    # Wire tools from parent
    if self._agent is not None and hasattr(self._agent, "tools"):
        self._ctx.tools = self._agent.tools
```

**Step 2: Add system prompt caching to SessionState**

In `context.py`, `SessionState` already has `cached_system_prompt`. Update `_sync_state_to_ctx` to use it on subsequent turns.

**Step 3: Write test**

```python
def test_sync_state_uses_parent_system_prompt(self):
    mock_agent = MagicMock()
    mock_agent._build_system_prompt.return_value = "You are a helpful assistant."
    mock_agent.tools = [{"function": {"name": "test_tool"}, "type": "function"}]
    shim = AIAgentCompatShim(parent_agent=mock_agent, model="test")
    shim._sync_state_to_ctx()
    assert shim._ctx.system_prompt == "You are a helpful assistant."
    assert len(shim._ctx.tools) == 1
```

**Step 4: Run tests**

Run: `pytest tests/orchestrator/test_compat.py -v`

**Step 5: Commit**

```bash
git add agent/orchestrator/compat.py tests/orchestrator/test_compat.py
git commit -m "orchestrator: wire system prompt and tools from parent AIAgent"
```

---

### Task 5: Wire streaming delta emission to AIAgent.callbacks

**Objective:** Streaming tokens from the API call reach the gateway's stream_consumer via the existing callback path.

**Files:**
- Modify: `agent/orchestrator/compat.py` (EventBus subscription → AIAgent callbacks)
- Modify: `agent/orchestrator/events.py` (add stream delta event type)

**Step 1: Add STREAM_DELTA event type to EventBus**

In `events.py`, add to the PipelineEvent kinds that get emitted during streaming:

```python
# In PipelineEvent, the "kind" field already supports free-form strings.
# We'll use "stream_delta" and "stream_reasoning" as convention.
```

Actually, the EventBus is already generic — `PipelineEvent(kind="stream_delta", data={...})`. We just need to subscribe to it.

**Step 2: Add EventBus→callback bridge in CompatShim**

In `compat.py`, add a method that subscribes to the EventBus and forwards stream events to the parent AIAgent's callbacks:

```python
def _bridge_streaming(self) -> None:
    """Subscribe to EventBus stream events and forward to AIAgent callbacks."""
    def on_stream_delta(event: PipelineEvent) -> None:
        if event.kind == "stream_delta" and self._agent is not None:
            callback = getattr(self._agent, "stream_delta_callback", None)
            if callback:
                callback(event.data.get("delta", ""))
    
    def on_stream_reasoning(event: PipelineEvent) -> None:
        if event.kind == "stream_reasoning" and self._agent is not None:
            callback = getattr(self._agent, "_fire_reasoning_delta", None)
            if callback and callable(callback):
                callback(event.data.get("delta", ""))
    
    self._event_bus.subscribe("stream_delta", on_stream_delta)
    self._event_bus.subscribe("stream_reasoning", on_stream_reasoning)
```

**Step 3: Add subscribe method to EventBus**

Check if EventBus already has subscribe. If not, add it:

```python
def subscribe(self, kind: str, handler: Callable) -> None:
    """Subscribe to events of a given kind."""
    if kind not in self._subscribers:
        self._subscribers[kind] = []
    self._subscribers[kind].append(handler)

def emit(self, event: PipelineEvent) -> None:
    """Emit an event to all subscribers."""
    # Existing logging behavior
    self._log.append(event)
    # New: notify subscribers
    for handler in self._subscribers.get(event.kind, []):
        try:
            handler(event)
        except Exception:
            pass  # Don't let subscriber errors break the pipeline
```

**Step 4: Write test**

```python
def test_stream_delta_bridge(self):
    mock_agent = MagicMock()
    shim = AIAgentCompatShim(parent_agent=mock_agent, model="test")
    shim._bridge_streaming()
    
    # Emit a stream delta event
    shim._event_bus.emit(PipelineEvent(
        kind="stream_delta",
        data={"delta": "Hello"},
        session_id="test",
    ))
    
    mock_agent.stream_delta_callback.assert_called_once_with("Hello")
```

**Step 5: Run tests**

Run: `pytest tests/orchestrator/test_compat.py tests/orchestrator/test_stages.py -v`

**Step 6: Commit**

```bash
git add agent/orchestrator/compat.py agent/orchestrator/events.py tests/orchestrator/test_compat.py
git commit -m "orchestrator: wire streaming delta emission through EventBus to AIAgent callbacks"
```

---

### Task 6: Wire run_conversation to pass API kwargs from _build_api_kwargs

**Objective:** The ProviderCallStage gets proper API kwargs (model name, temperature, tools, etc.) built by the real AIAgent._build_api_kwargs method.

**Files:**
- Modify: `agent/orchestrator/compat.py` (override RequestPrepStage.process)
- Modify: `agent/orchestrator/stages.py` (make RequestPrepStage overridable)

**Step 1: Make RequestPrepStage accept an injected prepare_fn**

Allow RequestPrepStage to accept a callable that overrides its default message/api_kwargs building:

```python
class RequestPrepStage:
    def __init__(self, model_name: str = "default-model", prepare_fn=None) -> None:
        self._model_name = model_name
        self._prepare_fn = prepare_fn

    def process(self, ctx: ConversationContext) -> PreparedRequest:
        if self._prepare_fn is not None:
            return self._prepare_fn(ctx)
        # ... existing default logic ...
```

**Step 2: Inject AIAgent._build_api_kwargs through the CompatShim**

In `compat.py`:

```python
def _prepare_request(self, ctx: ConversationContext) -> PreparedRequest:
    """Build API kwargs using AIAgent._build_api_kwargs.
    
    Called by RequestPrepStage when _prepare_fn is set.
    """
    if self._agent is None:
        raise RuntimeError("No parent AIAgent for request preparation")
    
    # Use the parent's system prompt builder (already wired in _sync_state_to_ctx)
    # Call _build_api_kwargs with the context's messages
    api_kwargs = self._agent._build_api_kwargs(ctx.messages)
    
    return PreparedRequest(
        messages=ctx.messages,
        api_kwargs=api_kwargs,
        provider_name=self._resolve_provider_name(ctx),
    )
```

Then wire it in `__init__`:

```python
self._orchestrator = Orchestrator(
    ...,
    request_prep=RequestPrepStage(
        model_name=model,
        prepare_fn=self._prepare_request,
    ),
)
```

**Step 3: Write test**

```python
def test_prepare_request_uses_parent_build_api_kwargs(self):
    mock_agent = MagicMock()
    mock_agent._build_api_kwargs.return_value = {"model": "gpt-4", "temperature": 0.7}
    mock_agent._build_system_prompt.return_value = "You are helpful."
    mock_agent.tools = []
    
    shim = AIAgentCompatShim(parent_agent=mock_agent, model="gpt-4")
    ctx = ConversationContext(session_id="test", messages=[{"role": "user", "content": "hi"}])
    
    prepared = shim._prepare_request(ctx)
    assert prepared.api_kwargs["model"] == "gpt-4"
    assert prepared.provider_name is not None
```

**Step 4: Run tests**

Run: `pytest tests/orchestrator/test_compat.py tests/orchestrator/test_stages.py -v`

**Step 5: Commit**

```bash
git add agent/orchestrator/compat.py agent/orchestrator/stages.py tests/orchestrator/test_compat.py
git commit -m "orchestrator: wire RequestPrepStage to AIAgent._build_api_kwargs"
```

---

### Task 7: Wire response processing — map AIAgent streaming result to ParsedResponse

**Objective:** The OpenAI-compatible streaming result from AIAgent's API call needs to be converted to a ProviderResult that ResponseProcessingStage can parse.

**Files:**
- Modify: `agent/orchestrator/provider_adapters.py` (OpenAICompatibleProvider._default_parse_response)
- Modify: `agent/orchestrator/compat.py` (_make_provider_call returns structured result)

**Step 1: Understand the AIAgent streaming result shape**

AIAgent._interruptible_streaming_api_call returns a SimpleNamespace that mimics:
```python
SimpleNamespace(
    choices=[SimpleNamespace(
        message=SimpleNamespace(
            content="...",
            tool_calls=[...],
            reasoning_content=...,
        ),
        finish_reason="stop"
    )],
    usage=SimpleNamespace(prompt_tokens=..., completion_tokens=..., total_tokens=...),
)
```

This is already OpenAI-compatible, so the ResponseProcessingStage can parse it directly IF we wrap it correctly.

**Step 2: Update _make_provider_call to return a proper ProviderResult**

```python
def _make_provider_call(self, api_kwargs: dict) -> ProviderResult:
    """Delegate the API call to the parent AIAgent and wrap the result."""
    if self._agent is None:
        return ProviderResult(error=RuntimeError("No parent AIAgent"))
    
    try:
        raw_result = self._agent._interruptible_streaming_api_call(
            api_kwargs,
            on_first_delta=None,
        )
        
        # Convert SimpleNamespace to dict for ProviderResult
        # The result has .choices[].message etc. — same shape as OpenAI
        if raw_result is None:
            return ProviderResult(error=RuntimeError("API call returned None"))
        
        # Extract the core response dict
        response_dict = _namespace_to_dict(raw_result)
        
        return ProviderResult(
            response=response_dict,
            finish_reason=_get_finish_reason(raw_result),
        )
    except Exception as exc:
        logger.error(f"Provider call failed: {exc}")
        return ProviderResult(error=exc, should_fallback=True)
```

**Step 3: Write _namespace_to_dict helper**

```python
def _namespace_to_dict(obj) -> Any:
    """Recursively convert SimpleNamespace/dotted objects to dicts."""
    if isinstance(obj, (list, tuple)):
        return [_namespace_to_dict(item) for item in obj]
    if hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
        return {k: _namespace_to_dict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    return obj

def _get_finish_reason(result) -> str:
    """Extract finish_reason from a SimpleNamespace response."""
    try:
        return result.choices[0].finish_reason or "stop"
    except (AttributeError, IndexError):
        return "stop"
```

**Step 4: Write test**

```python
def test_make_provider_call_returns_provider_result(self):
    from types import SimpleNamespace
    
    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="Hello!", tool_calls=[]),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    mock_agent = MagicMock()
    mock_agent._interruptible_streaming_api_call.return_value = mock_response
    
    shim = AIAgentCompatShim(parent_agent=mock_agent, model="test")
    result = shim._make_provider_call({"model": "test"})
    
    assert result.error is None
    assert result.finish_reason == "stop"
    assert result.response["choices"][0]["message"]["content"] == "Hello!"
```

**Step 5: Run tests**

Run: `pytest tests/orchestrator/test_compat.py -v`

**Step 6: Commit**

```bash
git add agent/orchestrator/compat.py tests/orchestrator/test_compat.py
git commit -m "orchestrator: map AIAgent streaming result to ProviderResult"
```

---

### Task 8: Wire conversation message management and return format

**Objective:** The CompatShim's run_conversation returns a dict matching AIAgent's format, with proper message history, token counts, etc.

**Files:**
- Modify: `agent/orchestrator/compat.py` (_sync_ctx_to_state, run_conversation)

**Step 1: Update run_conversation to match AIAgent's return format**

AIAgent.run_conversation returns a dict with many fields. Let's check what the gateway/CLI actually depend on:

```python
# Key fields the gateway reads from the result:
# - final_response: str — the text the assistant said
# - messages: list[dict] — full conversation history
# - iterations: int — how many loop iterations
# - interrupted: bool — whether interrupt was requested
```

The current CompatShim already returns these. But we need to also populate:
- `total_tokens`, `input_tokens`, `output_tokens` — from UsageInfo
- `finish_reason` — from ParsedResponse
- Any additional fields the gateway reads

**Step 2: Sync token counts from the pipeline result**

```python
def _sync_ctx_to_state(self, result: PipelineResult) -> None:
    """Copy ConversationContext fields back to SessionState and self."""
    self._state.messages = list(self._ctx.messages)
    self._state.iteration_count = self._ctx.iteration
    self._api_call_count += result.iterations
    self._last_activity_ts = time.time()
    
    # Sync token counts from usage
    if result.response and result.response.usage:
        usage = result.response.usage
        self._state.total_tokens = getattr(usage, 'total_tokens', 0)
        self._state.input_tokens = getattr(usage, 'prompt_tokens', 0)
        self._state.output_tokens = getattr(usage, 'completion_tokens', 0)
```

**Step 3: Update run_conversation return dict**

```python
return {
    "final_response": final_response,
    "messages": list(self._ctx.messages),
    "iterations": result.iterations,
    "interrupted": result.interrupted,
    "finish_reason": result.response.finish_reason if result.response else None,
    "total_tokens": self._state.total_tokens,
    "input_tokens": self._state.input_tokens,
    "output_tokens": self._state.output_tokens,
}
```

**Step 4: Write test**

```python
def test_run_conversation_return_format(self):
    mock_agent = MagicMock()
    # Set up a minimal scenario where the pipeline returns a result
    ...
    result = shim.run_conversation("Hello")
    assert "final_response" in result
    assert "messages" in result
    assert "iterations" in result
    assert "total_tokens" in result
```

**Step 5: Run tests**

Run: `pytest tests/orchestrator/test_compat.py -v`

**Step 6: Commit**

```bash
git add agent/orchestrator/compat.py agent/orchestrator/context.py tests/orchestrator/test_compat.py
git commit -m "orchestrator: wire conversation message management and return format"
```

---

### Task 9: End-to-end smoke test — flip the flag and run a real conversation

**Objective:** Flip USE_NEW_PIPELINE=True on the Studio and verify a real conversation works.

**Files:**
- Modify: `run_agent.py` (line 145, flip flag)
- No new test files — this is an integration验证

**Step 1: Flip the flag on Studio**

```bash
ssh studio "cd ~/.hermes/hermes-agent && sed -i '' 's/USE_NEW_PIPELINE: bool = False/USE_NEW_PIPELINE: bool = True/' run_agent.py"
```

**Step 2: Restart the gateway on Studio**

```bash
ssh studio "launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway"
```

**Step 3: Send a test message through iMessage or CLI**

Wait for the gateway to come up, then trigger a conversation:
- Via iMessage to the Studio's agent number
- Or via CLI: `hermes --message "Hello, are you alive?"`

**Step 4: Monitor gateway logs**

```bash
ssh studio "tail -50 ~/.hermes/logs/gateway.log"
```

Look for:
- No crashes
- The pipeline processes the message
- A response comes back
- Token counts are reported

**Step 5: If it works — commit and merge**

```bash
git add run_agent.py
git commit -m "orchestrator: enable USE_NEW_PIPELINE for battle-testing on Studio"
```

**Step 6: If it fails — debug, fix, and re-test**

Examine the error log, fix in the relevant wiring module, and re-test.

---

### Task 10: Add logging and observability for the new pipeline

**Objective:** When USE_NEW_PIPELINE is True, emit structured logs that let us compare the new pipeline's behavior to the old one.

**Files:**
- Modify: `agent/orchestrator/compat.py` (add logging at delegation points)

**Step 1: Add logging at each delegation point**

```python
logger.info(f"[pipeline] run_conversation called: {len(self._ctx.messages)} messages in context")
logger.info(f"[pipeline] provider call delegated to AIAgent._interruptible_streaming_api_call")
logger.info(f"[pipeline] tool dispatch: {name}({list(args.keys())})")
logger.info(f"[pipeline] system prompt: {len(system_prompt)} chars")
logger.info(f"[pipeline] pipeline completed: {result.iterations} iterations, finish_reason={finish_reason}")
```

**Step 2: Add a pipeline identifier to make logs greppable**

All log messages use the prefix `[pipeline]` so we can `grep '\[pipeline\]' gateway.log` to see only new-pipeline events.

**Step 3: Commit**

```bash
git add agent/orchestrator/compat.py
git commit -m "orchestrator: add structured logging at delegation points"
```

---

## Post-Implementation Verification

After all tasks are complete:

1. **Unit tests pass**: `pytest tests/orchestrator/ -v` — all 465+ tests green
2. **Full test suite pass**: `pytest tests/ -q` — no regressions
3. **Studio smoke test**: Send a real message through the gateway, verify response
4. **Log inspection**: `grep '\[pipeline\]' ~/.hermes/logs/gateway.log` shows delegation events
5. **No crashes**: Gateway stays up for 24 hours with the new pipeline

## Rollback Plan

If the new pipeline breaks on Studio:
1. Flip `USE_NEW_PIPELINE = False` in `run_agent.py`
2. Restart the gateway: `launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway`
3. Investigate the error logs, fix, and re-test

The feature flag ensures zero-downtime rollback — the old path is completely untouched.