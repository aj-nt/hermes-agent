# Phase 7: Memory Coordination

## Goal
Extract all memory operations from `run_agent.py` into `MemoryCoordinator`
(`agent/orchestrator/memory.py`), so the CompatShim can route memory calls
through the orchestrator pipeline instead of reaching back into AIAgent.

## Why
Memory logic is currently scattered across ~15 call sites in run_agent.py:
- `__init__`: MemoryStore + MemoryManager initialization (L1252-1339)
- `_build_system_prompt`: memory blocks injection (L3131-3200)
- `handle_function_call`: memory tool routing (L6255-6273, L6789-6894)
- `_sync_external_memory_for_turn` (L2863-2902)
- `shutdown_memory_provider` / `commit_memory_session` (L2824-2859)
- `_build_memory_write_metadata` (L2391)
- Review agent memory copy (L2334-2336)
- Compression hooks (L6068-6103)
- Nudge tracking (ad-hoc `_turns_since_memory` counters)

These all need to be routed through the pipeline's `MemoryCoordinator` so
that when AIAgent is eventually removed, memory still works.

## Approach: Incremental extraction, TDD each step

### Task 1: Wire MemoryCoordinator into CompatShim (shell delegation)
- CompatShim gets a `memory` property pointing to a `MemoryCoordinator`
- MemoryCoordinator.__init__ gets store/manager references from parent
- Wire `_build_system_prompt` to use coordinator's `build_prompt_blocks()`
- **TDD**: Test that CompatShim's memory coordinator is wired and functional

### Task 2: Extract memory tool routing
- Add `handle_tool_call()` to MemoryCoordinator (routes to store or manager)
- Add `get_tool_schemas()` to MemoryCoordinator (merges built-in + external)
- CompatShim's `_dispatch_tool` routes memory tools through coordinator
- **TDD**: Test memory tool dispatch goes through coordinator

### Task 3: Extract nudge tracking
- NudgeTracker is already implemented in memory.py
- Wire `on_turn_start()` into the pipeline loop
- Wire `record_tool_use()` into tool dispatch feedback
- **TDD**: Test nudges fire at correct intervals

### Task 4: Extract write provenance
- WriteMetadataTracker is already implemented in memory.py
- Wire `build_metadata()` into memory tool calls
- Wire `set_write_origin()` for background review context
- **TDD**: Test metadata is attached to memory writes

### Task 5: Extract sync, commit, shutdown lifecycle
- `_sync_external_memory_for_turn` → coordinator's `on_turn_end()`
- `commit_memory_session` → coordinator's `on_session_end()`
- `shutdown_memory_provider` → coordinator's `shutdown()`
- CompatShim delegates already delegate to coordinator
- **TDD**: Test lifecycle methods route through coordinator

## Non-Goals
- Changing how MemoryStore or MemoryManager works internally
- Removing AIAgent (that's Phase 8+)
- Changing the external API of memory tools

## Success Criteria
- MemoryCoordinator handles all memory operations for the pipeline
- No direct memory attribute access from CompatShim (everything via coordinator)
- All existing tests still pass (CompatShim delegates to AIAgent methods that use coordinator)
- NudgeTracker and WriteMetadataTracker fully wired