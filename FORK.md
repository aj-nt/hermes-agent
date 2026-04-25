# aj-nt/hermes-agent

Independent project sharing git history with [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent). All 8 upstream PRs were closed without merging (2025-04-25), so this fork diverges freely with production-hardened bug fixes and features that upstream hasn't accepted.

**I run this in production daily.** Every patch on the `dogfood` branch has been tested against real workloads, not just unit tests.

## What's different from upstream

The `dogfood` branch is 33 commits ahead of upstream `main`, including:

### Bug fixes

| Fix | Upstream PR | Severity |
|-----|-------------|----------|
| GLM stop-to-length heuristic false positives | [#15463](https://github.com/NousResearch/hermes-agent/pull/15463) | P2 |
| Compression threshold blocks at small contexts | [#15496](https://github.com/NousResearch/hermes-agent/pull/15496) | P1 |
| Gemini streaming usage metadata not extracted | [#15493](https://github.com/NousResearch/hermes-agent/pull/15493) | P2 |
| API key drift on provider switch | [#14370](https://github.com/NousResearch/hermes-agent/pull/14370) | P2 |
| JSONDecodeError misclassified as local validation error | [#14366](https://github.com/NousResearch/hermes-agent/pull/14366) | P2 |
| State-sync output leaked to terminal | [#15469](https://github.com/NousResearch/hermes-agent/pull/15469) | P1 |
| Delegation fails for local providers without API keys | fork-only | P2 |
| Vault init AttributeError | fork-only | P1 |
| Red-team QA (5 bugs) | fork-only | P2 |

### Features

| Feature | Upstream PR | Notes |
|---------|-------------|-------|
| Checkpoint tool | [#14351](https://github.com/NousResearch/hermes-agent/pull/14351) | Save task state before compression |
| Term index instant search | [#13794](https://github.com/NousResearch/hermes-agent/pull/13794) | FTS5 inverted index |
| Session backfill from JSON | fork-only | Recover sessions from raw files |

### Internal refactoring (Kore)

Module extractions from `run_agent.py` (12,880 -> 12,613 lines). Pure functions, no behavioral changes:

- `agent/sanitization.py` -- input sanitization
- `agent/kore/config.py` -- parameter objects (Fowler Step 0)
- `agent/kore/provider_headers.py` -- provider header construction
- `agent/kore/think_blocks.py` -- think-tag handling + emoji/Unicode ending detection
- `agent/kore/glm_heuristic.py` -- GLM stop heuristic with config opt-out + 500-char safety gate

### Upstream features now included

Via rebase onto upstream main (38 commits absorbed):

- One-shot mode (`hermes -z`, `--model`/`--provider` flags)
- Cron `context_from` field for job output chaining
- Compression: reserve system+tools headroom, pass provider to length resolver
- Auxiliary: retry without temperature, generalized unsupported-parameter detector
- Dashboard page-scoped plugin slots
- `/stop` immediately aborts streaming retry loop
- Terminal watch_patterns notification spam defense
+ various Discord, Feishu, and TUI fixes

## Branch structure

| Branch | Purpose |
|--------|---------|
| `main` | Tracks `NousResearch/hermes-agent/main` exactly. No local commits. |
| `dogfood` | Production branch. All tested patches on top of `main`. |

## Install

Same as upstream. The `dogfood` branch is a drop-in replacement:

```bash
# From source (dogfood branch)
git clone -b dogfood https://github.com/aj-nt/hermes-agent.git
cd hermes-agent
pip install -e ".[all]"

# Or use the upstream installer and switch branches afterward
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
cd ~/.hermes/hermes-agent
git remote add fork https://github.com/aj-nt/hermes-agent.git
git fetch fork
git checkout dogfood
```

## Updates

I rebase `dogfood` onto upstream `main` regularly (at least weekly). If upstream merges one of our patches, I drop it from the branch in the next rebase.

## License

MIT -- same as upstream. Copyright notice preserved.
