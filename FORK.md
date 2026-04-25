# aj-nt/hermes-agent

Community-maintained fork of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent), tracking upstream with production-hardened bug fixes and features that haven't been merged yet.

**I run this in production daily.** Every patch on the `dogfood` branch has been tested against real workloads, not just unit tests.

## What's different from upstream

The `dogfood` branch includes these unmerged upstream PRs plus fork-exclusive improvements:

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

### Features

| Feature | Upstream PR | Notes |
|---------|-------------|-------|
| Checkpoint tool — save task state before compression | [#14351](https://github.com/NousResearch/hermes-agent/pull/14351) | 9 commits |
| Term index instant search | [#13794](https://github.com/NousResearch/hermes-agent/pull/13794) | FTS5 inverted index |

### Internal refactoring (Kore)

These extract modules from `run_agent.py` to make future patches smaller and rebase cleaner:

- `agent/sanitization.py` — input sanitization
- `agent/kore/config.py` — parameter objects (Fowler Step 0)
- `agent/kore/provider_headers.py` — provider header construction
- `agent/kore/think_blocks.py` — think-tag handling
- `agent/kore/glm_heuristic.py` — GLM stop heuristic

All Kore extractions preserve backward compatibility — no behavioral changes.

## Branch structure

| Branch | Purpose |
|--------|---------|
| `main` | Tracks `NousResearch/hermes-agent/main` exactly. No local commits. |
| `dogfood` | Production branch. All tested patches on top of `main`. |
| `kore/refactoring` | Module extractions from `run_agent.py`. Proposed as upstream PRs independently. |

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

## Contributing upstream

Bug fixes and features that land on `dogfood` are also submitted as PRs to [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent). If they merge, great — the patch drops from this fork. If they don't, users of this fork still get the fix.

## License

MIT — same as upstream. Copyright notice preserved.