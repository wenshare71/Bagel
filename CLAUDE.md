## BAGEL 项目上下文

本仓库是 BAGEL 7B-MoT 多模态模型的推理/训练代码，当前工作重心是 `experiments/` 下的
think-vs-image 延迟 benchmark（详见 `EXPERIMENT_SUMMARY.md` 全链路总结）。

| 需要 | 看这里 |
|------|--------|
| Quick Start / Train / Eval 命令 | `README.md` / `TRAIN.md` / `EVAL.md` |
| 实验踩坑记录（OOM、多卡、dtype 等） | `DEBUG_NOTES.md` |
| 实验结论数字 | `experiments/PROFILE_ANALYSIS.md` |
| budget forcing 机制 | `experiments/BUDGET_FORCING.md` |

**Gotchas：**
- `inferencer.py::decode_image` 在 autocast 上下文下有 dtype 冲突历史，改动前先看 `DEBUG_NOTES.md` §3-4
- 多卡 worker 用 `subprocess`（非 `multiprocessing.spawn`）启动，`CUDA_VISIBLE_DEVICES` 必须在子进程 `import torch` **之前**设置，否则不生效（`DEBUG_NOTES.md` §8.4）

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **Bagel** (4442 symbols, 6277 relationships, 107 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/Bagel/context` | Codebase overview, check index freshness |
| `gitnexus://repo/Bagel/clusters` | All functional areas |
| `gitnexus://repo/Bagel/processes` | All execution flows |
| `gitnexus://repo/Bagel/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
