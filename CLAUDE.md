## BAGEL 项目上下文

本仓库 fork 自 ByteDance-Seed/Bagel，当前工作重心是 `experiments/` 下的
think-vs-image 延迟 benchmark（详见 `docs/EXPERIMENT_SUMMARY.md` 全链路总结）。

| 需要 | 看这里 |
|------|--------|
| Quick Start / Train / Eval 命令 | `README.md` / `TRAIN.md` / `EVAL.md` |
| 实验踩坑记录（OOM、多卡、dtype 等） | `docs/DEBUG_NOTES.md` |
| 实验结论数字 | `experiments/docs/PROFILE_ANALYSIS.md` |
| budget forcing 机制 | `experiments/docs/BUDGET_FORCING.md` |

**Gotchas：**
- `inferencer.py::decode_image` 在 autocast 上下文下有 dtype 冲突历史，改动前先看 `docs/DEBUG_NOTES.md` §3-4
- 多卡 worker 用 `subprocess`（非 `multiprocessing.spawn`）启动，`CUDA_VISIBLE_DEVICES` 必须在子进程 `import torch` **之前**设置，否则不生效（`docs/DEBUG_NOTES.md` §8.4）

## 项目结构（相对上游 fork 新增的部分）

除上游标准文件（`README.md`/`TRAIN.md`/`EVAL.md`、`modeling/`、`data/`、`eval/`、`train/`、`scripts/`、`app.py` 等）外，本 fork 新增内容按类型集中存放，不再散落在根目录：

```
CLAUDE.md                        — 本文件（AI 协作说明，唯一留在根目录的新增文档）
docs/
  bagel-image-inference.md       — BAGEL 图像推理使用说明
  DEBUG_NOTES.md                 — 踩坑记录（OOM / 多卡 / dtype 等）
  EXPERIMENT_SUMMARY.md          — think-vs-image 实验全链路总结
  speedup/
    starry-purring-moonbeam.md   — 阶段性性能优化方案文档（如非对称双卡放置规划）
experiments/
  scripts/    — 可执行的 benchmark / 模型加载脚本（.py）
  notebooks/  — 交互式分析 notebook（.ipynb）
  outputs/    — 脚本/notebook 产出的 CSV、图表（多数被 experiments/.gitignore 忽略，仅白名单的聚合结果入库）
  docs/       — 实验设计/结论文档（BUDGET_FORCING.md、PROFILE_ANALYSIS.md）
```

**新增文件时的路径约束：**

| 新文件类型 | 放哪里 |
|-----------|--------|
| 可运行的 benchmark / 加载器脚本 | `experiments/scripts/*.py` |
| 交互式分析 notebook | `experiments/notebooks/*.ipynb` |
| 脚本/notebook 的运行产出（CSV、图片） | `experiments/outputs/<name>_outputs/`，默认不入库；确需保留的聚合结果要在 `experiments/.gitignore` 里显式加白名单 |
| 实验设计/结论文档 | `experiments/docs/*.md` |
| 不特定于某个实验的项目级文档 | `docs/*.md` |
| 跨阶段的性能优化规划文档 | `docs/speedup/*.md` |

- `experiments/scripts/*.py` 里用 `_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))` 定位仓库根目录（`scripts/` → `experiments/` → 根目录，两层）；notebook 同理用 `os.getcwd()` 计算根目录。**新增文件嵌套层数变化时必须同步改 `".."` 的个数**，否则 `sys.path` / `MODEL_PATH` / `OUTPUT_DIR` 会全部指错且大概率是静默的（除非像 `think_bottleneck_benchmark.ipynb` 那样显式加了 `isdir(root/data)` 校验）。

> 以下 GitNexus 章节由 `npx gitnexus analyze` 自动生成/维护。它是本机的代码索引辅助工具，**不是强制依赖**——训练/推理用的远程 GPU 机器上通常不会装 GitNexus MCP/CLI，遇不到就直接跳过下面的 "Always Do / Never Do" 要求，改用普通的 grep/阅读代码/`git log` 完成同样的工作，不必因为装不上而阻塞任务。

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **Bagel** (4615 symbols, 6491 relationships, 105 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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
