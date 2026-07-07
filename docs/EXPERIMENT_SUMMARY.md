# BAGEL 延迟实验:全链路总结

从最早的单卡探索性 benchmark,到最终的三维分辨率 sweep + budget forcing 修正,完整记录这条实验链路上踩过的坑、做过的架构改造、以及每个阶段产出的数据。结论性的数字详见 `PROFILE_ANALYSIS.md`,本文重点是**过程**。

## 0. 实验目标

回答一个问题:BAGEL 7B-MoT 在 t2i think 模式下,延迟到底花在哪——是自回归生成 `<think>` 文本,还是流匹配去噪出图?这个比例随文本长度 T、去噪步数 N、图像分辨率 R 怎么变?

## 1. 时间线

### 阶段 0 — 单卡探索:`think_bottleneck_benchmark.ipynb`

最初的 think-vs-image 延迟对比 notebook。跑起来后 VAE decode 在 autocast 下报 dtype 不匹配(`inferencer.py::decode_image` 与 autocast 上下文冲突),定位到根因后在 `decode_image` 里做了防御性 dtype 转换,并修了 notebook 里 `run_trial`/warm-up 两处调用方式。产出:确认了 think/image 两段可独立计时,但仅单卡、样本量小,不足以拟合定律。

### 阶段 1 — `think_cap_benchmark.ipynb`:think cap × 蒸馏步数

把自变量扩成 think token 上限(cap)× 去噪步数(N),固定分辨率。跑起来即 OOM——单卡装不下 7B-MoT 权重 + KV cache + 激活值同时存在。先用手动 `gc.collect()` + `PYTORCH_CUDA_ALLOC_CONF` 缓解碎片化,并在 `run_trial` 的 `finally` 块里显式清 KV cache,顺带把 §8 单卡顺序版重命名为 §8a。

### 阶段 2 — 多卡并行改造:3 小时 → 50 分钟

单卡跑一轮 sweep 要 3 小时,不可持续。改造成多 worker 并行(`run_cap_sweep_mp.py`),过程中连续踩了 5 个坑(完整记录见 `docs/DEBUG_NOTES.md` §8):

| # | 问题 | 根因 | 修复 |
|---|---|---|---|
| 8.3 | 模型单卡装不下 | 7B-MoT bf16 ~28GB,4090 只有 24GB | 每 worker 分配 2 卡,层间流水线 |
| 8.4 | `CUDA_VISIBLE_DEVICES` 在 `multiprocessing.spawn` 子进程里不生效 | spawn 子进程继承环境变量的时机晚于 CUDA 初始化 | 改用 subprocess 起 worker,在 `import torch` **之前**设置环境变量 |
| 8.5 | `infer_auto_device_map` 把部分层 offload 到磁盘 | 自动切分算法保守估计显存余量 | 弃用自动切分,手动写 device_map |
| 8.6 | 手动切分 14/14 层仍 OOM | 各层显存占用不均衡(embed/norm/vit 等辅助模块集中在某几层) | 调整为 13/15 不对称切分,辅助模块按依赖就近放置 |
| 8.9 | 手动 device_map 下跨设备张量冲突 | 辅助模块(embed_tokens/vit/lm_head)与主 LLM 层分卡不一致 | 把所有辅助模块与 `embed_tokens` 固定在同一张卡 |

最终架构:8 卡 → 4 个 worker,每 worker 独占 2 卡,`CUDA_VISIBLE_DEVICES` 子进程隔离,层间流水线跨 PCIe(无 NVLink/P2P)。这一套架构后续所有 sweep 复用。

### 阶段 3 — Budget Forcing:为了测长 think

`think_cap_benchmark` 数据显示:cap=256/1000 时模型自然收尾,实际长度只有 ~160 token,0% 触发截断——想研究"长 think"根本测不到。引入 s1 论文式 budget forcing(`1da739d`):解码时若模型在 `min_length` 之前想输出 EOS,就用 `" Wait,"` 续写插入语强制它继续,过了 `min_length` 才放行 EOS。改动集中在 `bagel.py::generate_text`(`forced_queue` 注入逻辑)和 `inferencer.py::gen_text` 的参数透传,默认关闭(`min_length=0`),不影响原有调用方。

`think_cap_benchmark` 用这个开关补跑到 480/480 trial 全部成功,验证了 `t_think ≈ 0.055·T` 这条线性关系与 N 无关。

### 阶段 4 — 三维 Resolution Sweep

把分辨率 R 也纳入自变量,做 R × N × cap 三维全交叉网格(`run_res_sweep_mp.py`,复用阶段 2 的多卡架构),60 条件 × 16 prompt × 2 repeats = 1920 trial,全部成功。

### 阶段 5 — 发现 bug:forcing 开关忘记打开

第一轮 1920 trial 跑完后发现 `FORCE_THINK_LENGTH` 默认值 `False` 没有在 sweep 脚本里被打开——cap=256/1000 两档实际测到的还是自然长度(~162 token),数据被"腰斩"。排查后发现只有这两档失真,cap≤128 本来就被自然长度截断钉死,不受影响;R/N 相关的三条结论(阶段 4 的核心目的)也不受影响。

补救方案:给 `run_res_sweep_mp.py` 加 `--force-think`/`--caps`/`--output-dir` 三个 CLI 参数,只对 cap∈{256,1000} 补跑 768 trial(24 条件 × 32),而不必重跑全部 1920 条。同时注意 worker 子进程会重建 trial 列表,CLI 覆盖必须在 main 和 worker 两侧同步应用,否则 trial index 会错位。notebook 里加了一个合并 cell,按 `think_res_outputs_forced/trials.csv` 是否存在自动把强制批次替换进 `aggregated.csv`。

### 阶段 6 — 数据合并与结论产出

远程补跑完成、notebook 重跑后推送,本地拉取验证合并正确(cap=1000 行 `think_token_count_mean≈999`,cap=256 行 `≈255`),据此写出最终结论文档 `PROFILE_ANALYSIS.md`。

## 2. 最终产出物

| 类别 | 文件 |
|---|---|
| 多卡并行 sweep 脚本 | `run_cap_sweep_mp.py`(think cap × N)、`run_res_sweep_mp.py`(+ R,支持 `--force-think`/`--caps`/`--output-dir`) |
| 分析 notebook | `think_bottleneck_benchmark.ipynb`(单卡探索)、`think_cap_benchmark.ipynb`、`resolution_sweep_benchmark.ipynb`(含 §3b 强制批次合并) |
| 原始数据 | `think_res_outputs/`(aggregated.csv + 4 张图),`think_res_outputs_forced/trials.csv` |
| 工程记录 | `docs/DEBUG_NOTES.md`(dtype bug、OOM、多卡 5 连坑完整排查过程) |
| 方案文档 | `BUDGET_FORCING.md`(budget forcing 设计与改动点) |
| 结论文档 | `PROFILE_ANALYSIS.md` |

## 3. 结论(简写)

- `t_think ≈ 0.055·T`,`t_image ≈ s(R)·N`,两段严格独立、各自线性;
- 平衡点 `T* = s(R)·N / 0.055`:默认配置(R=1024, N=50)下 T*≈486,自然思考只占 25%;蒸馏到 N=5 后 T*降到 49,think 占比可冲到 95%+;
- 结论:**蒸馏把扩散做便宜之后,BAGEL 的延迟瓶颈会转移到 LLM 解码上。**

详细数字、拟合统计、局限性说明见 `PROFILE_ANALYSIS.md`。
