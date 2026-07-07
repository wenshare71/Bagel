# edit-with-think ratio 实测

`PROFILE_ANALYSIS.md` 给的延迟定律 `t_total ≈ t_prefill + 0.055·T + s(R)·N`
基于纯文本 t2i prompt。本 doc 用真实编辑样本验证 **多模态输入**(文本+参考图)
下两段延迟的相对关系是否依然成立,以及 `t_image` 是否因 attention 要扫更长的 KV
(image token 加在 KV cache 里)而显著恶化。

**TL;DR**:编辑场景下 `t_think` 跟纯文本几乎一致(同 token 数同 decode 成本),
`t_image` 跟 `image_shape²` 严格正相关(`octupusy 688×512 → 2.1s`,
`women 1024×800 → 4.6s`)。`ratio = t_think / t_image` 因此**对分辨率反敏感**:
小图场景 ratio 高达 25(think 绝对主导),大图场景 ratio 约 12(跟纯文本相当)。

---

## 1. 实验设置

| 项 | 值 |
|---|---|
| 脚本 | `experiments/scripts/run_edit_think_bench.py` |
| 加载 | pipeline(13/15 accelerate,跟 cap sweep 同) |
| GPU | 2,3 (RTX 4090,本次空闲;上次 cap sweep 在 0,1 上跑,本机有残留 kernel) |
| cfg | bench (cfg_text=cfg_img=1.0,去噪每步 1 次前向,与 cap sweep 同口径) |
| cap | 1000 (budget forcing min=max=cap,与 cap sweep 同) |
| N | 10 (num_timesteps) |
| trial | 4 = women.jpg × 2 prompts + octupusy.jpg × 2 prompts |
| prompt 来源 | `test_images/` 下 BAGEL 官方 inference.ipynb 示例(本仓库已 commit) |
| 输入分辨率上限 | vae `ImageTransform(1024, 512, 16)` / vit `ImageTransform(980, 224, 14)` |

**关键代码修复**(commit 提交中):`inferencer.update_context_image` 内
`padded_images`(vae)和 `packed_vit_tokens`(vit)在 `torchvision.transforms`
流水线里是 **fp32 CPU tensor**,但 `vit_model` 由 pipeline loader 以 `bf16`
加载,直接喂会触发 "Input type (float) and bias type (BFloat16)" dtype 冲突。
修复:把两个 tensor 对齐到对应子模块参数的真实 dtype/device。模式同
`decode_image` 在 commit `d7aa26a` 的同类修复。

## 2. 结果

### 逐 trial

| trial | image | shape | t_prefill | t_think | t_image | ratio | tok/s | kv_len(prefill) |
|---|---|---|---|---|---|---|---|---|
| 1 | women.jpg | 1024×800 | 1.27s | 55.0s | 4.6s | 11.93 | 18.15 | 7121 |
| 2 | women.jpg | 1024×800 | 1.27s | 55.1s | 4.6s | 11.96 | 18.13 | 7121 |
| 3 | octupusy.jpg | 688×512 | 0.76s | 54.5s | 2.2s | 25.34 | 18.31 | 3254 |
| 4 | octupusy.jpg | 688×512 | 0.71s | 54.6s | 2.1s | 25.47 | 18.28 | 3256 |

### 均值 (4 trials)

| 指标 | 编辑实测 | 纯文本 cap sweep pipeline 参考 | 倍数 |
|---|---|---|---|
| t_prefill | 1.00s | 0.22s | ×4.5 (image tokens 加进去) |
| t_think | 54.83s | 65.0s | -16% (注意:可能 GPU 状态差异,见 §4) |
| tok/s | 18.22 | 15.36 | +19% (同上) |
| t_image | 3.38s | 5.45s | -38% (N=10 cfg=1.0 同口径) |
| ratio = t_think/t_image | 16.2 | 11.94 | ×1.36 |
| kv_len(prefill) | 5188 | ~270 | ×19 |

## 3. 三条结论

### 3.1 think 阶段跟 prompt 形态基本无关

- `t_think` 在 women(1024×800, kv=7121)和 octupusy(688×512, kv=3254)
  下分别是 55.0s 和 54.5s — **kv_len 翻倍, t_think 几乎不变**。
- 原因: image tokens 在 prefill 阶段一次性进入 KV cache,think 阶段只往末尾 append 新
  文本 token,decode 路径跟纯文本基本一致。这一发现**强化了
  `PROFILE_ANALYSIS.md` 的可组合定律**: `t_think ≈ 0.055·T` 在多模态场景同样成立,
  image token 数量只影响 prefill,不影响 decode 斜率。

### 3.2 t_image 跟 image_shape 强相关,跟 kv_len 几乎无关

- octupusy 688×512 → t_image 2.1s, vae latent token 数 ≈ (688/16)·(512/16) = 1376
- women 1024×800 → t_image 4.6s, vae latent token 数 ≈ (1024/16)·(800/16) = 3200
- token 数 2.3× → 时间 2.2×, **符合去噪 attention 的 O(N) 预期**(不是超线性)。
- prefill 把 image token 加进 KV,理论上去噪 attention 要扫全部 kv(含 image),
  但实测时间只由 vae latent token 数决定 — 推测是 batch=1 下 attention 的 memory-bound
  主导, token 数(影响 compute)直接决定耗时。

### 3.3 ratio 对图像分辨率反敏感,小图场景 think 绝对瓶颈

| 场景 | ratio = t_think/t_image | 主导阶段 |
|---|---|---|
| 纯文本 cap sweep (1024×1024, N=10) | 11.94 | think 主导但 image 不可忽略 |
| 编辑 women (1024×800, N=10) | 11.93 | 同上,跟纯文本几乎一样 |
| 编辑 octupusy (688×512, N=10) | 25.34 | think **绝对**主导(96%) |

→ **对小图编辑场景,加速 think 比加速 image 重要得多**。这意味着
stage 1(CUDA Graph / torch.compile)的 ROI 在小图编辑场景下最高。

## 4. 局限

- **GPU 状态差异**:本次跑在 GPU 2,3(本机空闲卡);cap sweep 跑在 GPU 0,1
  (当时有旧 Jupyter kernel 残留占着 13+15 GiB)。两次跑的环境基线不同, 18.22 vs
  15.36 tok/s 的差异**不能归因于编辑 vs 纯文本**,更可能是干净 GPU 带来的
  5~10% 加速收益。要严格比较,需要在同一对卡上重新跑纯文本对照。
- **cfg=bench 单档**:cfg_text=cfg_img=1.0 下,去噪每步 1 次前向。真实编辑场景
  一般 cfg_text=4.0, cfg_img=2.0(README 推荐值),去噪每步 3 次前向 ——
  `t_image` 大约会 ×2.5。脚本支持 `--cfg realistic`, 待补跑对照。
- **样本量小**:4 trial,且 2×2 的图×prompt 设计不能区分"图大小影响"和
  "prompt 长度影响"(两条 prompt 长度相近,不是干净的 2×2 因子设计)。
  要严格拆解,需要 4 张图(2 大 2 小) × 2 prompt 共 8 trial。