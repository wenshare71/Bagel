# Think-Bottleneck Benchmark Notebook 调试记录

> 实验脚本: `experiments/think_bottleneck_benchmark.ipynb`(cherry-pick 自 wenshare71/Bagel commit `2220398`)
> 日期: 2026-07-01

## 1. 环境

| 项 | 值 |
|---|---|
| 机器 | Linux x86_64 (glibc 2.31) |
| Python | 3.9.5 |
| torch | 2.5.1+cu121 |
| CUDA / cuDNN | 12.1 / 9.1.0 |
| GPU | 8 × NVIDIA GeForce RTX 4090(每卡 24 GiB) |
| 虚拟环境 | `/home/wuwenxuan03/bagel/.venv_bagel`(已注册为 Jupyter kernel `bagel`) |
| 模型权重 | `/home/wuwenxuan03/bagel/BAGEL-7B-MoT`(`ema.safetensors` bf16,`ae.safetensors` fp32) |

## 2. 问题描述

实验 notebook 的 **第 7 节 warm-up(cell-14)** 抛出:

```
TypeError: must be real number, not NoneType
print("warm-up (N=50, CFG on): t_think=%.3fs t_image=%.3fs" % (_["t_think"], _["t_image"]))
```

### 2.1 诊断:这是次要错误

`49/49` 进度条说明 `inferencer.gen_image` 的去噪循环已经跑完,图像 latent 已生成。但 `run_trial`(cell-10)的 `try/except` 捕获了真实异常,把 `t_think` / `t_image` 全部置为 `None`,真实错误存进了 `record["error"]`。warm-up cell 没有检查 `record["ok"]` 就直接 `%.3f % None`,于是把真实错误掩盖成了 `TypeError`。

### 2.2 暴露真实错误

把 cell-14 改成失败时打印 `_["error"]`(成功才打印计时):

```python
_ = run_trial(_warmup_prompt, block_a_conditions[0], seed=999)
if not _["ok"]:
    print("warm-up (N=50, CFG on) FAILED:", _["error"])
else:
    print("warm-up (N=50, CFG on):        t_think=%.3fs  t_image=%.3fs" % (_["t_think"], _["t_image"]))
```

重跑后真实错误浮现:

```
warm-up (N=50, CFG on) FAILED: RuntimeError('Input type (float) and bias type (c10::BFloat16) should be the same')
```

错误位置:`inferencer.py` `decode_image()` 中的 `self.vae_model.decode(latent)`(VAE 反卷积解码)。

## 3. 根因:autocast 与 VAE decode 的 dtype 矛盾

`run_trial` 原始实现把整段推理包在 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` 里。`run_inference.py`(可正常工作的入口)**不用 autocast**。两者对比揭示了矛盾:

| | LLM 部分(权重 bf16) | VAE decode(`vae_model.decode`) |
|---|---|---|
| **有 autocast** | ✅ autocast 自动把 fp32 输入降成 bf16 | ❌ latent 是 fp32,而 autocast 把 VAE conv 的 bias 临时降成 bf16 → `Input float vs bias BFloat16` |
| **无 autocast** | ❌ `prepare_vae_latent` 产出的 fp32 输入撞 bf16 权重 → `mat1 and mat2 ... Float and BFloat16` | ✅ latent 与 VAE 都是 fp32,一致 |

实测两种情况:
- 原始(有 autocast):`Input type (float) and bias type (c10::BFloat16)`
- 去掉 autocast 后:`mat1 and mat2 must have the same dtype, but got Float and BFloat16`(LLM 部分报错)

也就是说:LLM 推理**需要** autocast 来做 fp32→bf16 转换,但 autocast 又会破坏 VAE decode。两者对 dtype 的要求相反。

补充事实:`ae.safetensors` 存储的是 **fp32** 权重,`load_ae` 用 `assign=True` 加载,VAE 在 cell-4 里只 `.to(DEVICE)` 不改 dtype,因此 VAE 实际权重是 fp32。但在外层 autocast 作用域内,conv 权重/bias 会被 autocast 临时降成 bf16,而 latent 作为中间产物保持 fp32,从而冲突。

## 4. 修复

### 4.1 `inferencer.py` — `decode_image`(防御性,兼容所有调用方)

让 VAE decode **不受外层 autocast 影响**,并把 latent 对齐到 VAE 权重的真实 dtype/device:

```python
def decode_image(self, latent, image_shape):
    H, W = image_shape
    h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

    latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
    # Align latent to the VAE's actual weight dtype/device and run decode
    # outside autocast: callers may wrap gen_image in torch.autocast, which
    # can leave the latent as float32 while autocast-cast VAE conv weights
    # are bfloat16 (or vice-versa). Disabling autocast here makes decode use
    # the VAE's real stored dtype consistently.
    _vae_p = next(self.vae_model.parameters())
    latent = latent.to(device=_vae_p.device, dtype=_vae_p.dtype)
    with torch.autocast(device_type=_vae_p.device.type, enabled=False):
        image = self.vae_model.decode(latent)
    image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
    image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

    return image
```

- 用 `next(self.vae_model.parameters())` 取 VAE 真实 dtype/device(`AutoEncoder` 没有 `.device`/`.dtype` 属性)。
- `torch.autocast(enabled=False)` 屏蔽外层 autocast 对 VAE conv 权重的降精度。
- 对 `run_inference.py` / `inference.ipynb`(无 autocast)是 no-op,完全兼容。

### 4.2 `experiments/think_bottleneck_benchmark.ipynb` — cell-10 `run_trial`

保留 autocast 包住 **LLM 推理**(gen_text + gen_image 的去噪循环,需要 fp32→bf16 自动转换);VAE decode 在 `gen_image` 内部由 `decode_image` 自己禁用 autocast 处理,无需 notebook 侧干预。

```python
try:
    gen_context = inferencer.init_gen_context()
    cfg_text_context = deepcopy(gen_context)
    cfg_img_context = deepcopy(gen_context)

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        with sync_timer() as t_prefill:
            ...  # update_context_text
        with sync_timer() as t_think:
            gen_text = inferencer.gen_text(...)
        gen_context = inferencer.update_context_text(gen_text, gen_context)
        with sync_timer() as t_image:
            img = inferencer.gen_image(...)   # 内部 decode_image 已禁用 autocast

    think_token_count = len(tokenizer(gen_text, add_special_tokens=False).input_ids)
    ...
```

### 4.3 `experiments/think_bottleneck_benchmark.ipynb` — cell-14 warm-up

失败时打印真实 `_["error"]`,不再掩盖:

```python
_ = run_trial(_warmup_prompt, block_a_conditions[0], seed=999)
if not _["ok"]:
    print("warm-up (N=50, CFG on) FAILED:", _["error"])
else:
    print("warm-up (N=50, CFG on):        t_think=%.3fs  t_image=%.3fs" % (_["t_think"], _["t_image"]))
```

## 5. 验证状态 / 待办

**改动尚未在 notebook 中验证成功。** 当前运行的 Jupyter kernel 仍持有 `inferencer` 模块的旧缓存(改动前 import 的版本),因此 cell-14 仍报与修复前完全相同的错误:

```
warm-up (N=50, CFG on) FAILED: RuntimeError('Input type (float) and bias type (c10::BFloat16) should be the same')
```

### 下一步(用户操作)

1. **重启 notebook kernel**(或重启 Jupyter),让 `inferencer.py` 的 `decode_image` 改动生效。
2. 从 cell-2 开始依次重跑(cell-4 重新加载模型)。
3. 重跑 cell-10 → cell-14。

预期:warm-up 正常打印 `t_think` / `t_image`,而非 `FAILED:`。

如果重启 kernel 后仍报相同错误,则说明 `decode_image` 内 `autocast(enabled=False)` 仍不足以阻止 VAE 内部 conv 的 dtype 冲突,届时需要进一步排查 `modeling/autoencoder.py` 的 `AutoEncoder.decode` 是否在内部自行 cast。

## 6. 本次提交涉及的文件

- `inferencer.py` — `decode_image` dtype/autocast 修复
- `experiments/think_bottleneck_benchmark.ipynb` — cell-10 (run_trial autocast 范围)、cell-14 (warm-up 错误暴露)
- `DEBUG_NOTES.md` — 本文档

未跟踪文件(`BAGEL-7B-MoT/` 模型权重、`output_*.png`、`run_inference.py`、`requirements_infer.txt`)不纳入本次提交。`.gitignore` 与 `inference.ipynb` 的既有改动也不属于本次诊断范围,不在本次提交内。

---

## 7. `think_cap_benchmark.ipynb` OOM 问题 (2026-07-02)

### 现象

cell-14 warm-up 报错：

```
FAILED: OutOfMemoryError('CUDA out of memory. Tried to allocate 1024.00 MiB.
GPU 0 has a total capacity of 23.65 GiB of which 692.50 MiB is free.
Process 56131 has 10.73 GiB memory in use.
Process 209081 has 12.23 GiB memory in use.
```

### 根因

**两个 Jupyter kernel 同时在 GPU 上加载了模型。**

| Kernel | 用途 | 显存占用 |
|--------|------|----------|
| PID 195457 | `think_bottleneck_benchmark.ipynb`(已跑完 162/162,未关闭) | ~11 GiB |
| PID 247936 | `think_cap_benchmark.ipynb`(warmup 阶段) | ~12 GiB |

GPU 0 总共 23.65 GiB,两个模型进程合计 ~23 GiB,没有剩余空间。

### 修复

#### 手动操作(必须)
在 Jupyter 界面关闭 `think_bottleneck_benchmark.ipynb` 的 kernel,或 `kill 195457`。

#### 代码防护(已应用)
对 `think_cap_benchmark.ipynb` 做了三层防御:

1. **cell-2**:添加 `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")` + `import gc`,按 PyTorch 官方建议缓解碎片化。
2. **cell-10** (`run_trial`):`finally` 块中显式 `del gen_context, cfg_text_context, cfg_img_context` + `gc.collect()` + `torch.cuda.empty_cache()`,确保每 trial 后释放三个 KV cache 的引用。
3. **cell-17** (sweep 循环):每 20 trial checkpoint 时额外 `gc.collect()` + `torch.cuda.empty_cache()`,防御 Python GC 滞后。

#### 不改的
- `inferencer.py` `gen_text()` 中的 `deepcopy(gen_context)` 是必需的——`generate_text` 的 autoregressive 循环会在原地修改 `past_key_values`,调用方依赖原始 context 不被污染来执行后续 `update_context_text`。

---

## 8. 多卡并行 sweep: 从 3h 到 ~50min 的完整踩坑记录 (2026-07-02)

### 8.1 动机

`think_cap_benchmark.ipynb` 有 480 trials，单进程跑 ~3h。本机 8×4090，想用多卡并行加速。

### 8.2 GPU 拓扑

```
GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7
  X  SYS  SYS  SYS  SYS  SYS  SYS  SYS   ← 全 SYS (PCIe), 无 P2P/NVLink
```

**结论**：多卡分载一个模型 → 每次 forward 跨 PCIe，慢。但多卡各跑各的 → 无通信开销，线性加速。

### 8.3 问题一：模型单卡装不下

**现象**：最初尝试 8×单卡并行（`--gpus-per-worker 1`），8 个 worker 全部 OOM。

```
RuntimeError: CUDA error: out of memory
model = model.to(dtype=torch.bfloat16, device="cuda:0").eval()
```

**根因**：`ema.safetensors` 磁盘上 28 GB（fp32 存储），即使用 `load_file(device="cpu")` + `model.to(bf16)` 转精度，实际模型 + VAE + ViT + connectors 在 GPU 上也占用 ~16-18 GB。加上 denoising forward pass 需要额外 ~4-6 GB 临时空间（1024×1024 分辨率，4096 image tokens），单卡 24 GB 不够。

**修复**：改为 `--gpus-per-worker 2`，8 卡 → 4 个并行 worker，每 worker 用 accelerate device_map 跨 2 卡分载。

### 8.4 问题二：CUDA_VISIBLE_DEVICES 在 spawn 子进程中不生效

**现象**：用 `multiprocessing.spawn` 启动 worker，`worker_main` 第一行设置 `os.environ["CUDA_VISIBLE_DEVICES"]`，但所有 worker 仍然挤在 GPU 0 上。

**根因**：`spawn` 子进程会重新执行模块顶层代码，`import torch` 发生在 `worker_main` 之前。此时环境变量还没设置，torch 已经初始化在默认 GPU 上了。

**修复**：改用 `subprocess.Popen` 启动 worker，通过 `env` 参数传入 `CUDA_VISIBLE_DEVICES`。subprocess 在 fork 前就设好环境变量，子进程从第一行代码（包括 `import torch`）就正确看到隔离的 GPU。

```python
# 关键代码
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = gpu_str  # e.g. "2,3"
p = subprocess.Popen(cmd, env=env)
```

### 8.5 问题三：infer_auto_device_map 把层 offload 到 disk（性能灾难）

**现象**：dtype 修复后 warmup 成功但极慢 — `t_think=264s`（正常 ~9s，慢 29×）。

device_map 输出：
```
layers 0-9  → GPU0
layers 10-21 → GPU1
layers 22-27 → disk      ← 6 层被卸到磁盘！
lm_head     → disk
vit_model   → disk
```

每次 forward pass 都需要从磁盘 swap 这 6 层 + lm_head + vit，IO 延迟完全摧毁了性能。

**根因**：`infer_auto_device_map` 默认按 **fp32** 估算模型大小（28 GB），但实际加载时 `load_checkpoint_and_dispatch(dtype=torch.bfloat16)` 只占用 ~14 GB。accelerate 误以为 2 卡装不下，就把多余的卸到 disk。

**修复**：在 `infer_auto_device_map` 中传入 `dtype=torch.bfloat16`：

```python
device_map = infer_auto_device_map(
    model,
    max_memory={i: max_mem_per_gpu for i in range(gpu_count)},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    dtype=torch.bfloat16,  # ← 告知真实加载精度
)
```

### 8.6 问题四：accelerate 分层严重不均衡（再次 OOM）

**现象**：加了 `dtype=bf16` 后，4 个 worker warmup 全部 OOM。

```
GPU 0: layers 0-23 (24 layers) → 23.42 GiB / 23.65 GiB  ← 几乎满了
GPU 1: layers 24-27 (4 layers) + norm/lm_head/vit → ~10 GiB  ← 浪费
```

warmup 阶段 gen_image 需要额外临时空间做 denoising forward pass，GPU 0 只剩 ~200 MiB，直接炸了。

**根因**：`max_mem_per_gpu="23GiB"` 设置太高，accelerate 倾向于尽量塞满第一张卡再往第二张放。24/28 层全压在 GPU 0 上，留给 forward pass 的工作空间不够。

**修复**：抛弃 `infer_auto_device_map`，**手动构建均衡的 device_map**：

```python
# 28 层，14 层/卡
num_layers = 28
split = 14
device_map = {}

# LLM layers: 均衡分配
for layer_idx in range(num_layers):
    gpu = 0 if layer_idx < split else 1
    device_map[f"language_model.model.layers.{layer_idx}"] = gpu

# Embed → GPU 0
device_map["language_model.model.embed_tokens"] = 0

# Norm / lm_head / vit / connectors → GPU 1
device_map["language_model.model.norm"] = 1
device_map["language_model.lm_head"] = 1
device_map["vit_model"] = 1
# ... etc
```

预期：
```
GPU 0: layers 0-13 + embed → ~7 GB, 留 15+ GB 给 forward pass
GPU 1: layers 14-27 + norm/lm_head/vit/connectors → ~9 GB, 留 13+ GB
```

### 8.7 辅助改进

在跑多卡 sweep 过程中同步对 notebook 做了这些防御性改动：

| 位置 | 改动 | 作用 |
|------|------|------|
| cell-2 | `import gc` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 缓解碎片化 |
| cell-4 | `N_ACCELERATE_GPUS = 2` | 限制 accelerate 只用 2 卡 |
| cell-10 `run_trial` finally | `del gen_context, cfg_text_context, cfg_img_context` + `gc.collect()` | 每 trial 后强制释放 KV cache |
| cell-17 sweep 循环 | checkpoint 时 `gc.collect()` + GPU empty_cache | 防止 Python GC 滞后 |

### 8.8 最终架构

```
8 卡 GPU = 4 worker × 2 GPU/worker × 120 trial/worker

Main (PID X)
├─ subprocess Worker 0: CUDA_VISIBLE_DEVICES=0,1 → accelerate 手动 device_map → 120 trials
├─ subprocess Worker 1: CUDA_VISIBLE_DEVICES=2,3 → accelerate 手动 device_map → 120 trials
├─ subprocess Worker 2: CUDA_VISIBLE_DEVICES=4,5 → accelerate 手动 device_map → 120 trials
└─ subprocess Worker 3: CUDA_VISIBLE_DEVICES=6,7 → accelerate 手动 device_map → 120 trials
                                ↓
                        合并 → trials.csv → notebook §9-13 分析
```

### 8.9 问题五：手动 device_map 跨设备冲突

**现象**：4 个 worker warmup 立即报错（denoising 进度条 0/49）：

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cuda:1!
```

**根因**：手动 device_map 把 `embed_tokens` 放在 GPU 0，但 `vit_model`、`connector`、`time_embedder`、`vae2llm`、`llm2vae`、`latent_pos_embed`、`vit_pos_embed` 这些**辅助模块**全部放在 GPU 1。

而在 Bagel 的 `prepare_vae_latent` / `prepare_prompts` 等 forward 方法中，这些模块需要协同工作——比如 `llm2vae` 的输出要和 `latent_pos_embed` 的结果拼接，如果它们在不同的 GPU 上，tensor 拼接就会报跨设备错误。

对比：原来的 accelerate 代码有这一段强制同卡逻辑，但我们手动构建 device_map 时跳过了：

```python
same_device_modules = [
    "language_model.model.embed_tokens", "time_embedder", "latent_pos_embed",
    "vae2llm", "llm2vae", "connector", "vit_pos_embed",
]
first_device = device_map.get(same_device_modules[0])
for k in same_device_modules:
    device_map[k] = first_device  # 强制同一张卡
```

**预期修复**（待实施）：把所有 `same_device_modules` 强制放到 GPU 0（embed_tokens 所在的卡），或者全部放到 GPU 1。关键是这些模块必须**同一张卡**。

```python
# 修复：将所有辅助模块与 embed_tokens 放在同一卡
device_map["language_model.model.embed_tokens"] = 0
for k in same_device_modules:
    device_map[k] = 0  # 全部 GPU 0
# vit_model 和 lm_head 也放 GPU 0
device_map["vit_model"] = 0
device_map["language_model.lm_head"] = 0
device_map["connector"] = 0
# 然后调整层分配：GPU 0 放少一些层（比如 12 层），让出空间给 aux 模块
```

这样 GPU 0 装 embed + aux 模块 + 12 层 LLM，GPU 1 装剩余 16 层 LLM + norm。显存大致均衡。

### 8.10 当前状态

sweep 尚未成功跑通。问题链：

1. ✅ 两 kernel 冲突 → 已解决（关旧 kernel）
2. ✅ CUDA_VISIBLE_DEVICES 不生效 → 已解决（subprocess.Popen + env）
3. ✅ 层被 offload 到 disk → 已解决（dtype=bf16）
4. ✅ 分层不均衡 OOM → 已解决（手动 device_map）
5. ⚠️ 手动 device_map 跨设备冲突 → **待修复**

文件清单：
- `experiments/run_cap_sweep_mp.py` — 多卡并行 sweep 脚本
- `experiments/think_cap_benchmark.ipynb` — cell-8b 调用上述脚本
- `inferencer.py` — §4 的 VAE decode dtype 修复
