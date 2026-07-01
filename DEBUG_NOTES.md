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
