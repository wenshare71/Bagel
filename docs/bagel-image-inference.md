# BAGEL 图像生成完整推理链路解析

> 基于仓库代码整理:`inferencer.py`、`modeling/bagel/bagel.py`、`modeling/bagel/qwen2_navit.py`、`modeling/cache_utils/taylorseer.py`
> 重点覆盖带 `<think>` 思维链的文生图 / 图像编辑路径。

---

## 0. 总览:一条链路,三层代码

```
app.py (Gradio 入口)
   │  inferencer(text=prompt, think=True, ...)
   ▼
inferencer.py :: InterleaveInferencer
   │  __call__ → interleave_inference()          ← 编排层(本文主角)
   │     ├─ update_context_text / update_context_image   (填 KV cache)
   │     ├─ gen_text()                                   (生成 <think> 文本)
   │     └─ gen_image()                                  (flow matching 生成图像)
   ▼
modeling/bagel/bagel.py :: Bagel                  ← 模型层
   │  prepare_* (打包输入) + forward_cache_update_* (prefill)
   │  generate_text (自回归解码)
   │  generate_image → _forward_flow (去噪循环 + 双路CFG)
   ▼
modeling/bagel/qwen2_navit.py                     ← 骨干网络
   Qwen2 (NaViT 风格变长打包) + NaiveCache + FlashAttention + MoT(mode="und"/"gen")
```

三种任务(文生图 / 图像理解 / 图像编辑)共用 `interleave_inference()`,
差异只由 `input_lists` 内容和 `understanding_output` 开关决定,函数内部没有任务分支。

---

## 1. 核心设计:三份并行 KV Cache

图像生成使用**两级 Classifier-Free Guidance (CFG)**,为此推理开始时建立三套独立上下文
(`inferencer.py:228-231`):

| 上下文 | 最终包含的条件 | 用途 |
|---|---|---|
| `gen_context` | 全部 text + image(含 think 系统提示、think 输出) | 主生成路径(完整条件) |
| `cfg_text_context` | 只有 image,**无任何文本** | "无文本条件"基线 → 文本 CFG |
| `cfg_img_context` | 只有 text(含系统提示),**无输入图像** | "无图像条件"基线 → 图像 CFG |

三份 cache 是"故意做残"的版本:提前把某类条件剔除,这样去噪循环中每一步可直接做
1~3 次前向,而不必反复重建上下文。

维护手法藏在主循环的 **deepcopy 时机** 里(`inferencer.py:242-256`):

- 遇到**文本**:先 `cfg_text_context = deepcopy(gen_context)`(写入文本**之前**快照),
  再把文本写入 `gen_context` 与 `cfg_img_context`
  → `cfg_text_context` 永远停在"没有文本"的状态。
- 遇到**图像**:图像只写入 `gen_context`(跳过 `cfg_img_context`),
  写入**之后** `cfg_text_context = deepcopy(gen_context)`
  → `cfg_img_context` 永远不含图像;`cfg_text_context` 含图像。

以图像编辑输入 `[<image>, "make it blue"]`(think 模式)为例:

```
步骤                 gen_context               cfg_text_context      cfg_img_context
──────────────────────────────────────────────────────────────────────────────────
注入 think 系统提示   +sys                      (不注入!保持纯净)     +sys
处理 <image>          +sys,+img                 +sys,+img             +sys        (跳过)
处理 "make it blue"   +sys,+img,+txt            +sys,+img  (不变)     +sys,+txt
生成 <think>...       +sys,+img,+txt,+think     (不变)                (不变)
```

注意 `cfg_text_context` 连系统提示都不注入(`inferencer.py:234-240`)——
它要模拟"完全无文本条件",基线必须纯粹。

---

## 2. 带 `<think>` 的文生图/编辑完整时序

### 2.1 注入系统提示(think=True 时)

```python
# inferencer.py:15-19
GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process
in the mind and then generate the image. The planning process is enclosed
within <think> </think> tags ...'''
```

通过 `update_context_text()` 写入 `gen_context` 和 `cfg_img_context` 的 KV cache。

### 2.2 上下文填充(prefill 阶段)

每段输入走一次 `prepare_* → forward_cache_update_*`:

- **文本**(`bagel.py:232-297`):tokenize → 加 BOS/EOS → 一次前向写入 KV cache
  (`is_causal=True`)。
- **图像**(`inferencer.py:61-96`),`update_context_image(vae=not understanding_output, vit=True)`:
  - **ViT 支路**(`forward_cache_update_vit`):SigLIP 编码 → connector 投影到 LLM 维度,
    提供**语义级**特征;
  - **VAE 支路**(`forward_cache_update_vae`):VAE encode → latent patchify →
    `vae2llm` 投影 + timestep embedding(t=0)+ 位置编码,提供**像素级**特征。
  - 生成/编辑任务两条都走(编辑效果好的原因:同时拿到语义与像素两种表示);
    理解任务只走 ViT。
  - 图像 token 序列形如 `<start_of_image> [latent/vit tokens...] <end_of_image>`,
    整幅图共享同一个 RoPE position id(`packed_position_ids` 全部相同,rope 只 +1),
    图内部空间关系由 2D `packed_vae_position_ids`/`vit position ids` 承担。
  - 图像 token 内部注意力为**双向**(`is_causal=False`)。

另外注意:输入图像会覆盖输出尺寸(`inferencer.py:252`,`image_shapes = input_term.size[::-1]`),
编辑任务输出自动跟随输入图尺寸,外部传入的 `image_shapes` 参数会被无视。

### 2.3 生成 `<think>` 思维链文本

```python
# inferencer.py:263-266
if think:
    gen_text = self.gen_text(gen_context, ...)              # 自回归解码
    gen_context = self.update_context_text(gen_text, gen_context)  # 写回主上下文!
    output_list.append(gen_text)
```

- `gen_text()`(`inferencer.py:188-205`)→ `prepare_start_tokens`(以 BOS 起步)→
  `model.generate_text()`(`bagel.py:930`):逐 token 自回归,采样或 argmax,
  遇 `eos_token_id` 停止,同时支持 s1 风格 budget forcing
  (`wait_token_ids`:模型想提前停时强插 "Wait," 续想到 `min_length`)。
- **关键**:think 文本被 `update_context_text` **写回 `gen_context`**——
  模型自己的规划过程真实成为后续去噪的条件,这就是"先想后画"。
- think 文本**不写入** `cfg_text_context` / `cfg_img_context`,两路 CFG 基线不受影响。
- `gen_text()` 内部 `deepcopy(gen_context)`,解码过程中产生的 KV 不污染原上下文;
  写回靠的是显式的 `update_context_text(gen_text, ...)` 重新 prefill 一遍。

### 2.4 图像生成:flow matching 去噪循环

`gen_image()`(`inferencer.py:99-171`)先做三件准备:

1. `prepare_vae_latent`(`bagel.py:552`):按目标尺寸 `(H,W)` 生成
   `h*w = (H/16)*(W/16)` 个 latent token 的**初始高斯噪声** `packed_init_noises`,
   以及 `<start_of_image> ... <end_of_image>` 的打包索引;
2. `prepare_vae_latent_cfg` × 2(`bagel.py:610`):为两路 CFG 上下文各生成一套
   query/kv 索引(它们的 kv 长度不同,索引要单独算);
3. 调 `model.generate_image()`(`bagel.py:644`)。

去噪主循环(`bagel.py:691-754`):

```python
x_t = packed_init_noises                                  # 纯噪声起步
timesteps = linspace(1, 0, num_timesteps)                 # 默认 50 步
timesteps = shift*t / (1 + (shift-1)*t)                   # timestep_shift=3.0,向高噪端加密

for i, t in enumerate(timesteps):
    if cfg_interval[0] < t <= cfg_interval[1]:            # 默认 (0.4, 1.0)
        cfg_text_scale_, cfg_img_scale_ = cfg_text_scale, cfg_img_scale
    else:
        cfg_text_scale_, cfg_img_scale_ = 1.0, 1.0        # 后 40% 时间步关掉 CFG,省 2/3 前向
    v_t = self._forward_flow(x_t, t, ..., 三套 past_key_values ...)
    x_t = x_t - v_t * dts[i]                              # 欧拉法 ODE 积分一步
```

这不是 DDPM 加噪/去噪,而是 **flow matching**:模型直接预测从数据指向噪声的
速度场 v_t,采样即沿 ODE 从 t=1(噪声)积分到 t=0(数据)。

### 2.5 单步内部:`_forward_flow` 与双路 CFG(`bagel.py:757-907`)

每个时间步最多 3 次 LLM 前向(均 `is_causal=False`、`update_past_key_values=False`,
纯读 cache 不写):

```
v_t          = LLM(x_t | gen_context 的 KV)          ← 完整条件
cfg_text_v_t = LLM(x_t | cfg_text_context 的 KV)     ← 无文本条件   (cfg_text_scale>1 才算)
cfg_img_v_t  = LLM(x_t | cfg_img_context 的 KV)      ← 无图像条件   (cfg_img_scale>1 才算)
```

x_t 每步经 `vae2llm` 投影 + 当前 timestep embedding + 位置编码后填进 packed 序列;
MoT 模型(`use_moe`)此时走 `mode="gen"`:latent token 用 `mlp_moe_gen`/生成专家,
文本 token 用原 `mlp`/理解专家(`qwen2_navit.py:781-820`)。

CFG 融合(嵌套外推 + renorm,`bagel.py:873-905`):

```python
v_t_text = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)      # 先文本外推
v_t_     = cfg_img_v_t  + cfg_img_scale  * (v_t_text - cfg_img_v_t)  # 再图像外推
# renorm:把外推后的模长拉回不超过原 v_t 的模长,防过曝/伪影
scale = (||v_t|| / ||v_t_||).clamp(min=cfg_renorm_min, max=1.0)      # global 或 channel 范数
v_t = v_t_ * scale
```

`cfg_renorm_type`: `global`(整体范数,默认)/ `channel`(逐 token)/
`text_channel`(只对文本 CFG 一级做 channel renorm)。

### 2.6 VAE 解码回像素

`decode_image()`(`inferencer.py:174-185`):
去噪完的 latent `(h*w, patch²·C)` reshape 回 `(1, C, H/8, W/8)` →
`vae_model.decode()` → `[-1,1]` 映回 `[0,255]` → PIL Image。

---

## 3. 全链路时序图(think 模式文生图)

```
用户 prompt "画一只戴帽子的猫", think=True
        │
        ▼
┌─ Prefill 阶段 ──────────────────────────────────────────────┐
│ GEN_THINK_SYSTEM_PROMPT ─► gen_context, cfg_img_context     │
│ prompt 文本            ─► gen_context, cfg_img_context      │
│                          (cfg_text_context 保持无文本快照)   │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─ Think 阶段(自回归解码)──────────────────────────────────┐
│ gen_text(gen_context) → "<think>猫应该坐姿,帽子是红色     │
│   贝雷帽,背景虚化...</think>"                              │
│ update_context_text(think文本, gen_context)   ← 写回!      │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─ 去噪阶段(50 步 flow matching)────────────────────────────┐
│ x₁ = 纯噪声 (h·w 个 latent token)                           │
│ for t = 1 → 0:                                              │
│    v  = LLM(x_t | sys+prompt+think)      主路径             │
│    v⁰ = LLM(x_t | ∅ 无文本)              t>0.4 才算         │
│    vⁱ = LLM(x_t | sys+prompt+think 无图)  文生图时 scale=1.5 │
│    v ← CFG 外推 + renorm                                    │
│    x ← x - v·dt                                             │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
   VAE decode → PIL Image
   输出 [think文本, 图像]
```

---

## 4. 链路上的推理加速点

| 加速点 | 位置 | 默认状态 | 原理 |
|---|---|---|---|
| KV cache 复用 (`NaiveCache`) | 整个 prefill/decode | 常开 | 多模态 token 统一进 cache,新输入只算增量 |
| FlashAttention varlen + NaViT 打包 | `qwen2_navit.py:361,579` | 常开 | 变长序列无 padding,`flash_attn_varlen_func` |
| CFG 条件跳过 | `bagel.py:835,854` | 常开 | scale≤1.0 的 CFG 支路整次前向直接不算 |
| `cfg_interval=(0.4,1.0)` | `bagel.py:701` | 常开 | 低噪声段强制关 CFG → 每步 3 次前向降为 1 次 |
| TaylorSeer | `taylorseer.py` + `qwen2_navit.py:773-829` | **默认关** | 约每 3 步真算一次,中间步用六阶泰勒外插整层输出,精度换速度 |
| 量化/多卡 (mode 1/2/3) | `app.py:104-130` | 部署选项 | bf16+device_map / NF4 / INT8,解决显存可行性 |

TaylorSeer 开启时三条 CFG 路径各自独立维护一套导数缓存(`bagel.py:680-689`),
因为三条路径的特征轨迹不同。

---

## 5. 关键代码索引速查

| 环节 | 文件:行号 |
|---|---|
| think 系统提示定义 | `inferencer.py:15-19` |
| 三套上下文初始化与 deepcopy 时机 | `inferencer.py:228-256` |
| think 文本生成并写回 | `inferencer.py:263-266` |
| 文本 prefill | `bagel.py:232` (prepare) / `bagel.py:267` (forward) |
| 图像 ViT prefill | `bagel.py:299` / `bagel.py:362` |
| 图像 VAE prefill | `bagel.py:417` / `bagel.py:491` |
| 初始噪声与生成打包 | `bagel.py:552` (prepare_vae_latent) |
| CFG 索引打包 | `bagel.py:610` (prepare_vae_latent_cfg) |
| 去噪主循环 | `bagel.py:644-754` (generate_image) |
| 单步前向 + CFG 融合 | `bagel.py:757-907` (_forward_flow) |
| 自回归文本解码 | `bagel.py:930` (generate_text) |
| latent → 图像 | `inferencer.py:174-185` (decode_image) |
| MoT 双专家分流 (und/gen) | `qwen2_navit.py:781-820` |

---

## 6. 一句话总结

BAGEL 把文生图做成了"**LLM 续写**":文本、ViT 语义、VAE latent 全部打包进同一个
Qwen2 骨干的 KV cache;think 模式先自回归"写规划"并把规划**写回上下文**,
再以纯噪声 latent 为 query、以完整上下文为条件,用 flow matching ODE 积分 50 步
逐步去噪;每步用三套"故意做残"的 KV cache 做两级 CFG 外推(文本引导 + 图像引导)
并 renorm,最后 VAE 解码回像素。
