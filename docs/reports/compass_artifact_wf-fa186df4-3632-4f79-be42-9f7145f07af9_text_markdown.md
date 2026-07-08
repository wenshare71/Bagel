# BAGEL-7B-MoT「think」阶段 t_think 推理层加速方案研究报告

## TL;DR
- **第一优先级动手方案:先做「非对称单卡放置」——把 BAGEL 的理解/文本专家(≈7.62B 参数、bf16 ≈15.2GB)完整放进一张 4090,把生成专家(≈6.53B)+VAE+ViT 放到第二张卡。** think 阶段是纯文本自回归,只走文本专家,这样逐 token 解码完全在单卡内完成,彻底消除当前每 token 的跨卡 PCIe 流水线开销。此项免训练、改动仅为 device_map,预计把 0.055 s/token(18.2 tok/s)提升到约 40–50 tok/s(约 2–2.5×)。
- **在单卡基础上再叠加两项免训练优化:CUDA Graph / 静态 KV cache 捕获(消除 kernel launch 开销,batch=1 约 1.2–1.4×)+ 免训练投机解码(用同族 Qwen2.5-0.5B/1.5B-Instruct 当 draft,贪心下约 1.5–2×)。三者叠加后端到端有望达到 4–6× 总加速(t_think 降到约 0.010–0.014 s/token)。**
- **次优先级备选(需轻量训练):ByteDance 官方已发布 Hyper-Bagel(Lu et al., arXiv:2509.18824),用 EAGLE-3 draft head 在单卡 A100 SGLang 环境下把 BAGEL 文本解码从 98.3 TPS 提到约 212.4 TPS(2.16×,论文摘要口径为「over 2x speedup in multimodal understanding」)——这证明投机解码在 BAGEL 上确实可行,但需训练 draft head,故列为备选。**

## Key Findings

**1. BAGEL 骨干与结构已完全确认,对方案有决定性影响。** BAGEL-7B-MoT 从 Qwen2.5-7B-Instruct 微调而来(视觉编码器 siglip-so400m,VAE 用 FLUX.1-schnell),LLM 骨干是标准 Qwen2.5-7B(28 层,hidden 3584,GQA 28 查询头 / 4 KV 头,vocab 152064)。MoT 的两个专家**不共享 QKV/FFN 权重,只共享每层的自注意力操作**(paper §2.2/§2.4:"we duplicate all trainable parameters of Qwen2.5 LLM to create a full-size generation expert");生成专家(`_moe_gen`)是理解专家的完整复制,专门处理 VAE token。**因此 think(纯文本 CoT)阶段只激活理解专家**,其权重 ≈7.616B、bf16 ≈15.2GB,可完整放进单张 24GB 4090,留约 8GB 给 KV cache/激活。

**2. 当前 0.055 s/token 明显慢于单卡硬件上限,瓶颈就是跨卡 PCIe。** RTX 4090 显存带宽 1008 GB/s(384-bit、24GB GDDR6X;NVIDIA 官方标注 NVLink=No,直接佐证本方案「PCIe 无 P2P/NVLink」前提)。7B bf16 单卡解码理论上限约 1008/15.2 ≈ 66 tok/s,实测普遍落在理论上限的 50–80%,即约 33–55 tok/s(此 FP16 区间为据带宽公式外推的估算;公开单卡 4090 实测多为 Q4 量化下的更高值,如 llama.cpp Q4 约 104–135 tok/s)。用户当前 18.2 tok/s 仅为单卡潜力的约 1/3,差距几乎全部来自 13/15 层间流水线切分导致的逐 token 跨卡传输。

**3. BAGEL 有现成的单卡量化/压缩版本,但对 batch=1 延迟不一定有利。** 社区已发布 `DFloat11/BAGEL-7B-MoT-DF11`(无损压缩,比 bf16 小 32%、压缩权重实际 20.2GB、输出比特级一致,"BAGEL can now run smoothly on a single 24GB GPU without any quality loss"),以及 NF4/INT8(bitsandbytes)量化。但 DFloat11 在 batch=1 时因逐前向解压反而比原生 bf16 慢(A100 上约慢 40%–2×)。对 think 阶段而言,原生 bf16 的非对称单卡放置比 DF11 更快。

**4. 免训练投机解码在 BAGEL 上原则可行但有词表兼容风险。** BAGEL 文本专家就是 Qwen2.5-7B,理论上可直接用同族 Qwen2.5-0.5B/1.5B-Instruct 作 draft model(投机解码本身免训练,只需词表兼容)。文献中 Qwen2.5-Coder-0.5B 对 7B 贪心解码可达 2.64×,0.5B 对 14B 一般任务约 1.4×。但已知风险:Qwen2.5 的 0.5B/1.5B/3B 的 vocab(151936)与 7B/BAGEL(152064)不一致,需处理 embedding 对齐。

**5. vLLM 已支持 BAGEL 但仅限「图像理解(vision-to-text)」,图像生成不支持。** 这意味着可以把 think(纯文本自回归)阶段单独放到 vLLM/SGLang 跑(享受 CUDA Graph、投机解码、优化 attention 内核),图像生成阶段仍用原实现——即「两段式部署」。SGLang 原生支持 NGRAM 与 STANDALONE 投机解码。

## Details

### 方向 A:免训练投机 / 并行解码

| 方法 | 是否免训练 | batch=1 贪心实测加速(文献) | 对 BAGEL think 的适配性 |
|---|---|---|---|
| Draft-model 投机(Qwen2.5-0.5B/1.5B) | 是* | 1.4–2.64×(Qwen 系) | 高:同族模型;*风险=0.5/1.5B 与 7B 词表不一致 |
| Prompt-lookup / n-gram | 是 | 2–2.4×(输入重叠高时);CoT 场景低 | 中:CoT 与 prompt 重叠少,增益偏小 |
| Lookahead Decoding | 是 | 1.5–2.3×(代码类高) | 中:免 draft、免 datastore,可直接接 HF |
| SWIFT / 自投机层跳过 | 是 | 1.3–1.6× | 中:即插即用,大模型层稀疏性更明显 |
| Jacobi(vanilla) | 是 | ≈1.05× | 低:裸 Jacobi 几乎无加速 |

关键点:所有投机/并行方法在贪心下都能保持输出与自回归一致(lookahead、投机解码为精确方法;SWIFT 在贪心下 LLaMA2 系接受率 98%–100%)。draft-model 投机的加速对接受率极敏感,接受率跌破约 0.5 后收益骤降;draft 越贴近 target(同族、同分布)接受率越高。

BAGEL 官方推理基于自研代码(非纯 HF `generate`),已实现 KV cache(paper 明确 think 阶段文本走 Next-Token-Prediction 自回归)。接入 HF 风格的 assisted generation / lookahead 需要一定改造,但 think 阶段是标准因果 LM,改造量可控。

### 方向 B:消除双卡 PCIe 逐 token 开销

**首选:非对称切分(免训练、免量化)。** 由子代理据 config.json/safetensors 张量形状精确核算:理解专家(28 层 + embed + lm_head,未 tie)≈7.616B,bf16 ≈15.2GB;生成专家 ≈6.53B;ViT(siglip-so400m/14,27 层)≈0.42B;VAE(FLUX ae.safetensors 335MB)≈84M。三者相加 14.14B ≈ 官方「14B total」,自洽。把理解专家(+可选 ViT ≈+0.85GB)放卡 A,生成专家+VAE 放卡 B。think 阶段全程单卡,零跨卡通信。注意:BAGEL 官方代码未提供这种一级切分,需手动把 `*_moe_gen` 张量与 VAE 映射到第二张卡。

**若坚持全模型单卡:** 用 DFloat11(20.2GB,比特级无损)或 NF4。但 batch=1 下 DF11 比 bf16 慢,NF4 有质量与速度权衡;think 延迟视角不推荐,除非显存实在放不下非对称方案。

**张量并行 vs 流水线并行(PCIe 无 P2P):** 文献共识明确——PCIe 环境张量并行的逐层 all-reduce 在关键路径上(每 transformer block 两次 all-reduce),NVLink 约 900 GB/s 而 PCIe 好情况仅约 64 GB/s,通信开销可占推理时间相当大比例,「不推荐用于延迟敏感场景」;PCIe 只适合流水线并行(传小激活张量)。但对 batch=1 逐 token,流水线并行无法填满流水线(无微批),两卡实际串行,这正是当前 0.055 s/token 斜率偏大的根因。**结论:最优解是根本不跨卡(非对称单卡放置),而非在两种并行间二选一。**

### 方向 C:解码 kernel 与运行时优化

- **CUDA Graph(batch=1 收益最大的场景):** 消除逐 kernel 的 CPU dispatch。Fireworks 实测 LLaMA-7B 单卡 A100 从 30→69 tok/s(2.3×,收益完全来自 CPU 开销削减);另一 44-cell 跨 GPU 研究测得 H100 上 Qwen2.5-7B ctx2048 约 1.26×、L4 仅 1.028×(慢卡带宽受限,launch 开销被掩盖)。4090 属高带宽卡,launch 开销可见,预计 1.2–1.5×。HF transformers 需用 static KV cache + `torch.compile` 才能启用 graph 捕获。
- **torch.compile / FlashAttention:** 对单 batch 解码有正收益,主要来自减少 Python dispatch 和稳定 kernel 执行。
- **框架路线(两段式部署):** vLLM 已有 `bagel.py`(仅理解,`vllm/model_executor/models/bagel.py`,官方注明「image generation part is not supported」),SGLang Diffusion 亦点名支持 BAGEL 类 AR+diffusion。可行做法:think 文本阶段走优化框架(CUDA Graph + 投机),图像阶段回原实现。学术系统 vLLM-Omni、M* 已把 BAGEL 拆成多 stage 服务。
- **距硬件上限的空间:** 当前 18.2 tok/s vs 单卡 bf16 估算上限约 33–55 tok/s——单是消除跨卡就有约 2–3× 空间;再叠加 CUDA Graph 与投机,尚有可观余量。

### 方向 D:需轻量训练的备选

- **Hyper-Bagel(ByteDance Seed 官方,Lu et al., arXiv:2509.18824):** 直接在 BAGEL 上主要沿用 EAGLE-3 训练范式训练轻量 draft。论文正文原文:朴素套用 EAGLE-3 在 BAGEL 上效果不佳——"community reproductions on the latest VLM Qwen3 achieved only a 1.7x speedup in TPS, diverging sharply from the 4-5x accelerations typically observed in LLMs such as Vicuna, LLaMA, and DeepSeek";原因是多模态 token 嵌入空间差异大,且 BAGEL 还需处理扩散去噪后 prefill 的 clean latent token。作者设计中间层聚合 target 特征后,在单卡 A100 SGLang chain decoding 下 TPS 从 98.3 提升到约 212.4(2.16×);论文摘要对外口径为「over a 2x speedup in multimodal understanding」。这证明 BAGEL 投机解码可行且官方已验证。
- **EAGLE-3 / Medusa 训练成本:** EAGLE-3 8–9B 级约 48–128 H200-GPU-hour(SpecForge 参考配置),社区单卡 head 训练可低至约 1.5 小时(如 GLM-4.7-Flash EAGLE3 head,277MB,单 H100 训 1h26m);Medusa/Hydra 更贵(768–1800 GPU-hour @ Qwen3-8B)。draft head 通常 1–3B、几百 MB,可与 target 同卡部署。

## Recommendations

**阶段 0(立刻做,免训练,1–2 天):非对称单卡放置。** 改 device_map,让理解专家整体驻留卡 A、生成专家+VAE 驻留卡 B。基准:若 think 段 tok/s 未从 ~18 提升到 ~35+,说明放置或显存有问题需排查(优先确认 `*_moe_gen` 张量确已全部映射到卡 B,think 路径无任何算子落在卡 B)。这是投入产出比最高的一步。

**阶段 1(免训练,3–5 天):叠加 CUDA Graph + static KV cache。** 在单卡 think 路径上启用 static cache + torch.compile 的 graph 捕获。基准:再获 ≥1.2×。若无提升,检查是否触发 graph recompile(变长序列需用固定长度 bucket)。

**阶段 2(免训练,1–2 周):接入投机解码。** 优先 Qwen2.5-1.5B-Instruct 作 draft(先验证其 vocab 是否与 BAGEL 的 152064 兼容;Qwen2.5 小模型多为 151936,须处理 embedding/词表对齐);若词表冲突无法解决,退回 **Lookahead Decoding**(完全无 draft、无词表问题)或 SGLang NGRAM。基准:接受率 >0.6 时应得 ≥1.5×;接受率 <0.5 则放弃该 draft 换更贴近的。

**阶段 3(可选,需轻量训练):若免训练方案仍不达标,复现 Hyper-Bagel 的 EAGLE-3 路线。** 成本约几十 GPU-hour(参照 SpecForge/EAGLE-3 8–9B 级 48–128 H200-GPU-hour),官方已验证 >2×(约 2.16×),是投机解码的上限路线。

**触发阈值:** 若阶段 0–2 已把 t_think 降到 ~0.012 s/token(约 80 tok/s)且满足业务延迟,则无需进入阶段 3;若长思考(1000 token)端到端仍不可接受,再上 EAGLE-3。

## Caveats
- 所有加速比为基于文献实测外推到「7B / batch=1 / 贪心 / 4090」的估算,非 BAGEL 实测;BAGEL 自研解码循环可能与标准 HF 路径有差异,实际需 profile 验证。
- 单卡 bf16 上限 33–55 tok/s(FP16)为据 1008 GB/s 带宽公式外推的估算,公开 4090 实测多为 Q4 量化(约 104–135 tok/s),非同精度直接可比。
- Qwen2.5 小模型词表(151936)与 BAGEL/7B(152064)不一致是 draft 投机的真实风险,须先验证再投入。
- DFloat11 batch=1 慢于 bf16 的数据来自 A100,4090 上具体幅度未实测。
- CUDA Graph 在 4090 上的收益幅度为跨 GPU 外推(H100 1.26×、A100 2.3×,跨度大),需实测。
- 非对称切分的参数量与显存(理解专家 ≈15.2GB)由子代理据 config/safetensors 张量形状计算,与官方「7B active/14B total/~15B」headline 自洽,但 ViT(≈0.42B)/VAE(≈84M)为估算(±数个百分点)。
- Hyper-Bagel 的 98.3→212.4 TPS(2.16×)取自论文正文引言的自研实现叙述(单卡 A100、SGLang chain decoding);对外摘要口径为「over 2x」。
- vLLM 的 BAGEL 支持仅覆盖理解(vision-to-text),不含图像生成;两段式部署需自行拼接 think(框架)与图像生成(原实现)两段。