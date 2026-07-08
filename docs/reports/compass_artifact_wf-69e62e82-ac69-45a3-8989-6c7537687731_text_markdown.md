# 多模态 KV Cache 压缩(图像 token)在 BAGEL-7B-MoT 推理加速中的适用性评估

## TL;DR(该不该做、什么时候做、做哪个方法)

**结论:在你当前的设定(batch=1、KV 仅 3~7k、decode 已被实测证明不受 KV 长度约束、KV 总量 <1 GiB)下,图像 token KV 压缩对"单请求 think 延迟"几乎没有收益,应排在阶段1(CUDA Graph + static KV)和阶段2(投机解码)之后,作为 P2/机会性优化而非主线。** 原因是:文献中几乎所有多模态 KV 压缩(FastV、VisionZip、LOOK-M、VL-Cache、MEDA 等)的加速来自两处——(a)prefill 阶段减少注意力/FLOPs,(b)decode 阶段减少 KV 读取带宽或扩大 batch——而你的 t_think≈55s 瓶颈既不在 KV 读取(KV 翻倍 decode 只变 0.5s),也不在显存容量。唯一在你场景中真实存在的收益是 **prefill 阶段的一次性提速**(把 3~7k 中的图像 token 砍半可省一部分 TTFT),这是 think 之前的一次性成本,占端到端比例有限。**如果要做,唯一值得做的是 prefill 期的一次性视觉 token 剪枝(FastV / VisionZip 思路,且在进入 LLM 之前对 ViT token 下手),而不是 decode 期的 KV 驱逐类方法。** 触发"重新评估并提升优先级"的条件:转向 batch>1 服务、多图/多轮编辑导致 KV 累积到数万 token、或部署目标显存降到放不下 CFG 多份上下文时。特别提醒:编辑任务里 VAE latent token 是生成分支的 CFG 条件,**绝不能用通用文本 KV 驱逐方法盲压**,否则直接损伤编辑质量。

---

## 方法对比表

| 方法 | 类型 | 免训练? | 报告压缩率/保留率 | 精度损失(理解 benchmark) | 收益来源 | 适配你场景的判断 |
|---|---|---|---|---|---|---|
| **FastV** (ECCV'24 Oral) | 视觉 token 剪枝(LLM 第2层后) | 是 | ~45% FLOPs 削减(K=2,R=50%,LLaVA-1.5-13B) | 基本无损(单图理解;A-OKVQA 82.0→81.3) | prefill FLOPs;decode 需开 KV-cache 变体 | prefill 一次性有用;官方 README 明确单图开 KV-cache "仅约8%延迟下降,视频(约10倍图像 token)才达25%" |
| **PyramidDrop** (CVPR'25) | 分层递进视觉 token 丢弃 | 是(也可训练) | LLaVA-NeXT 推理 FLOPs 减55%、训练时间减40%,推理加速最高2.22× | 可忽略 | prefill 计算 | prefill 有用;深层丢弃对短 KV 收益递减 |
| **VisionZip** (CVPR'25) | 视觉 token 选择+合并(进 LLM 前) | 是(30分钟微调更优) | 保留10% token→约95%性能;LLaVA-NeXT-7B prefill 提速8× | 保留10%时约损5% | prefill(text-agnostic,进 LLM 前,兼容 FlashAttention) | **最契合**:进 LLM 前砍 ViT token,直接缩短 prefill |
| **SparseVLM / LLaVA-PruMerge** | 视觉 token 剪枝/合并 | 是 | 保留~64~192 token | 中低预算下下降明显 | prefill | 同类,收益同 prefill |
| **LOOK-M** (EMNLP'24 Findings) | 模态感知 KV 驱逐+合并(text-prior) | 是 | KV 内存降80~95%、decode 提速1.3~1.5×(LLaVA-1.5-7B/13B,MileBench) | MileBench 上极小,甚至超 full | decode KV 带宽+显存 | **收益在 decode/长上下文,你场景不存在**;且 text-prior 会丢视觉 KV |
| **MEDA** (NAACL'25) | 熵引导动态分层 KV 分配 | 是 | KV 降最高72%,最高2.82×加速 | 维持或提升 | decode KV 带宽+显存(长上下文多图) | 收益在长上下文;你 KV 太短 |
| **VL-Cache** (ICLR'25) | 稀疏+模态感知 KV 预算分配 | 是 | 保留10% KV→98%精度;端到端最高2.33×,decode 最高7.08× | ~2% | decode KV 带宽;**作者明确端到端受 prefill 约束** | decode 收益需长输出;你 decode 非 KV-bound |
| **MadaKV** (2025) | 模态自适应 KV 驱逐 | 是 | decode 延迟改善1.3~1.5× | 高精度维持 | decode KV 带宽(长上下文) | 同上,不适配 |
| **SnapKV / H2O** | 通用 KV 驱逐(heavy-hitter) | 是 | KV 降最高92% | 长上下文可近无损 | decode KV 带宽;固定 cache 使 decode 恒速 | 收益随输入序列长/batch 才显现;你短 KV+batch1 无收益 |
| **KIVI / KVQuant** | KV 量化(2-bit) | 是(免训练) | 2.6×峰值内存↓,batch 最高4×,吞吐2.35~3.47× | 近无损(GQA/小 KV 模型2-bit 损失更大) | 显存→更大 batch/吞吐 | **收益是吞吐/显存,batch1 无延迟收益;GQA 小 KV 2-bit 更易掉点** |
| **G²TR** (2026,直接在 BAGEL 上测) | 生成分支引导的 ViT token 削减(进 LLM 前) | 是 | 保留50% ViT token;prefill FLOPs 1.94×↓、KV 1.90×↓ | 理解 rel. 99%;编辑 rel. 98% | prefill 计算+KV 体积 | **唯一直接针对 BAGEL 类分离编码器 UMM;但 decode 仅1.04×,证实 decode 无收益** |

---

## 逐条回答

### 1. 方法学全景(2023–2026)

多模态/视觉 token 的 KV 压缩方法可分为三大类,加上一个专门的第四类(统一生成模型):

**(A)视觉 token 剪枝/合并类(在进入 LLM 或 LLM 浅层处削减 token 数)。** 代表:FastV(ECCV'24 Oral,发现视觉 token 在 LLM 深层注意力极稀疏,第2层后剪枝可省约45% FLOPs——具体在 LLaVA-1.5-13B 上 K=2/R=50% 时,层3~32 视觉 token 从 576→288,理论 FLOPs 削减45%、A40 上实测延迟 0.539s→0.341s,约36%,免训练);PyramidDrop(CVPR'25,分层递进丢弃,LLaVA-NeXT 推理 FLOPs 减55%、推理加速最高2.22×,浅层必须保留全部视觉 token,冗余随层加深);VisionZip(CVPR'25,进 LLM 前选择主导 token 并合并,保留10% token 达约95%性能、LLaVA-NeXT-7B prefill 提速8×,text-agnostic 且兼容 FlashAttention);SparseVLM、LLaVA-PruMerge 及后续 VisionTrim/SCOPE 等。这类**都是免训练、收益主要在 prefill 计算与 TTFT**,附带缩小 KV。关键局限:FastV 依赖显式注意力图、动态剪枝,与 FlashAttention/KV-cache 管理不兼容(需专门变体);而在进 LLM 前剪枝(VisionZip、LLaVA-PruMerge)才能兼容 FlashAttention。

**(B)模态感知的 KV 驱逐/合并类(在 decode 期管理 KV)。** 代表:LOOK-M(EMNLP'24 Findings,首个多模态 KV 压缩,text-prior 驱逐+合并,在 LLaVA-1.5-7B/13B 上跑 MileBench 时"KV 内存降 80% 到 95%、decode 提速 1.3x 到 1.5x");MEDA(NAACL'25,注意力熵引导动态分层分配,最高2.82×加速/KV 降72%);VL-Cache(ICLR'25,稀疏+模态感知预算分配,保留10% KV 达98%精度、端到端最高2.33×、decode 最高7.08×);MadaKV(2025,模态偏好自适应,1.3~1.5×)。**这类的加速全部在 decode 期 KV 读取带宽,且论文均在长上下文/多图 benchmark 上报告**——例如 MileBench 共 6,440 个长上下文样本、平均每样本 15.2 张图 / 422.3 词(单样本最多达 109 图、11,821 词),与你"1~2 张图、prompt 数千 token"的编辑场景相差一到两个数量级。

**(C)通用 KV 压缩在多模态下的表现。** SnapKV、H2O(heavy-hitter 驱逐,KV 降最高92%,固定 cache 使 decode 恒速);KIVI/KVQuant(2-bit 量化,KIVI 报告"含权重在内峰值内存降 2.6×,从而支持最高 4× 的 batch,在真实推理负载上带来 2.35× 到 3.47× 吞吐")。SnapKV 的加速图明确显示:**收益随输入序列长度和 batch 增大才出现**——16k 长度/batch=2 时约3.6×,短序列基本无收益。

**(D)统一生成模型专用(与 BAGEL 最相关)。** G²TR(2026,上海交大+华为)专门针对分离编码器统一多模态模型(UMM),**主实验就在 BAGEL-7B-MoT 上做**:用生成分支(VAE latent,2×2 均值池化后的锚点)作为引导只削减理解侧 ViT token(保留率 ρ=0.5),prefill FLOPs 从 4.13T→2.13T(1.94×)、KV 从 42.02MB→22.06MB(1.90×),理解 benchmark 相对保持99%、编辑相对保持98%。UniCompress(CVPR'26)指出:朴素下采样/均匀剪枝对理解有效,但**对生成任务性能下降超过15%**。

**每类与你场景的差距:** (A)类收益在 prefill,你 think 前有一次 prefill,故有部分收益;(B)(C)类收益在 decode KV 带宽或显存→更大 batch,而你 decode 非 KV-bound、batch=1、KV<1 GiB,**这些收益在你场景中系统性不存在**。

### 2. 收益归因分析:哪些收益在你的设定下不存在

多模态 KV 压缩的加速来自三个互相独立的来源,逐一对照你的实测:

- **(a)prefill 计算量/注意力 FLOPs↓** → **真实存在,但为一次性**。你带图 prompt 有3~7k token,其中图像 token 占大头。进 LLM 前把 ViT token 砍半(如 G²TR/VisionZip)确实能减少 prefill 的 attention 与 MLP 计算。G²TR 在 BAGEL 上测得 prefill FLOPs 1.94× 降低。但注意:这是 think 自回归解码**之前**的一次性成本,而你的 t_think≈55s 是1000 步 decode 的累积,prefill 占端到端比例有限。
- **(b)decode 期 KV 读取带宽↓** → **在你场景中不存在**。你的实测已经证明:KV 从3254翻倍到7121,t_think 只从54.5s→55.0s(<1% 变化),说明 decode 是权重搬运/kernel launch 瓶颈(典型 batch=1 小 KV 的 memory-bound-on-weights 而非 memory-bound-on-KV)。G²TR 在 BAGEL 上的独立测量佐证:砍掉50% ViT token 后 decode 仅从49.60→47.88 ms/token(1.04×),几乎无变化。值得注意的是,连 VL-Cache 自己也承认 decode 提速会被 prefill 稀释——其原文指出"端到端加速受 prefill 延迟约束……在 128K prompt、batch=1 时,7.08× 的 decode 加速被占端到端 53% 的 prefill 稀释,最终只有 1.66× 端到端"。**所有以 decode 提速为卖点的方法(LOOK-M/MEDA/VL-Cache/SnapKV/KIVI)在你的场景收益为零。**
- **(c)显存容量↓→更大 batch/更长上下文** → **在你场景中不存在**。你 KV 总量含 CFG 多份也 <1 GiB,24GB 显存完全放得下;batch=1 无需更大 batch。KIVI 的 batch 4×、吞吐2.35~3.47× 收益全部依赖"同显存塞更多请求",对单请求延迟无意义。

**结论:三个收益来源中,只有 (a) prefill 在你场景成立,且是一次性、占比有限;(b)(c) 系统性不存在。**

### 3. BAGEL 特有约束(标注推断)

**(3.1)VAE latent token 是生成分支的 CFG 条件——压缩风险高【关键,部分推断】。** 据 BAGEL 论文(§2.3 Generalized Causal Attention)直接陈述:推理时"仅存储 clean VAE token 和 ViT token 的 KV";编辑时输入图同时被 ViT 和 VAE 编码,**两类 token 都进入 KV cache**,后续文本(think)解码与图像生成都会注意它们。BAGEL 编辑用**双 CFG**:官方 inference notebook 确认 `cfg_text_scale=4.0` + `cfg_img_scale=2.0`,这是嵌套式引导(多篇复现工作如 UniT/Shape-of-Thought 均描述为 v_text = v_unc + 4.0·(v−v_unc);v_final = v_img_unc + 2.0·(v_text−v_img_unc)),每个去噪步需在**多份上下文副本**(全条件 / 去文本条件 / 去图像条件)上前向。因此:**丢弃或压缩 VAE latent token 会直接改变 image-CFG 的无/有条件差,损伤编辑保真**。目前专门处理"KV 同时服务文本解码和扩散/流生成条件"的工作极少;最接近的是 G²TR,其做法恰恰是**只动 ViT token、把 VAE latent 当作不可动的引导信号**。视觉自回归生成侧另有 HeatKV、Forcing-KV、Entropy-Aware(ICML'26)等,但它们压的是"生成过程中自身累积的视觉 KV",与"输入图作为编辑条件"不是同一问题。【推断:VAE 压缩损伤编辑质量是基于 CFG 机制与 UniCompress"生成任务下降>15%"的外推,BAGEL 上无直接消融】

**(3.2)ViT token 与 VAE token 应区别对待——是。** G²TR 及其对分离编码器 UMM 的分析明确:纯图像生成不涉及理解侧 ViT token,**只有图像编辑会被理解侧 ViT token 削减影响**;而 ViT token 只服务理解/语义、VAE token 服务生成。BAGEL 论文也报告:去掉 ViT token 对 GEdit-Bench 影响很小,但 Intelligent Edit 掉约16%,说明 ViT token 对"智能编辑"有实质贡献但对基础编辑冗余较高。**因此正确策略是:优先且只压缩 ViT 理解 token(冗余度证据充分),VAE latent token 保守保留。** token 规模上,据 SigLIP2-so400m/14(max 980)与 FLUX VAE(÷8 后 2×2 patch,净 ÷16)参数估算:ViT token 在 980² 时约 4,900 个,VAE latent 在 1024² 时约 4,096 个,两者量级相当【此为按 patch 参数计算的估算,非论文直给】。

**(3.3)GQA 只有4个 KV 头,KV 本来就小——文献压缩率多在 MHA 上报告【部分推断】。** BAGEL 骨干 Qwen2.5-7B 是 GQA(28 query 头 / 4 KV 头),你实测 KV≈57 KB/token,7k token≈0.4 GiB。GQA 本身已经是7:1 的 KV 压缩。文献中很多高压缩率(如 KIVI 的 batch 4×)是在 MHA 或 KV 头更多的模型上报告的;LogQuant 论文明确指出**KV 头更少的模型(如只保留1/8 KV 头)在2-bit 下精度损失更显著**。这意味着:在 GQA+已经很小的 KV 上,再叠加 KV 量化/驱逐的边际收益更小、精度风险更高。【推断:GQA 下边际收益更小是基于"KV 已被 GQA 压过一轮"+ LogQuant 观察的外推】

### 4. 与既定路线图的交互

**(4.1)对阶段1(CUDA Graph + static KV bucket):以冲突为主,但有一个正向例外。** CUDA Graph 要求**固定张量形状**(NVIDIA 官方:graph 拓扑必须静态,变长序列只能靠 padding 或 bucketing)。而绝大多数动态视觉 KV 压缩(FastV 动态剪枝、LOOK-M/SnapKV 按注意力驱逐)会产生**运行时可变的 KV 长度**,与 static KV cache/CUDA Graph 捕获直接冲突——文献(KV-RM,2026)专门讨论了"动态 KV 驱逐 vs. 静态图重放"的张力,解决方案要么 padding 要么多 bucket。**但有一个例外且正向的组合:如果视觉 token 剪枝在 prefill 一次性完成(进 LLM 前,如 VisionZip/G²TR),得到的是一个"更短但仍然固定"的 KV 长度**,这不但不冲突,反而能**降低 static bucket 的尺寸上限**——把最坏情况 7121 降到约 4000~5000,让 CUDA Graph 捕获的 bucket 更小更省。这是唯一与阶段1协同的方式:一次性 prefill 剪枝 → 确定长度 → 再进 static bucket。

**(4.2)对阶段2(免训练投机解码):中性到轻微正向。** 投机解码的收益本身来自"把 memory-bandwidth-bound 的 decode 变成 compute-bound 的并行验证",这与你 decode 瓶颈(权重搬运/kernel launch)高度契合,是比 KV 压缩更对症的方案。KV 压缩对投机解码基本中性:draft 与 target 都要维护 KV,draft token 会临时增长 KV 页(文献指出这与分页 KV 分配交互不佳);但你 KV 很小,这点无关紧要。若采用 self-speculative/EAGLE 类,draft KV 更小。**结论:阶段2与图像 KV 压缩不冲突,但两者都对 decode 相关,而只有投机解码真正命中你的瓶颈。** 另有一篇 CVPR'26 工作 HiViS("Hiding Visual Tokens from the Drafter for Speculative Decoding in VLMs")专门处理 VLM 投机解码中 draft 如何不背负视觉 token,值得在阶段2一并参考。

**优先级总排序(针对你的瓶颈):阶段2投机解码(命中 decode 瓶颈)≈ 阶段1 CUDA Graph(命中 kernel launch)> prefill 期视觉 token 剪枝(命中一次性 TTFT)>> decode 期 KV 驱逐/量化(在你场景无收益)。**

### 5. 间接收益场景与"重新评估"触发条件

图像 token KV 压缩会在以下条件下从"鸡肋"变为"重要",应设为明确触发器:

- **多图输入**:编辑参考图从1张增到多张,ViT+VAE token 线性增长,prefill 成本和 KV 都放大,prefill 剪枝收益变显著。**触发阈值:输入图 ≥3 张,或图像 token 总数 >8k。**
- **多轮编辑对话累积 KV**:每轮把生成图的 clean VAE+ViT token 追加进 KV,KV 会累积到数万 token,此时 decode 可能真正进入 KV-bound 区间。**触发阈值:累积 KV >20k token,或实测 decode 速度随轮次明显下降(如 t_think 随 KV 翻倍变化 >10%)。**
- **batch>1 服务化**:一旦离开 batch=1,KV 显存成为 batch 上限的约束,KIVI/VL-Cache 类的显存→吞吐收益立即成立。**触发阈值:目标 QPS 要求 batch ≥4,或显存成为并发瓶颈。**
- **更小显存部署目标**:若目标从 2×4090 24GB 下移到单卡 16GB/12GB,CFG 多份上下文+权重可能吃紧,KV 量化/剪枝成为可行性问题而非优化问题。**触发阈值:部署显存 <16GB 且需保留 CFG 多份上下文。**

---

## 明确建议

**优先级排序(相对既定路线图):**
1. **保持阶段1(CUDA Graph + static KV)、阶段2(投机解码)为主线不变。** 这两者分别命中你的 kernel-launch 和 decode 权重搬运瓶颈,是对症方案。
2. **将"prefill 期视觉 token 剪枝"作为阶段1的一个子项/协同项**,而非独立阶段——因为它能顺带降低 static KV bucket 尺寸上限。推荐采用 **G²TR 的思路(只削 ViT 理解 token、保留 VAE latent、用生成分支引导),因为它是唯一在 BAGEL 上验证过的方案(理解99%、编辑98%、prefill 1.94×)**;若嫌 G²TR 复杂,退化为 VisionZip 式的进-LLM-前 ViT token 选择(保留约50%)。
3. **明确不做**:decode 期 KV 驱逐(LOOK-M/SnapKV/MEDA)、KV 量化(KIVI/KVQuant)——在 batch=1、短 KV、decode 非 KV-bound 下无延迟收益,且 GQA 下精度风险更高。**尤其不要对 VAE latent token 做任何有损压缩。**

**第一个验证实验设计(2×4090,复用现有 benchmark 脚本):**

- **目标问题**:量化"prefill 期把 ViT token 保留率从100%降到50%/25%"对(i)TTFT、(ii)端到端延迟(TTFT+t_think)、(iii)编辑质量的影响,验证 prefill 收益占比是否值得工程投入。
- **实验组**:baseline(全 token)vs. ViT token 保留50% vs. 保留25%;VAE latent 始终100%保留(对照 BAGEL 双 CFG)。
- **测量点**(复用你现有脚本的计时结构):分别打点 prefill 时间、think decode 时间、图像生成时间;KV 长度;峰值显存。重点看 **prefill 节省的绝对秒数 ÷ 端到端秒数**——若 <5%,则确认此方向对单请求延迟为鸡肋。
- **质量评估**:在你真实业务的"带参考图编辑+think"样本上跑,主观比对编辑保真(是否因 ViT token 减少而丢失参考图细节),并在小规模 GEdit/IntelligentBench 子集上对齐 G²TR 报告的编辑 rel. 98% 掉点趋势。
- **静态化验证**:测试剪枝后 KV 长度是否稳定落入更小的 static bucket(如从7121 上限降到约4000),据此评估阶段1 CUDA Graph 捕获的收益。
- **决策阈值**:若50%保留下编辑质量掉点 <2%(对齐 G²TR)且 prefill 节省转化为端到端 >8% 提速 → 排进路线图并入阶段1;若端到端提速 <5% → 搁置,标记为"多图/多轮/batch>1 时重估"。

---

## Caveats(依赖外推而非直接文献证据的部分)

1. **"decode 在你场景完全无 KV 收益"** 主要基于你自己的实测(KV 翻倍 t_think 仅变0.5s)+ G²TR 在 BAGEL 上 decode 仅1.04× 的独立佐证。这是强证据,但两者都是特定配置;若换 attention kernel 实现或超长 KV,结论可能变化。
2. **"VAE latent token 压缩会损伤编辑质量"** 是基于 BAGEL 双 CFG 机制 + UniCompress"生成任务朴素压缩下降>15%"的**外推**,BAGEL 上没有"压 VAE token"的直接消融。属推断,需实验验证。
3. **BAGEL 编辑任务中 ViT vs. VAE 的具体 token 数**(ViT@980²≈4900、VAE@1024²≈4096)是根据 patch/下采样参数的**计算估算**,论文未直接给出每图 token 数,实际随理解/生成分辨率变化。
4. **"GQA+小 KV 下压缩边际收益更小、量化精度风险更高"** 基于 LogQuant 关于"KV 头更少模型 2-bit 掉点更多"的观察外推,未见 BAGEL/Qwen2.5-7B 上的直接 KV 量化实验。
5. **G²TR 的数据**来自2026年的近期 preprint(arXiv 2605.12309),在单张 A6000 上"each run once"(未报告多次均值/方差),其绝对数值应视为方向性证据而非精确基准;其 prefill "1.94×" 是 FLOPs/计算指标,非端到端墙钟时间;其 KV cache 报告值(42→22MB)是单序列不含 CFG 多份的量。
6. **CUDA Graph 与动态 KV 压缩冲突**的结论对"静态一次性剪枝"不适用——一次性 prefill 剪枝反而有利,这一区分是本报告的推理判断,基于 CUDA Graph 固定形状约束的通用原理。
7. **FastV 官方 README** 的"单图开 KV-cache 约8%延迟下降、视频约25%"是在 LLaVA 系列单图理解任务上测的,BAGEL 编辑+think 的 token 构成不同,该比例仅作方向参考。