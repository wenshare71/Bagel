# BAGEL 模型的 Profile 分析

BAGEL 7B-MoT 在 **t2i think 模式**(先自回归生成 `<think>...</think>` 推理文本,再流匹配去噪出图)下的端到端延迟剖析。基于三轮实验共 2688 个有效 trial(分辨率三维 sweep 1920 + budget forcing 补跑 768,另有前期 think-cap sweep 480 用于交叉验证),得到一条可组合的延迟定律:

```
t_total ≈ t_prefill + 0.055·T + s(R)·N

T    = think 文本 token 数
R    = 图像边长(image token 数 = (R/16)²)
N    = 去噪步数(蒸馏步数)
s(R) = 每步去噪成本 ≈ 0.0718 + 1.10e-4 · (R/16)²   [s/step]
```

**TL;DR**:think 与 image 两段完全独立、各自严格线性。谁是瓶颈由 `T* = s(R)·N / 0.055` 决定:T > T* 时 think 主导。默认配置(R=1024, N=50)下 T* ≈ 486,自然思考(~162 token)只占总时长 25%;但一旦叠加**蒸馏(N=5~10)+ 长思考(T=1000)**,think 占比可达 91%~99%,优化重点应从扩散步数彻底转向文本解码。

---

## 1. 机器环境

所有绝对时间数字都依赖以下环境,复现或外推前先读这一节。

| 项 | 配置 |
|---|---|
| GPU | 8× NVIDIA RTX 4090 24 GB,单机,PCIe 互联(无 NVLink / P2P) |
| 模型 | BAGEL-7B-MoT,bf16 权重 ~28 GB,单卡放不下 |
| 部署 | 每个 worker 占 2 卡:手动均衡 device_map(LLM 层 13/15 切分,VAE/ViT 及词嵌入按依赖就近放置),层间流水线跨 PCIe |
| 并行 | 4 workers 并行跑 trial(`CUDA_VISIBLE_DEVICES` 子进程隔离),互不共享显存 |
| 推理 | batch = 1,贪心解码(`do_sample=False`),TaylorSeer 关闭 |
| 计时 | `torch.cuda.synchronize()` + `time.perf_counter()`,分段记录 t_prefill / t_think / t_image;每个 worker 对**每档分辨率单独 warm-up**(kernel 按 shape 缓存,只 warm 一档会污染其余分辨率的首个 trial) |
| 负载 | 16 条 prompt(GenEval 8 + WISE 8,固定随机种子采样),每个条件 × 2 repeats |

t_prefill(系统提示 + prompt 的一次性前向)为与 T、N、R 基本无关的小常数项,本文的占比分析只比较 t_think 与 t_image 两个主段。

## 2. 局限

- **绝对数字不可直接外推。** 2 卡 PCIe 流水线给逐 token 解码引入跨卡通信开销,0.055 s/token 的斜率相对单卡或 NVLink 系统**偏大**;可迁移的是各定律的函数形式和相对比例,不是系数本身。
- **batch=1 的延迟视角**,不代表服务化吞吐场景(连续批处理下结论会不同)。
- **长 think 数据来自 budget forcing。** 模型自然思考只有 ~162 token,cap=256/1000 两档是用 s1 式 budget forcing(EOS → " Wait," 续写)把长度精确钉在 cap 上测得的。强制续写内容有重复("车轱辘话"),对延迟测量无影响(解码耗时只由 token 数决定),但**不能用这批数据评价长思考的质量收益**。
- **低分辨率的质量代价未评估。** BAGEL 训练分辨率为 1024(`max_latent_size=64` 恰好卡在 64×16),R<1024 时出图质量可能明显下降,本文只回答延迟问题。
- **s(R) 只有 4 个标定点**,二次项拟合自由度为 1,attention 超线性只能定性判断;更严格需要非方形 shape(如 512×1024,tok=2048)补点。

## 3. 实验设计

三维全交叉网格,60 个条件 × 16 prompts × 2 repeats = 1920 trials,全部成功:

| 自变量 | 取值 |
|---|---|
| 图像边长 R | 1024, 768, 512, 256(上限 1024,只能向下扫;须为 16 的倍数) |
| 去噪步数 N | 50, 10, 5 |
| think cap T | 1000, 256, 128, 64, 32 |

其中 cap∈{256, 1000} 两档用 budget forcing 补跑批次(768 trials,min=max=cap)替换,使 think 长度精确等于 cap;cap≤128 本来就被自然长度(~162)截断钉死,无需强制。

## 4. 文本 token 长度(think 段)

**定律:`t_think ≈ 0.055 · T`,与 N、R 均无关。**

| cap | 实际 token 数 | t_think(4 档 R 平均) | 换算 s/token |
|---|---|---|---|
| 32 | 31 | 1.76 s | 0.0568 |
| 64 | 63 | 3.52 s | 0.0559 |
| 128 | 127 | 7.06 s | 0.0556 |
| 256(forcing) | 255 | 14.08 s | 0.0552 |
| 1000(forcing) | 999 | 55.06 s | 0.0551 |

- 斜率收敛到 **0.0552 s/token**(短序列略高,来自固定启动开销的摊薄);
- **自然思考长度只有 ~162 token**(t_think ≈ 9.0 s):不加干预时,cap≥256 全部提前自然收尾,这也是需要 budget forcing 才能测长思考档位的原因;
- t_think 与分辨率无关(§7 自检):t2i 的 think 阶段没有任何图像输入,R 只在 gen_image 生效。

## 5. 图像分辨率 → token 长度

图像 token 数 = `(R/16)²`(VAE 8× 下采样 × latent patch 2×):

| R | image tokens | 每步成本 s(R) | 拟合 R²(t_image ~ N) |
|---|---|---|---|
| 1024 | 4096 | 0.536 s/step | 0.9995 |
| 768 | 2304 | 0.298 s/step | 0.9995 |
| 512 | 1024 | 0.176 s/step | 0.9989 |
| 256 | 256 | 0.118 s/step | 0.9997 |

对 4 个点拟合 `s(R) ~ tok`:

```
线性:  s = 0.0718 + 1.096e-4·tok              RSS = 1.32e-3
二次:  s = 0.1032 + 5.84e-5·tok + 1.15e-8·tok²  RSS = 2.01e-6
```

两个要点:

1. **每步固定开销 a ≈ 0.07~0.10 s/step**(两个模型外推到 tok→0 一致给出正截距)。这解释了"低分辨率不成比例地贵":R 从 1024 降到 256,token 数缩 16 倍,s 只缩 4.5 倍——tok=256 时一半以上的每步成本是与分辨率无关的固定开销(CFG 双前向的调度、跨卡通信、norm/投影等)。
2. **二次项使 RSS 下降 ~650 倍**,attention 的平方复杂度在 4096 token 处已可见;但仅 4 点、自由度 1,该结论定性看待(见局限)。

![s(R) vs image tokens](think_res_outputs/s_vs_tokens.png)

## 6. 图像推理步数 N

**定律:`t_image = s(R) · N`,对 N 严格线性,与 cap 无关。**

四档分辨率的线性拟合 R² 全部 ≥ 0.9989,截距 -0.10 ~ -0.22 s(近似过原点,微小负截距来自首步 warm 效应)。t_image 实测(N=50):26.97 s @1024 → 14.98 @768 → 8.65 @512 → 5.81 @256。

蒸馏是 image 段唯一的大杠杆:N 从 50 → 5,t_image 无论哪档分辨率都恰好缩 ~10 倍。

![t_image vs N per R](think_res_outputs/t_image_vs_N_per_R.png)

## 7. 计时自检:t_think 与 R 无关

按 cap 分组对 4 档 R 做 Kruskal-Wallis:

| cap | 32 | 64 | 128 | 256 | 1000 |
|---|---|---|---|---|---|
| p 值 | 0.672 | 0.323 | 0.880 | **0.026** | 0.169 |

cap=256 名义显著,但 4 组均值为 14.05~14.11 s(极差 0.4%),是大样本(每组 n=128)放大了 worker/卡间的微小系统差,且经 Bonferroni 校正(5 次检验,阈值 0.01)后不显著。结论:**t_think 与分辨率无关**成立,计时未被污染。

## 8. 耗时占比(核心结果)

两段独立线性 ⇒ 占比闭式解:

```
ratio = t_think / t_image ≈ 0.055·T / (s(R)·N)
平衡点 T*(N, R) = s(R)·N / 0.055        (T > T* ⇒ think 主导)
```

**T\* 速查表**(think token 数超过该值,think 段就压过 image 段):

| | N=50 | N=10 | N=5 |
|---|---|---|---|
| **R=1024** | 486 | 97 | 49 |
| **R=768** | 270 | 54 | 27 |
| **R=512** | 159 | 32 | 16 |
| **R=256** | 107 | 21 | 11 |

实测锚点(forcing 修正后):

- 慢配置 R=1024, N=50:T=1000 时 ratio=2.04(think 占 67%);自然思考 162 token 只占 25%,cap=32 时仅 6%——**未蒸馏 + 全分辨率下,image 是绝对瓶颈**;
- 蒸馏后 R=1024, N=5:T=1000 → think 占 **95.7%**;哪怕 T=256 也占 85%;
- 极端角 R=256, N=5, T=1000:ratio=112,think 占 **99.1%**——图像段已缩到 0.49 s,继续优化扩散毫无意义。

![ratio vs cap per R](think_res_outputs/ratio_vs_cap_per_R.png)

![fraction_think heatmap per R](think_res_outputs/fraction_think_heatmap_per_R.png)

## 9. 结论速查

三条定律:

1. `t_think ≈ 0.0552 · T`(与 N、R 无关;自然长度 ~162 token);
2. `t_image ≈ s(R) · N`,`s(R) ≈ 0.072 + 1.10e-4 · (R/16)²`(与 T 无关;含 ~0.07 s/step 固定开销);
3. `ratio ≈ 0.055·T / (s(R)·N)`,平衡点 `T* = s(R)·N / 0.055`。

要降延迟先动哪个旋钮:

| 当前 regime | 典型配置 | 先动什么 |
|---|---|---|
| image 主导(占比 >70%) | 未蒸馏 N=50,自然 think | **蒸馏降 N**(50→5 直接省 ~24 s @1024),其次降 R(注意质量代价) |
| 两段相当 | N=10 + 自然 think @1024 | 双管齐下;降 R 收益被固定开销折半 |
| think 主导(占比 >70%) | 蒸馏 N≤10 + 长 think | **压 T 或加速解码**(投机解码、单卡部署消除 PCIe 逐 token 开销);再动 N/R 只优化剩下不到 30% |

一句话:**蒸馏把扩散做便宜之后,BAGEL 的延迟问题就变成了 LLM 解码问题。**

---

*数据与图表:`experiments/outputs/think_res_outputs/`(aggregated.csv + 4 PNG);实验脚本 `run_res_sweep_mp.py`,分析 notebook `resolution_sweep_benchmark.ipynb`;budget forcing 实现见 `BUDGET_FORCING.md`。*
