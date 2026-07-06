# 阶段0：非对称双卡放置（und 专家单卡驻留），加速 t_think

## Context

调研文档（compass 报告）结论：当前 t_think ≈ 0.055 s/token（18.2 tok/s）的根因是 13/15 层间流水线切分导致逐 token 跨 PCIe 传输（2×RTX 4090 24GB，无 P2P/NVLink）。think 阶段（`mode="und"`）在代码上已确认**只触碰无后缀的理解专家权重 + embed_tokens + norm + lm_head**（`bagel.py:967-984`、`qwen2_navit.py:781-820`），VAE/ViT 完全不参与。理解专家 ≈15.2GB bf16，可整体放进单张 24GB 4090。

**目标**：think 阶段零跨卡通信，t_think 提升到 ≥35 tok/s（≤0.029 s/token），免训练，只改权重放置 + 少量 gen 分支设备搬运补丁。

## 放置方案

| 设备 | 内容 | 显存估算 |
|------|------|----------|
| GPU 0 | und 专家全部 28 层（无后缀 q/k/v/o_proj、q/k_norm、mlp、两个 layernorm）+ embed_tokens + lm_head + norm + rotary_emb + 辅助模块（time_embedder/latent_pos_embed/vae2llm/llm2vae/connector/vit_pos_embed）+ vit_model + vae_model | ~16.5GB，余 ~7GB 给 KV cache/激活/VAE decode |
| GPU 1 | **仅** 所有 `*_moe_gen` 子模块（每层 q/k/v/o_proj_moe_gen、q/k_norm_moe_gen、mlp_moe_gen、input/post_attention_layernorm_moe_gen + 顶层 norm_moe_gen），≈13.1GB | ~13.1GB，余量大 |

设计要点：
- **gen 模式下 hidden states 仍留 GPU 0**（与现在一致），只有 `_moe_gen` 权重在 GPU 1；每层把 VAE token 切片搬去 GPU 1 计算再搬回。think 路径完全不变、纯单卡。
- 辅助模块 + connector + vit_model 全留 GPU 0 → 图像理解/编辑路径零改动，避免 DEBUG_NOTES §8.9 跨设备冲突复发。
- **不用 accelerate**（hook 的逐子模块 Python 开销正是要消除的 CPU 开销之一），复用 `run_cap_sweep_mp.py:263-271` 的手动加载模式（CPU load_file → load_state_dict → 按规则逐模块 `.to()`）。

## 实现步骤

### 1. 新建 `experiments/asym_placement.py`
- `load_model_asym(model_path, und_device="cuda:0", gen_device="cuda:1")`：
  - 照抄 `run_cap_sweep_mp.py:230-261` 的模型构建（CPU 实例化，不用 init_empty_weights）+ `load_file(ema.safetensors, device="cpu")` + `load_state_dict(strict=False)`。
  - cast bf16 后按规则逐子模块搬运：`named_modules()` 中名字含 `_moe_gen` 的叶子模块 → gen_device，其余 → und_device（从 CPU 分别搬，避免整模型先上 GPU0 OOM）。vae_model 单独 `.to(und_device)`。
  - `verify_placement(model)`：断言所有无后缀 LLM 参数 + embed/lm_head/norm 在 und_device、所有 `_moe_gen` 参数在 gen_device，打印两卡参数量/字节数。

### 2. 补丁 `modeling/bagel/qwen2_navit.py`（仅 forward_inference 的 gen 分支，und 分支零改动）
写法统一为：`gen_dev = self.<some>_moe_gen.weight.device`，把 VAE token 切片 `.to(gen_dev)` 一次、在 GPU 1 上连续算完、结果 `.to(main_dev)` 一次搬回（同卡时 `.to()` 是 no-op，单卡/流水线模式行为不变）：
- `PackedAttentionMoT.forward_inference` gen 分支（`qwen2_navit.py:520-548`）：`packed_vae_query_sequence` 搬到 gen_dev → q/k/v_proj_moe_gen + q/k_norm_moe_gen 全在 GPU 1 算 → 结果搬回 scatter；`o_proj_moe_gen`（attention 输出后同样处理，需看该函数后半段 ~560-680 行的 o_proj scatter 位置）。
- `Qwen2MoTDecoderLayer.forward_inference` gen 分支：`input_layernorm_moe_gen`（786 行）、`post_attention_layernorm_moe_gen + mlp_moe_gen`（811-820 行，把 813-819 的 vae 切片整段搬 GPU 1 算完搬回）。
- `Qwen2Model.forward_inference`（1078-1082 行）：`norm_moe_gen` 同模式。
- 编辑前按项目 CLAUDE.md 要求跑 `gitnexus_impact` 报告 blast radius；提交前跑 `gitnexus_detect_changes()`。

### 3. 新建 `experiments/run_asym_bench.py`
- 单进程、`CUDA_VISIBLE_DEVICES=0,1`，参数 `--placement {asym,pipeline}`（pipeline 分支复用现 13/15 accelerate 加载代码，作对照组）。
- 复用 `run_cap_sweep_mp.py:405-457` 的 trial 结构：`sync_timer` 计 t_prefill / t_think / t_image，s/token = t_think/token 数；warmup 1 次 + 每配置 ≥3 trials；配置建议 R=1024、cap∈{256,1000}、固定 seed 同一组 prompt。
- 输出 CSV + 屏幕摘要（think tok/s、t_image、对比基线 0.055）。

## 验证

1. `verify_placement` 断言通过（und 参数全在 GPU 0）。
2. **think 正确性**：同 prompt 贪心解码，asym 输出与单卡/基线输出对比（bf16 数值路径变化可能有细微 diff，检查文本合理性与前若干 token 一致性）。
3. **think 性能**：≥35 tok/s（doc 预期 40-50）。跑 think 时用 `nvidia-smi` 观察 GPU 1 利用率应 ≈0（证明零跨卡）。
4. **图像生成冒烟**：1024² 生成 1 张图不崩、目测正常；记录 t_image 回归幅度（已知代价：去噪时 VAE token 激活每层往返 PCIe，1024²/4096 tokens 每次前向 ~3GB 传输，t_image 预计明显变慢——阶段0 接受，数字实测记录）。
5. 结果写入 `EXPERIMENT_SUMMARY.md` / `DEBUG_NOTES.md` 新章节。

## 风险与回退

- **gen 分支补丁遗漏 scatter 点** → 运行时 device mismatch 错误，冒烟测试会立刻暴露，逐点补 `.to()`。
- **t_image 回归过大**：备选后续优化（不在阶段0）——去噪时把 hidden states 主体放 GPU 1、只把少量文本 token 切片搬 GPU 0，并把 KV cache 一次性搬 GPU 1（每层 KV 仅 ~MB 级）。
- **回退**：`--placement pipeline` 保留原路径；`qwen2_navit.py` 补丁在同卡时为 no-op，不影响现有脚本。
- 图像理解（ViT）路径不在 benchmark 内，本方案已让其全在 GPU 0，理论无影响，不专门测试。
