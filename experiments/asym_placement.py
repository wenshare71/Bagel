#!/usr/bin/env python3
"""
非对称双卡放置：und 专家全驻单卡，gen 专家 (_moe_gen) 全驻另一卡。

设计:
  - GPU 0 (und_device) ≈16.5GB：所有 28 层无后缀 q/k/v/o_proj + q/k_norm + mlp +
    input/post_attention_layernorm + embed_tokens + lm_head + norm + rotary_emb +
    time_embedder / latent_pos_embed / vae2llm / llm2vae / connector / vit_pos_embed +
    vit_model + vae_model。
  - GPU 1 (gen_device) ≈13.1GB：仅所有 *_moe_gen 子模块。
  - 不用 accelerate：避免子模块 hook 的 Python 开销。
  - 逐 leaf 模块搬运：先在 CPU 完整实例化 + load_state_dict，cast bf16 后按规则
    named_modules() 走一遍，每个含直接参数/缓冲的最小单元 .to(target)。
    避免整模型先上 GPU 0 OOM。

用法:
  from experiments.asym_placement import load_model_asym, verify_placement
  model, vae_model, tokenizer, new_token_ids = load_model_asym(MODEL_PATH)
  verify_placement(model, und_device="cuda:0", gen_device="cuda:1", vae_model=vae_model)
"""

import os
import sys
import gc
from typing import Tuple

import torch
from safetensors.torch import load_file

_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from data.data_utils import add_special_tokens
from data.transforms import ImageTransform  # noqa: F401  (供 run_asym_bench.py 复用)
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae

# 默认路径（与 run_cap_sweep_mp.py 保持一致）
DEFAULT_MODEL_PATH = os.path.join(_proj_root, "BAGEL-7B-MoT")

# Bagel 顶层非 LLM 子模块名（无 _moe_gen，全部驻 und_device）
_AUX_TOP_MODULES = (
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "latent_pos_embed",
    "vit_pos_embed",
    "connector",
)


def _build_cpu_model(model_path: str):
    """CPU 实例化 BAGEL + 加载 ema.safetensors + cast bf16。复用 run_cap_sweep_mp.py:230-271 单卡模式。"""
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True, visual_und=True,
        llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2, max_latent_size=64,
    )
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # CPU 加载权重 + strict=False (moe_gen 共享基座权重，会出现 unexpected / missing)
    sd = load_file(os.path.join(model_path, "ema.safetensors"), device="cpu")
    msg = model.load_state_dict(sd, strict=False)
    del sd
    gc.collect()

    # cast bf16 on CPU
    model = model.to(dtype=torch.bfloat16)

    return model, vae_model, tokenizer, new_token_ids


def _partition_to_devices(
    model: Bagel,
    und_device: torch.device,
    gen_device: torch.device,
) -> None:
    """逐 leaf 把 model 的参数 / 缓冲搬到对应卡。

    规则: 模块全名含 "_moe_gen" → gen_device; 否则 → und_device。
    叶 = 当前模块有直接 parameters(recurse=False) 或 buffers(recurse=False)
    且不为 root (model 本身)。容器（如 mlp_moe_gen / Qwen2MoTDecoderLayer）
    若无直接参数/缓冲则跳过，其叶子会被递归到时单独搬。
    """
    for name, m in model.named_modules():
        if name == "":
            continue
        target = gen_device if "_moe_gen" in name else und_device
        has_direct_params = bool(list(m.parameters(recurse=False)))
        has_direct_buffers = bool(list(m.buffers(recurse=False)))
        if has_direct_params or has_direct_buffers:
            m.to(target)


def load_model_asym(
    model_path: str = DEFAULT_MODEL_PATH,
    und_device: str = "cuda:0",
    gen_device: str = "cuda:1",
) -> Tuple[Bagel, object, Qwen2Tokenizer, dict]:
    """非对称放置加载 BAGEL-7B-MoT。

    Args:
        model_path: 包含 llm_config.json / ema.safetensors / ae.safetensors 的目录。
        und_device: 理解专家 / 视觉 / VAE / 辅助模块所在设备。
        gen_device: 所有 *_moe_gen 子模块所在设备。

    Returns:
        (model, vae_model, tokenizer, new_token_ids)
        - model: Bagel, .eval() 状态。
        - vae_model: AE, 已 .to(und_device).eval()。
        - tokenizer: Qwen2Tokenizer + 特殊 token。
        - new_token_ids: add_special_tokens 返回的 dict (与 run_cap_sweep_mp.py 兼容)。
    """
    und_dev = torch.device(und_device)
    gen_dev = torch.device(gen_device)

    print(f"[asym] building CPU model from {model_path} ...")
    model, vae_model, tokenizer, new_token_ids = _build_cpu_model(model_path)

    # vit_model 整体搬到 und_device（不含 _moe_gen）
    model.vit_model = model.vit_model.to(und_dev).eval()
    print(f"[asym] vit_model → {und_device}")

    # Bagel 顶层辅助模块：time_embedder / vae2llm / llm2vae / latent_pos_embed /
    # vit_pos_embed / connector 全部驻 und_device
    for attr in _AUX_TOP_MODULES:
        m = getattr(model, attr, None)
        if m is not None:
            m.to(und_dev)
    print(f"[asym] aux modules {list(_AUX_TOP_MODULES)} → {und_device}")

    # language_model 内部逐 leaf 切分
    _partition_to_devices(model, und_dev, gen_dev)
    print(f"[asym] language_model partitioned: _moe_gen → {gen_device}, rest → {und_device}")

    # vae_model 整体到 und_device
    vae_model = vae_model.to(und_dev).eval()
    print(f"[asym] vae_model → {und_device}")

    # 顶层置 eval
    model.eval()
    return model, vae_model, tokenizer, new_token_ids


def verify_placement(
    model: Bagel,
    und_device: str = "cuda:0",
    gen_device: str = "cuda:1",
    vae_model=None,
    strict: bool = True,
) -> dict:
    """断言 model 与 vae_model 的所有参数 / 缓冲落在期望设备上。

    Args:
        model: 已 load_model_asym 加载的 Bagel。
        und_device / gen_device: 与 load_model_asym 保持一致。
        vae_model: 可选, 验证其全部参数都在 und_device。
        strict: True 时发现错放直接抛 AssertionError；False 仅打印告警。

    Returns:
        dict 包含 per_device 的 params/bytes 统计 (含 vae_model 时合并到 und)。
    """
    und_dev = torch.device(und_device)
    gen_dev = torch.device(gen_device)

    per_dev_params = {und_device: 0, gen_device: 0}
    per_dev_bytes = {und_device: 0, gen_device: 0}
    per_dev_tensors = {und_device: 0, gen_device: 0}
    errors = []

    # model: 参数
    for name, p in model.named_parameters():
        target = gen_dev if "_moe_gen" in name else und_dev
        if p.device != target:
            msg = f"PARAM {name} on {p.device}, expected {target}"
            errors.append(msg)
        else:
            per_dev_tensors[p.device.type + ":" + str(p.device.index)] = (
                per_dev_tensors.get(p.device.type + ":" + str(p.device.index), 0) + 1
            )
            per_dev_params[str(p.device)] += p.numel()
            per_dev_bytes[str(p.device)] += p.numel() * p.element_size()

    # model: 缓冲 (rotary_emb.inv_freq 等)
    for name, b in model.named_buffers():
        target = gen_dev if "_moe_gen" in name else und_dev
        if b.device != target:
            msg = f"BUFFER {name} on {b.device}, expected {target}"
            errors.append(msg)
        else:
            per_dev_params[str(b.device)] += b.numel()
            per_dev_bytes[str(b.device)] += b.numel() * b.element_size()

    # vae_model: 全部应在 und_device
    if vae_model is not None:
        for name, p in vae_model.named_parameters():
            if p.device != und_dev:
                errors.append(f"VAE PARAM {name} on {p.device}, expected {und_dev}")
            else:
                per_dev_params[str(p.device)] += p.numel()
                per_dev_bytes[str(p.device)] += p.numel() * p.element_size()
        for name, b in vae_model.named_buffers():
            if b.device != und_dev:
                errors.append(f"VAE BUFFER {name} on {b.device}, expected {und_dev}")
            else:
                per_dev_params[str(b.device)] += b.numel()
                per_dev_bytes[str(b.device)] += b.numel() * b.element_size()

    # 报告
    print(f"[verify] und_device ({und_device}): "
          f"{per_dev_params[und_device] / 1e9:.3f}B params, "
          f"{per_dev_bytes[und_device] / 1024 ** 3:.2f} GiB")
    print(f"[verify] gen_device ({gen_device}): "
          f"{per_dev_params[gen_device] / 1e9:.3f}B params, "
          f"{per_dev_bytes[gen_device] / 1024 ** 3:.2f} GiB")

    if errors:
        print(f"[verify] ❌ {len(errors)} placement violations:")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        if strict:
            raise AssertionError(f"placement verification failed: {len(errors)} violations")
    else:
        print(f"[verify] ✅ all params/buffers on expected devices")

    return dict(
        params=per_dev_params,
        bytes=per_dev_bytes,
        n_violations=len(errors),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Asymmetric dual-GPU placement loader")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--und-device", default="cuda:0")
    parser.add_argument("--gen-device", default="cuda:1")
    parser.add_argument("--no-strict", action="store_true", help="don't raise on violations")
    args = parser.parse_args()

    model, vae_model, tokenizer, new_token_ids = load_model_asym(
        model_path=args.model_path,
        und_device=args.und_device,
        gen_device=args.gen_device,
    )
    verify_placement(
        model,
        und_device=args.und_device,
        gen_device=args.gen_device,
        vae_model=vae_model,
        strict=not args.no_strict,
    )
    print("[main] done.")