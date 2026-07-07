#!/usr/bin/env python3
"""
非对称放置 vs 13/15 流水线放置 的 think 加速 benchmark。

用法:
  # ⚠ 必须在 import torch 之前设置 CUDA_VISIBLE_DEVICES=0,1 (单进程双卡)
  CUDA_VISIBLE_DEVICES=0,1 python experiments/run_asym_bench.py --placement asym
  CUDA_VISIBLE_DEVICES=0,1 python experiments/run_asym_bench.py --placement pipeline

设计:
  - 单进程, 双卡
  - --placement {asym,pipeline}:
      asym    → experiments.asym_placement.load_model_asym
      pipeline→ 复用 run_cap_sweep_mp.py:272-356 的 13/15 accelerate 加载 (对照组)
  - 复用 run_cap_sweep_mp.py:388-457 的 sync_timer / trial 结构
  - warm-up 1 次 (默认开启, --no-warmup 关闭)
  - 每配置 ≥3 trials (--trials N 可调)
  - 配置: R=1024, cap ∈ {256, 1000}, num_timesteps ∈ {50, 10}
  - 输出 CSV + 屏幕摘要 (think tok/s, t_image, 对比基线 0.055 s/token)
"""

import os
import sys
import time
import json
import random
import gc
import argparse
from contextlib import contextmanager
from copy import deepcopy

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
import pandas as pd

# ── 常量 ──
MODEL_PATH = os.path.join(_proj_root, "BAGEL-7B-MoT")
IMAGE_SHAPE = (1024, 1024)
OUTPUT_DIR = os.path.join(_proj_root, "experiments", "asym_bench_outputs")

DEFAULT_CAPS = [256, 1000]
DEFAULT_NUM_TIMESTEPS = [50, 10]
N_TRIALS_PER_CONFIG = 3
N_PROMPTS = 4
SEED_BASE = 1000
WARMUP_SEED = 999

# 基线 (来自 compass 报告 / PROFILE_ANALYSIS.md)
BASELINE_T_THINK_PER_TOKEN = 0.055  # s/token
TARGET_T_THINK_TOK_PER_SEC = 35     # ≥35 tok/s

BENCH_PROMPTS = [
    "a photo of a cute cat sitting on a windowsill",
    "an oil painting of a mountain landscape at sunset",
    "a futuristic cityscape with neon lights at night",
    "a still life of fruits on a wooden table",
]


def _load_model_pipeline(model_path, primary_device="cuda:0"):
    """复用 run_cap_sweep_mp.py:272-356 的多卡 accelerate 加载 (13/15 流水线对照组)。"""
    from accelerate import load_checkpoint_and_dispatch, init_empty_weights
    from data.data_utils import add_special_tokens
    from modeling.bagel import (
        BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
        SiglipVisionConfig, SiglipVisionModel,
    )
    from modeling.qwen2 import Qwen2Tokenizer
    from modeling.autoencoder import load_ae

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

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    num_layers = llm_config.num_hidden_layers  # 28
    split = num_layers // 2 - 1  # 13
    device_map = {}
    for layer_idx in range(num_layers):
        gpu = 0 if layer_idx < split else 1
        device_map[f"language_model.model.layers.{layer_idx}"] = gpu

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]
    for k in same_device_modules:
        device_map[k] = 0

    device_map["language_model.model.norm"] = 1
    device_map["language_model.model.norm_moe_gen"] = 1
    device_map["language_model.model.rotary_emb"] = 1
    device_map["language_model.lm_head"] = 1
    device_map["vit_model"] = 1

    print(f"[pipeline] device_map: GPU 0 = layers 0-{split - 1} + embed + aux(same-device), "
          f"GPU 1 = layers {split}-{num_layers - 1} + norm/head/vit")

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload_asym_bench",
    )
    model = model.eval()
    vae_model = vae_model.to(primary_device).eval()
    return model, vae_model, tokenizer, new_token_ids


# ── timing utilities ──

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_taylorseer_state(m):
    lm = m.language_model.model
    lm.enable_taylorseer = False
    for attr in ("cache_dic", "current"):
        if hasattr(lm, attr):
            delattr(lm, attr)
    for layer in lm.layers:
        layer.enable_taylorseer = False
        for attr in ("cache_dic", "current"):
            if hasattr(layer, attr):
                delattr(layer, attr)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


class _Elapsed:
    elapsed = None


@contextmanager
def sync_timer():
    torch.cuda.synchronize()
    result = _Elapsed()
    t0 = time.perf_counter()
    yield result
    torch.cuda.synchronize()
    result.elapsed = time.perf_counter() - t0


def run_trial(inferencer, tokenizer, prompt, cond, seed):
    """复用 run_cap_sweep_mp.py:400-470 trial 结构, 加 think_tok_per_sec 指标。"""
    reset_taylorseer_state(inferencer.model)
    set_all_seeds(seed)

    record = dict(prompt=prompt, seed=seed, **cond)
    gen_context = cfg_text_context = cfg_img_context = None
    try:
        gen_context = inferencer.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            with sync_timer() as t_prefill:
                from inferencer import GEN_THINK_SYSTEM_PROMPT
                gen_context = inferencer.update_context_text(GEN_THINK_SYSTEM_PROMPT, gen_context)
                cfg_img_context = inferencer.update_context_text(GEN_THINK_SYSTEM_PROMPT, cfg_img_context)
                cfg_text_context = deepcopy(gen_context)
                gen_context = inferencer.update_context_text(prompt, gen_context)
                cfg_img_context = inferencer.update_context_text(prompt, cfg_img_context)

            with sync_timer() as t_think:
                gen_text = inferencer.gen_text(
                    gen_context, do_sample=False, temperature=0.3,
                    max_length=cond["max_think_token_n"],
                    min_length=cond.get("min_think_token_n", 0),
                    wait_interjection=cond.get("wait_interjection"),
                )

            gen_context = inferencer.update_context_text(gen_text, gen_context)

            with sync_timer() as t_image:
                inferencer.gen_image(
                    IMAGE_SHAPE, gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cond["cfg_text_scale"],
                    cfg_img_scale=cond["cfg_img_scale"],
                    cfg_interval=cond["cfg_interval"],
                    cfg_renorm_min=cond["cfg_renorm_min"],
                    cfg_renorm_type=cond["cfg_renorm_type"],
                    timestep_shift=cond["timestep_shift"],
                    num_timesteps=cond["num_timesteps"],
                    enable_taylorseer=cond["enable_taylorseer"],
                )

        think_token_count = len(tokenizer(gen_text, add_special_tokens=False).input_ids)
        hit_cap = think_token_count >= cond["max_think_token_n"] - 2
        think_closed = gen_text.strip().endswith("</think>")
        tok_per_sec = think_token_count / t_think.elapsed if t_think.elapsed > 0 else None

        record.update(
            t_prefill=t_prefill.elapsed,
            t_think=t_think.elapsed,
            t_image=t_image.elapsed,
            think_token_count=think_token_count,
            think_tok_per_sec=tok_per_sec,
            hit_cap=hit_cap,
            think_closed=think_closed,
            gen_text=gen_text,
            ok=True, error=None,
        )
    except Exception as e:
        record.update(
            t_prefill=None, t_think=None, t_image=None,
            think_token_count=None, think_tok_per_sec=None,
            hit_cap=None, think_closed=None,
            gen_text=None, ok=False, error=repr(e),
        )
    finally:
        del gen_context, cfg_text_context, cfg_img_context
        reset_taylorseer_state(inferencer.model)
        gc.collect()
        torch.cuda.empty_cache()
    return record


def build_conditions(caps, num_timesteps):
    """预算强制 (s1 式): min = max = cap, 强制 think 精确到 cap。"""
    FORCE_THINK = True
    WAIT = " Wait,"
    return [
        dict(
            num_timesteps=n, max_think_token_n=cap,
            min_think_token_n=cap if FORCE_THINK else 0,
            wait_interjection=WAIT if FORCE_THINK else None,
            cfg_text_scale=1.0, cfg_img_scale=1.0,
            cfg_interval=[0.4, 1.0], cfg_renorm_min=0.0, cfg_renorm_type="global",
            timestep_shift=3.0, enable_taylorseer=False,
        )
        for n in num_timesteps
        for cap in caps
    ]


def main():
    parser = argparse.ArgumentParser(description="asym vs pipeline placement benchmark")
    parser.add_argument("--placement", choices=["asym", "pipeline"], default="asym")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--trials", type=int, default=N_TRIALS_PER_CONFIG)
    parser.add_argument("--prompts", type=int, default=N_PROMPTS)
    parser.add_argument("--caps", type=str, default=",".join(str(c) for c in DEFAULT_CAPS))
    parser.add_argument("--num-timesteps", type=str,
                        default=",".join(str(n) for n in DEFAULT_NUM_TIMESTEPS))
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    args = parser.parse_args()

    caps = [int(c) for c in args.caps.split(",")]
    num_ts = [int(n) for n in args.num_timesteps.split(",")]
    prompts = BENCH_PROMPTS[: args.prompts]

    print(f"[main] placement={args.placement} trials/cfg={args.trials} prompts={len(prompts)} "
          f"caps={caps} num_timesteps={num_ts}")
    print(f"[main] visible GPUs: {torch.cuda.device_count()} "
          f"({[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})")

    # ── 加载模型 ──
    if args.placement == "asym":
        from experiments.asym_placement import (
            load_model_asym, verify_placement, install_input_transfer_shim,
        )
        model, vae_model, tokenizer, new_token_ids = load_model_asym(
            model_path=args.model_path, und_device="cuda:0", gen_device="cuda:1",
        )
        verify_placement(model, und_device="cuda:0", gen_device="cuda:1",
                         vae_model=vae_model, strict=True)
        # P0 fix: prepare_* 返回 CPU tensor, asym 模式无 accelerate hook,
        # 必须显式搬到 cuda:0. 详见 asym_placement.install_input_transfer_shim.
        n = install_input_transfer_shim(model, und_device="cuda:0")
        assert n == 6, f"expected to shim 6 prepare_* methods, got {n}"
    else:
        model, vae_model, tokenizer, new_token_ids = _load_model_pipeline(args.model_path)

    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)
    inferencer = InterleaveInferencer(
        model=model, vae_model=vae_model, tokenizer=tokenizer,
        vae_transform=vae_transform, vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"{args.placement}_trials.csv")

    conditions = build_conditions(caps, num_ts)

    # ── warmup ──
    if args.warmup:
        warmup_cond = conditions[0]
        warmup_prompt = "a photo of a cat"
        print(f"[main] warmup: cap={warmup_cond['max_think_token_n']} "
              f"num_timesteps={warmup_cond['num_timesteps']}")
        w = run_trial(inferencer, tokenizer, warmup_prompt, warmup_cond, seed=WARMUP_SEED)
        if not w["ok"]:
            print(f"[main] WARMUP FAILED: {w['error']}")
            sys.exit(1)
        print(f"[main] warmup ok: t_think={w['t_think']:.2f}s "
              f"t_image={w['t_image']:.2f}s think_tok/s={w['think_tok_per_sec']:.1f}")

    # ── trials ──
    rows = []
    t_start = time.perf_counter()
    total = len(prompts) * len(conditions) * args.trials
    i = 0
    for pi, prompt in enumerate(prompts):
        for ci, cond in enumerate(conditions):
            for repeat in range(args.trials):
                i += 1
                seed = SEED_BASE + pi * 10000 + ci * 100 + repeat
                row = run_trial(inferencer, tokenizer, prompt, cond, seed)
                row["placement"] = args.placement
                row["prompt_idx"] = pi
                row["cond_idx"] = ci
                row["repeat"] = repeat
                row["trial_idx"] = i
                rows.append(row)

                status = "ok" if row["ok"] else f"FAIL({row['error'][:50]})"
                elapsed = time.perf_counter() - t_start
                eta = elapsed / i * (total - i) if i < total else 0
                print(f"[main] [{i}/{total}] cap={cond['max_think_token_n']} "
                      f"ts={cond['num_timesteps']} repeat={repeat}: {status} "
                      f"t_think={row.get('t_think')} t_image={row.get('t_image')} "
                      f"tok/s={row.get('think_tok_per_sec')} "
                      f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")

    # ── 写 CSV ──
    df = pd.DataFrame(rows)
    df["hardware"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    df["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    df.to_csv(out_csv, index=False)
    print(f"\n[main] wrote {len(df)} rows → {out_csv}")

    # ── 屏幕摘要 ──
    print("\n=== summary ===")
    ok = df[df["ok"]]
    if len(ok) > 0:
        grp = ok.groupby(["max_think_token_n", "num_timesteps"]).agg(
            t_think_mean=("t_think", "mean"),
            t_think_std=("t_think", "std"),
            tok_per_sec_mean=("think_tok_per_sec", "mean"),
            tok_per_sec_std=("think_tok_per_sec", "std"),
            t_image_mean=("t_image", "mean"),
            t_image_std=("t_image", "std"),
            n=("t_think", "count"),
        )
        print(grp.to_string())
        print(f"\nbaseline t_think ≈ {BASELINE_T_THINK_PER_TOKEN} s/token "
              f"({1 / BASELINE_T_THINK_PER_TOKEN:.1f} tok/s)")
        overall_mean_tps = ok["think_tok_per_sec"].mean()
        print(f"this run mean think tok/s: {overall_mean_tps:.1f}")
        if args.placement == "asym":
            if overall_mean_tps >= TARGET_T_THINK_TOK_PER_SEC:
                print(f"✅ ≥{TARGET_T_THINK_TOK_PER_SEC} tok/s target reached")
            else:
                print(f"❌ below {TARGET_T_THINK_TOK_PER_SEC} tok/s target "
                      f"(got {overall_mean_tps:.1f})")
    else:
        print("no successful trials to summarize.")


if __name__ == "__main__":
    main()