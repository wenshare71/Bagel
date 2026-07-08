#!/usr/bin/env python3
"""KV cache 长度受控扫描 benchmark：验证 decode 速度(t_think/token)是否受 KV 长度影响。

动机:
  EDIT_THINK_RATIO.md 的证据是观察性的 —— women.jpg(kv_len=7121) 和
  octupusy.jpg(kv_len=3254) 除了 KV 长度不同，图像内容/输出分辨率/prompt
  也全都不同，KV 长度这一个变量没有被单独隔离出来。本脚本在完全固定
  图像/输出分辨率/think cap/去噪步数 N 的前提下，只改变 KV 长度，测出
  think 解码速度、prefill 时间、显存占用相对 KV 长度的真实曲线。

  进一步调研发现：编辑+think 场景下 KV 里 98~99% 是图片 token(VAE+ViT)，
  文本(system prompt + 一条编辑指令)只占 1~2%。业务多轮对话场景里 KV
  暴涨靠的是新增参考图，不是新增文本。因此本脚本提供两种互补的扫描模式:

    --mode multi_image (主实验, 贴近业务): 固定最终编辑图+prompt+cap+N,
        在其之前累积注入 n_prior_turns 轮"参考图+短指令"，KV 长度随
        真实图片 token 线性累积。覆盖并超过 compass 报告 69e62e82 §5
        给出的"重新评估"触发阈值(≥3图/图像token>8k、累积KV>20k)。

    --mode text_filler (辅助实验, 隔离 token 类型): 固定同一张编辑图+
        prompt，在编辑文本之后注入可变长度的纯文本填充，单独扫 KV 长度。
        用于验证"KV 长度本身"(而非"图片 token 特殊")才是不影响 decode
        的原因 —— 如果两组实验结论一致，才能支持"通用 KV 压缩方法在
        此场景无收益"这个更强论断(LOOK-M/SnapKV 等方法本就不区分 token
        类型，只按 KV 位置/attention score 操作)。

用法:
  # ⚠ 必须在 import torch 之前设置 CUDA_VISIBLE_DEVICES=0,1 (单进程双卡)
  CUDA_VISIBLE_DEVICES=0,1 python experiments/scripts/run_kv_length_sweep.py --mode multi_image
  CUDA_VISIBLE_DEVICES=0,1 python experiments/scripts/run_kv_length_sweep.py --mode text_filler

设计:
  - 只用 pipeline (13/15 accelerate) 放置，与 EDIT_THINK_RATIO.md 口径一致。
  - CFG=bench (cfg_text_scale=cfg_img_scale=1.0)，跳过 CFG 分支单次前向，
    避免额外 KV 拷贝干扰计时信号。
  - cap=1000 (预算强制 min=max=cap)，N=10，与既有实验同口径。
  - 每个扫描点重复 --repeats 次(默认3)，不同 seed，算均值±标准差。
  - 每张 GPU 记录峰值显存(reset_peak_memory_stats + max_memory_allocated)。

计时口径:
  multi_image: t_prior_context(前置N轮图文) + t_final_prefill(最终图+文本)
               + t_think(gen_text) + t_image(gen_image)
  text_filler: t_prefill(system prompt+图+编辑文本+填充文本) + t_think + t_image
"""

import os
import sys
import gc
import time
import argparse
from copy import deepcopy

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
import pandas as pd
from PIL import Image

from experiments.scripts.run_asym_bench import (
    _load_model_pipeline, sync_timer, set_all_seeds, reset_taylorseer_state,
    MODEL_PATH, SEED_BASE, WARMUP_SEED,
)
from data.data_utils import pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer, GEN_THINK_SYSTEM_PROMPT

OUTPUT_DIR = os.path.join(_proj_root, "experiments", "outputs", "kv_length_sweep_outputs")
MEM_DEVICES = [torch.device("cuda:0"), torch.device("cuda:1")]

CFG_BENCH = dict(cfg_text_scale=1.0, cfg_img_scale=1.0)

# ── 实验 A: 多图/多轮累积 ──
PRIOR_IMAGE_POOL = ["test_images/women.jpg", "test_images/octupusy.jpg", "test_images/meme.jpg"]
PRIOR_TURN_TEXT = "For reference, please keep this in mind."
FINAL_IMAGE_PATH = "test_images/women.jpg"
FINAL_PROMPT = "She boards a modern subway, quietly reading a folded newspaper, wearing the same clothes."
DEFAULT_N_PRIOR_TURNS = [0, 1, 2, 3, 4, 5]

# ── 实验 B: 纯文本填充 ──
FILLER_SENTENCE = "The quick brown fox jumps over the lazy dog. "
TEXT_FILLER_TRIALS = [
    ("test_images/women.jpg",
     "She boards a modern subway, quietly reading a folded newspaper, wearing the same clothes."),
    ("test_images/octupusy.jpg",
     "Could you display the sculpture that takes after this design?"),
]
DEFAULT_FILLERS = [0, 1000, 4000, 8000, 16000, 24000]


def build_filler_text(tokenizer, target_tokens):
    """重复中性填充句, 截断到 target_tokens-2 (update_context_text 自动加 bos/eos)。"""
    goal = max(target_tokens - 2, 0)
    if goal == 0:
        return ""
    unit_ids = tokenizer.encode(FILLER_SENTENCE, add_special_tokens=False)
    reps = goal // len(unit_ids) + 1
    ids = tokenizer.encode(FILLER_SENTENCE * reps, add_special_tokens=False)[:goal]
    return tokenizer.decode(ids)


def build_cond(cap, num_timesteps):
    return dict(
        num_timesteps=num_timesteps,
        max_think_token_n=cap,
        min_think_token_n=cap,
        wait_interjection=" Wait,",
        cfg_interval=[0.4, 1.0], cfg_renorm_min=0.0, cfg_renorm_type="global",
        timestep_shift=3.0, enable_taylorseer=False,
        **CFG_BENCH,
    )


def _load_image(path):
    return Image.open(os.path.join(_proj_root, path))


def _reset_peak_memory():
    for d in MEM_DEVICES:
        torch.cuda.reset_peak_memory_stats(d)


def _read_peak_memory():
    return {f"mem_alloc_gpu{d.index}": torch.cuda.max_memory_allocated(d) for d in MEM_DEVICES}


def _gen_image_call(inferencer, image_shape, gen_context, cfg_text_context, cfg_img_context, cond):
    inferencer.gen_image(
        image_shape, gen_context,
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


# ══════════════════════════ 实验 A: 多图/多轮累积 ══════════════════════════

def run_multi_image_trial(inferencer, tokenizer, n_prior_turns, final_image_path, final_prompt, cond, seed):
    """在最终 edit-with-think trial 之前, 先累积注入 n_prior_turns 轮参考图+短指令。"""
    reset_taylorseer_state(inferencer.model)
    set_all_seeds(seed)
    _reset_peak_memory()

    record = dict(n_prior_turns=n_prior_turns, final_image=os.path.basename(final_image_path),
                  final_prompt=final_prompt, seed=seed, **cond)
    gen_context = cfg_text_context = cfg_img_context = None
    try:
        gen_context = inferencer.init_gen_context()
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            with sync_timer() as t_prior_context:
                gen_context = inferencer.update_context_text(GEN_THINK_SYSTEM_PROMPT, gen_context)
                cfg_img_context = inferencer.update_context_text(GEN_THINK_SYSTEM_PROMPT, cfg_img_context)

                for i in range(n_prior_turns):
                    prior_path = PRIOR_IMAGE_POOL[i % len(PRIOR_IMAGE_POOL)]
                    prior_image = inferencer.vae_transform.resize_transform(pil_img2rgb(_load_image(prior_path)))
                    gen_context = inferencer.update_context_image(prior_image, gen_context, vae=True, vit=True)
                    gen_context = inferencer.update_context_text(PRIOR_TURN_TEXT, gen_context)
                    cfg_img_context = inferencer.update_context_text(PRIOR_TURN_TEXT, cfg_img_context)

            with sync_timer() as t_final_prefill:
                final_image = inferencer.vae_transform.resize_transform(pil_img2rgb(_load_image(final_image_path)))
                gen_context = inferencer.update_context_image(final_image, gen_context, vae=True, vit=True)
                image_shape = final_image.size[::-1]
                cfg_text_context = deepcopy(gen_context)

                gen_context = inferencer.update_context_text(final_prompt, gen_context)
                cfg_img_context = inferencer.update_context_text(final_prompt, cfg_img_context)

            kv_len_prefill = gen_context["kv_lens"][0]

            with sync_timer() as t_think:
                gen_text = inferencer.gen_text(
                    gen_context, do_sample=False, temperature=0.3,
                    max_length=cond["max_think_token_n"],
                    min_length=cond.get("min_think_token_n", 0),
                    wait_interjection=cond.get("wait_interjection"),
                )

            gen_context = inferencer.update_context_text(gen_text, gen_context)

            with sync_timer() as t_image:
                _gen_image_call(inferencer, image_shape, gen_context, cfg_text_context, cfg_img_context, cond)

        think_token_count = len(tokenizer(gen_text, add_special_tokens=False).input_ids)
        record.update(
            t_prior_context=t_prior_context.elapsed,
            t_final_prefill=t_final_prefill.elapsed,
            t_think=t_think.elapsed, t_image=t_image.elapsed,
            think_token_count=think_token_count,
            think_tok_per_sec=think_token_count / t_think.elapsed,
            kv_len_prefill=kv_len_prefill,
            image_shape=f"{image_shape[0]}x{image_shape[1]}",
            gen_text=gen_text, ok=True, error=None,
            **_read_peak_memory(),
        )
    except Exception as e:
        record.update(
            t_prior_context=None, t_final_prefill=None, t_think=None, t_image=None,
            think_token_count=None, think_tok_per_sec=None, kv_len_prefill=None,
            image_shape=None, gen_text=None, ok=False, error=repr(e),
            mem_alloc_gpu0=None, mem_alloc_gpu1=None,
        )
    finally:
        del gen_context, cfg_text_context, cfg_img_context
        reset_taylorseer_state(inferencer.model)
        gc.collect()
        torch.cuda.empty_cache()
    return record


# ══════════════════════════ 实验 B: 纯文本填充 ══════════════════════════

def run_text_filler_trial(inferencer, tokenizer, image_path, prompt, filler_token_n, cond, seed):
    """在编辑文本之后注入可变长度纯文本填充, 单独扫 KV 长度。"""
    reset_taylorseer_state(inferencer.model)
    set_all_seeds(seed)
    _reset_peak_memory()

    record = dict(image=os.path.basename(image_path), prompt=prompt,
                  filler_token_n=filler_token_n, seed=seed, **cond)
    gen_context = cfg_text_context = cfg_img_context = None
    try:
        image = _load_image(image_path)
        gen_context = inferencer.init_gen_context()
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            with sync_timer() as t_prefill:
                gen_context = inferencer.update_context_text(GEN_THINK_SYSTEM_PROMPT, gen_context)
                cfg_img_context = inferencer.update_context_text(GEN_THINK_SYSTEM_PROMPT, cfg_img_context)

                image = inferencer.vae_transform.resize_transform(pil_img2rgb(image))
                gen_context = inferencer.update_context_image(image, gen_context, vae=True, vit=True)
                image_shape = image.size[::-1]
                cfg_text_context = deepcopy(gen_context)

                gen_context = inferencer.update_context_text(prompt, gen_context)
                cfg_img_context = inferencer.update_context_text(prompt, cfg_img_context)

                if filler_token_n > 0:
                    filler_text = build_filler_text(tokenizer, filler_token_n)
                    gen_context = inferencer.update_context_text(filler_text, gen_context)
                    cfg_img_context = inferencer.update_context_text(filler_text, cfg_img_context)

            kv_len_prefill = gen_context["kv_lens"][0]

            with sync_timer() as t_think:
                gen_text = inferencer.gen_text(
                    gen_context, do_sample=False, temperature=0.3,
                    max_length=cond["max_think_token_n"],
                    min_length=cond.get("min_think_token_n", 0),
                    wait_interjection=cond.get("wait_interjection"),
                )

            gen_context = inferencer.update_context_text(gen_text, gen_context)

            with sync_timer() as t_image:
                _gen_image_call(inferencer, image_shape, gen_context, cfg_text_context, cfg_img_context, cond)

        think_token_count = len(tokenizer(gen_text, add_special_tokens=False).input_ids)
        record.update(
            t_prefill=t_prefill.elapsed, t_think=t_think.elapsed, t_image=t_image.elapsed,
            think_token_count=think_token_count,
            think_tok_per_sec=think_token_count / t_think.elapsed,
            kv_len_prefill=kv_len_prefill,
            image_shape=f"{image_shape[0]}x{image_shape[1]}",
            gen_text=gen_text, ok=True, error=None,
            **_read_peak_memory(),
        )
    except Exception as e:
        record.update(
            t_prefill=None, t_think=None, t_image=None,
            think_token_count=None, think_tok_per_sec=None, kv_len_prefill=None,
            image_shape=None, gen_text=None, ok=False, error=repr(e),
            mem_alloc_gpu0=None, mem_alloc_gpu1=None,
        )
    finally:
        del gen_context, cfg_text_context, cfg_img_context
        reset_taylorseer_state(inferencer.model)
        gc.collect()
        torch.cuda.empty_cache()
    return record


def _linfit_summary(df, x_col, y_col):
    sub = df.dropna(subset=[x_col, y_col])
    if len(sub) < 2:
        return None
    x = sub[x_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


def _print_linfit(df):
    fit = _linfit_summary(df, "kv_len_prefill", "think_tok_per_sec")
    if fit:
        slope, intercept, r2 = fit
        span = df["kv_len_prefill"].max() - df["kv_len_prefill"].min()
        mean_tps = df["think_tok_per_sec"].mean()
        rel_swing = abs(slope) * span / mean_tps * 100 if mean_tps else float("nan")
        print(f"\nthink_tok_per_sec ~ kv_len_prefill: slope={slope:.6f} intercept={intercept:.3f} R²={r2:.3f}")
        print(f"|slope|×span / mean = {rel_swing:.1f}% "
              f"({'无实质敏感性' if rel_swing < 5 else '有可见敏感性, 需在文档中如实报告'})")


def main():
    parser = argparse.ArgumentParser(description="KV 长度受控扫描: 多图累积 vs 纯文本填充")
    parser.add_argument("--mode", choices=["multi_image", "text_filler"], required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--n-prior-turns", type=str, default=",".join(str(n) for n in DEFAULT_N_PRIOR_TURNS))
    parser.add_argument("--fillers", type=str, default=",".join(str(n) for n in DEFAULT_FILLERS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cap", type=int, default=1000)
    parser.add_argument("--num-timesteps", type=int, default=10)
    parser.add_argument("--cfg", choices=["bench"], default="bench")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    args = parser.parse_args()

    cond = build_cond(args.cap, args.num_timesteps)
    print(f"[main] mode={args.mode} cap={args.cap} N={args.num_timesteps} repeats={args.repeats}")
    print(f"[main] visible GPUs: {torch.cuda.device_count()} "
          f"({[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})")

    model, vae_model, tokenizer, new_token_ids = _load_model_pipeline(args.model_path)
    inferencer = InterleaveInferencer(
        model=model, vae_model=vae_model, tokenizer=tokenizer,
        vae_transform=ImageTransform(1024, 512, 16),
        vit_transform=ImageTransform(980, 224, 14),
        new_token_ids=new_token_ids,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "multi_image":
        sweep_points = [int(n) for n in args.n_prior_turns.split(",")]

        if args.warmup:
            print("[main] warmup: n_prior_turns=0, cap=64 ...")
            w_cond = dict(cond, max_think_token_n=64, min_think_token_n=64)
            w = run_multi_image_trial(inferencer, tokenizer, 0, FINAL_IMAGE_PATH, FINAL_PROMPT,
                                      w_cond, seed=WARMUP_SEED)
            assert w["ok"], f"warmup FAILED: {w['error']}"
            print(f"[main] warmup ok: t_think={w['t_think']:.2f}s t_image={w['t_image']:.2f}s "
                  f"kv_len={w['kv_len_prefill']}")

        rows, t0 = [], time.perf_counter()
        total = len(sweep_points) * args.repeats
        i = 0
        for ni, n_prior in enumerate(sweep_points):
            for repeat in range(args.repeats):
                i += 1
                seed = SEED_BASE + ni * 100 + repeat
                row = run_multi_image_trial(inferencer, tokenizer, n_prior, FINAL_IMAGE_PATH,
                                            FINAL_PROMPT, cond, seed)
                row["trial_idx"] = i
                rows.append(row)
                st = "ok" if row["ok"] else f"FAIL({str(row['error'])[:80]})"
                elapsed = time.perf_counter() - t0
                print(f"[{i}/{total}] n_prior_turns={n_prior} repeat={repeat}: {st} "
                      f"kv_len={row.get('kv_len_prefill')} t_think={row.get('t_think')} "
                      f"t_image={row.get('t_image')} tok/s={row.get('think_tok_per_sec')} "
                      f"elapsed={elapsed:.0f}s")

        df = pd.DataFrame(rows)
        df["hardware"] = torch.cuda.get_device_name(0)
        df["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        out_csv = os.path.join(args.output_dir, "multi_image_sweep_trials.csv")
        df.to_csv(out_csv, index=False)
        print(f"\nwrote {len(df)} rows → {out_csv}")

        ok = df[df["ok"] == True]
        if len(ok):
            print("\n=== summary (multi_image) ===")
            grp = ok.groupby("n_prior_turns").agg(
                kv_len_mean=("kv_len_prefill", "mean"),
                t_think_mean=("t_think", "mean"),
                tok_per_sec_mean=("think_tok_per_sec", "mean"),
                t_image_mean=("t_image", "mean"),
                n=("t_think", "count"),
            )
            print(grp.to_string())
            _print_linfit(ok)
        else:
            print("no successful trials to summarize.")

    else:  # text_filler
        fillers = [int(f) for f in args.fillers.split(",")]

        if args.warmup:
            print("[main] warmup: filler=0, cap=64 ...")
            w_cond = dict(cond, max_think_token_n=64, min_think_token_n=64)
            img, prompt = TEXT_FILLER_TRIALS[0]
            w = run_text_filler_trial(inferencer, tokenizer, img, prompt, 0, w_cond, seed=WARMUP_SEED)
            assert w["ok"], f"warmup FAILED: {w['error']}"
            print(f"[main] warmup ok: t_think={w['t_think']:.2f}s t_image={w['t_image']:.2f}s "
                  f"kv_len={w['kv_len_prefill']}")

        rows, t0 = [], time.perf_counter()
        total = len(TEXT_FILLER_TRIALS) * len(fillers) * args.repeats
        i = 0
        for pi, (img, prompt) in enumerate(TEXT_FILLER_TRIALS):
            for fi, filler_n in enumerate(fillers):
                for repeat in range(args.repeats):
                    i += 1
                    seed = SEED_BASE + pi * 10000 + fi * 100 + repeat
                    row = run_text_filler_trial(inferencer, tokenizer, img, prompt, filler_n, cond, seed)
                    row["trial_idx"] = i
                    rows.append(row)
                    st = "ok" if row["ok"] else f"FAIL({str(row['error'])[:80]})"
                    elapsed = time.perf_counter() - t0
                    print(f"[{i}/{total}] {os.path.basename(img)} filler={filler_n} repeat={repeat}: {st} "
                          f"kv_len={row.get('kv_len_prefill')} t_think={row.get('t_think')} "
                          f"t_image={row.get('t_image')} tok/s={row.get('think_tok_per_sec')} "
                          f"elapsed={elapsed:.0f}s")

        df = pd.DataFrame(rows)
        df["hardware"] = torch.cuda.get_device_name(0)
        df["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        out_csv = os.path.join(args.output_dir, "text_filler_sweep_trials.csv")
        df.to_csv(out_csv, index=False)
        print(f"\nwrote {len(df)} rows → {out_csv}")

        ok = df[df["ok"] == True]
        if len(ok):
            print("\n=== summary (text_filler) ===")
            grp = ok.groupby(["image", "filler_token_n"]).agg(
                kv_len_mean=("kv_len_prefill", "mean"),
                t_think_mean=("t_think", "mean"),
                tok_per_sec_mean=("think_tok_per_sec", "mean"),
                t_image_mean=("t_image", "mean"),
                n=("t_think", "count"),
            )
            print(grp.to_string())
            _print_linfit(ok)
        else:
            print("no successful trials to summarize.")


if __name__ == "__main__":
    main()
