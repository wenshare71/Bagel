#!/usr/bin/env python3
"""edit-with-think 多模态 prompt 的 t_think/t_image benchmark (pipeline 放置)。

动机: cap sweep (cap_sweep_asym_vs_pipeline.ipynb) 的 prompt 是纯文本 t2i,
而业务场景大多是 文本+图像 的编辑请求。带图后上下文多出 ViT + 输入图 VAE
token, 去噪每步的 attention 要扫更长的 KV → 预期 t_image 变大而 t_think
每 token 成本几乎不变, ratio = t_think/t_image 应比纯文本更低。本脚本实测。

用法:
  # ⚠ 必须在 import torch 之前设置 CUDA_VISIBLE_DEVICES=0,1 (单进程双卡)
  CUDA_VISIBLE_DEVICES=0,1 python experiments/scripts/run_edit_think_bench.py
  CUDA_VISIBLE_DEVICES=0,1 python experiments/scripts/run_edit_think_bench.py --cfg realistic

设计:
  - 只用 pipeline (13/15 accelerate) 放置: 阶段0 asym 人为放大了 t_image
    (每层跨卡搬 VAE 激活), 不代表业务基线, 故对照组选 pipeline。
  - 4 个 trial = 2 张图 (test_images/{octupusy,women}.jpg) × 2 条编辑 prompt。
  - cap=1000 (预算强制 min=max=cap, 与 cap sweep 同口径), N=10。
  - --cfg bench    (默认): cfg_text=cfg_img=1.0, 与 cap sweep 完全同条件,
                    差异可归因于多模态上下文本身 (去噪每步 1 次前向)。
  - --cfg realistic: cfg_text=4.0, cfg_img=2.0 (README 编辑推荐值),
                    去噪每步 3 次前向, 更接近业务真实开销。
  - 逐 trial 打印 t_prefill/t_think/t_image/ratio + 上下文 KV 长度,
    最后输出 4 个 trial 的平均 t_think/t_image, 并对照纯文本 cap sweep
    的 pipeline 参考值 (cap=1000, N=10: t_think 65.0s / t_image 5.45s)。

计时口径 (对齐 inferencer.interleave_inference 的 edit-with-think 分支):
  t_prefill = system prompt + 输入图 (VAE+ViT) + 编辑文本 的 KV 构建
  t_think   = gen_text (预算强制到 cap)
  t_image   = gen_image (输出尺寸 = vae_transform resize 后的输入图尺寸)
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

OUTPUT_DIR = os.path.join(_proj_root, "experiments", "outputs", "edit_think_outputs")

# 纯文本 cap sweep 的 pipeline 参考值 (cap=1000, N=10, 96-trial 实验 9870ca7)
TEXT_ONLY_REF = dict(t_think=65.020, t_image=5.446)

# 2 张图 × 2 条编辑 prompt = 4 个 trial (前两条取自 inference.ipynb 官方示例)
EDIT_TRIALS = [
    ("test_images/women.jpg",
     "She boards a modern subway, quietly reading a folded newspaper, wearing the same clothes."),
    ("test_images/women.jpg",
     "Change the background to a rainy street at night, keep her pose and clothes unchanged."),
    ("test_images/octupusy.jpg",
     "Could you display the sculpture that takes after this design?"),
    ("test_images/octupusy.jpg",
     "Turn this into a watercolor illustration with a light blue background."),
]

CFG_PRESETS = {
    # 与 cap sweep 同条件: CFG 关闭 → 去噪每步 1 次前向, 差异可归因于多模态上下文
    "bench": dict(cfg_text_scale=1.0, cfg_img_scale=1.0),
    # README/inference.ipynb 的编辑推荐值 → 去噪每步 3 次前向 (cond+cfg_text+cfg_img)
    "realistic": dict(cfg_text_scale=4.0, cfg_img_scale=2.0),
}


def run_edit_trial(inferencer, tokenizer, image_path, prompt, cond, seed):
    """edit-with-think 单 trial, 复刻 interleave_inference 的上下文构建顺序并分段计时。"""
    reset_taylorseer_state(inferencer.model)
    set_all_seeds(seed)
    record = dict(image=os.path.basename(image_path), prompt=prompt, seed=seed, **cond)
    gen_context = cfg_text_context = cfg_img_context = None
    try:
        image = Image.open(os.path.join(_proj_root, image_path))
        gen_context = inferencer.init_gen_context()
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            with sync_timer() as t_prefill:
                gen_context = inferencer.update_context_text(GEN_THINK_SYSTEM_PROMPT, gen_context)
                cfg_img_context = inferencer.update_context_text(GEN_THINK_SYSTEM_PROMPT, cfg_img_context)

                # 图像: 只进 gen_context (cfg_img 不含图), 输出尺寸取 resize 后的输入图
                image = inferencer.vae_transform.resize_transform(pil_img2rgb(image))
                gen_context = inferencer.update_context_image(image, gen_context, vae=True, vit=True)
                image_shape = image.size[::-1]
                cfg_text_context = deepcopy(gen_context)

                # 编辑文本: cfg_text 不含 (快照已拍), gen/cfg_img 含
                gen_context = inferencer.update_context_text(prompt, gen_context)
                cfg_img_context = inferencer.update_context_text(prompt, cfg_img_context)

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

        think_token_count = len(tokenizer(gen_text, add_special_tokens=False).input_ids)
        record.update(
            t_prefill=t_prefill.elapsed, t_think=t_think.elapsed, t_image=t_image.elapsed,
            ratio_think_image=t_think.elapsed / t_image.elapsed,
            think_token_count=think_token_count,
            think_tok_per_sec=think_token_count / t_think.elapsed,
            kv_len_prefill=kv_len_prefill,
            image_shape=f"{image_shape[0]}x{image_shape[1]}",
            gen_text=gen_text, ok=True, error=None,
        )
    except Exception as e:
        record.update(
            t_prefill=None, t_think=None, t_image=None, ratio_think_image=None,
            think_token_count=None, think_tok_per_sec=None, kv_len_prefill=None,
            image_shape=None, gen_text=None, ok=False, error=repr(e),
        )
    finally:
        del gen_context, cfg_text_context, cfg_img_context
        reset_taylorseer_state(inferencer.model)
        gc.collect()
        torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser(description="edit-with-think ratio benchmark (pipeline)")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--cap", type=int, default=1000)
    parser.add_argument("--num-timesteps", type=int, default=10)
    parser.add_argument("--cfg", choices=list(CFG_PRESETS), default="bench")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    args = parser.parse_args()

    cond = dict(
        num_timesteps=args.num_timesteps,
        max_think_token_n=args.cap,
        # 预算强制 (s1 式, 与 cap sweep 同口径): think 精确到 cap
        min_think_token_n=args.cap,
        wait_interjection=" Wait,",
        cfg_interval=[0.4, 1.0], cfg_renorm_min=0.0, cfg_renorm_type="global",
        timestep_shift=3.0, enable_taylorseer=False,
        **CFG_PRESETS[args.cfg],
    )
    print(f"[main] placement=pipeline cap={args.cap} N={args.num_timesteps} cfg={args.cfg} "
          f"trials={len(EDIT_TRIALS)}")
    print(f"[main] visible GPUs: {torch.cuda.device_count()} "
          f"({[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})")

    model, vae_model, tokenizer, new_token_ids = _load_model_pipeline(args.model_path)
    inferencer = InterleaveInferencer(
        model=model, vae_model=vae_model, tokenizer=tokenizer,
        vae_transform=ImageTransform(1024, 512, 16),
        vit_transform=ImageTransform(980, 224, 14),
        new_token_ids=new_token_ids,
    )

    if args.warmup:
        w_cond = dict(cond, max_think_token_n=64, min_think_token_n=64)
        print("[main] warmup: edit trial, cap=64 ...")
        w = run_edit_trial(inferencer, tokenizer, *EDIT_TRIALS[0], w_cond, seed=WARMUP_SEED)
        assert w["ok"], f"warmup FAILED: {w['error']}"
        print(f"[main] warmup ok: t_think={w['t_think']:.2f}s t_image={w['t_image']:.2f}s "
              f"kv_len={w['kv_len_prefill']}")

    rows, t0 = [], time.perf_counter()
    for i, (img, prompt) in enumerate(EDIT_TRIALS):
        row = run_edit_trial(inferencer, tokenizer, img, prompt, cond, seed=SEED_BASE + i)
        row["trial_idx"] = i
        rows.append(row)
        st = "ok" if row["ok"] else f"FAIL({str(row['error'])[:80]})"
        if row["ok"]:
            print(f"[{i + 1}/{len(EDIT_TRIALS)}] {row['image']}: {st} "
                  f"t_prefill={row['t_prefill']:.2f}s t_think={row['t_think']:.1f}s "
                  f"t_image={row['t_image']:.1f}s ratio={row['ratio_think_image']:.2f} "
                  f"tok/s={row['think_tok_per_sec']:.2f} kv_len={row['kv_len_prefill']} "
                  f"shape={row['image_shape']} elapsed={time.perf_counter() - t0:.0f}s")
        else:
            print(f"[{i + 1}/{len(EDIT_TRIALS)}] {row['image']}: {st}")

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df["hardware"] = torch.cuda.get_device_name(0)
    df["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_csv = os.path.join(args.output_dir, f"edit_think_trials_{args.cfg}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {len(df)} rows → {out_csv}")

    ok = df[df["ok"] == True]
    if len(ok):
        ratio = ok["ratio_think_image"].mean()
        ref_ratio = TEXT_ONLY_REF["t_think"] / TEXT_ONLY_REF["t_image"]
        print(f"\n===== 摘要 (cap={args.cap}, N={args.num_timesteps}, cfg={args.cfg}, "
              f"{len(ok)}/{len(df)} ok) =====")
        print(f"t_prefill        mean = {ok['t_prefill'].mean():8.2f} s")
        print(f"t_think          mean = {ok['t_think'].mean():8.2f} s "
              f"({ok['think_tok_per_sec'].mean():.2f} tok/s)")
        print(f"t_image          mean = {ok['t_image'].mean():8.2f} s")
        print(f"kv_len(prefill)  mean = {ok['kv_len_prefill'].mean():8.0f} tokens")
        print(f"\nt_think/t_image  mean = {ratio:.2f}")
        print(f"纯文本参考 (cap sweep pipeline, cap=1000/N=10, cfg=1.0): "
              f"{TEXT_ONLY_REF['t_think']:.1f}/{TEXT_ONLY_REF['t_image']:.2f} = {ref_ratio:.2f}")
        if args.cfg == "bench":
            print(f"多模态 vs 纯文本 ratio 变化: x{ratio / ref_ratio:.2f}")
        else:
            print("(realistic cfg 与纯文本参考的去噪前向次数不同, ratio 不能直接对比)")


if __name__ == "__main__":
    main()
