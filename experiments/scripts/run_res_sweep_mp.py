#!/usr/bin/env python3
"""
resolution_sweep_benchmark 多卡并行 sweep 脚本(最终实验)

在 think_cap 实验(cap × N)基础上加入第三维:图像分辨率 R。
网格: R ∈ {1024, 768, 512, 256} × N ∈ {50, 10, 5} × cap ∈ {1000, 256, 128, 64, 32}
图像 token 数 = (R / 16)^2 → 4096 / 2304 / 1024 / 256(上限 R=1024, max_latent_size=64)

用法:
  python experiments/scripts/run_res_sweep_mp.py --gpus 0,1,2,3,4,5,6,7

设计(与 run_cap_sweep_mp.py 一致):
  - 每个 worker 进程拥有独立的模型实例 (2 GPUs via accelerate device_map)
  - trials 均匀分配, 结果写入 experiments/think_res_outputs/worker_*.csv, 自动合并
  - warm-up 按分辨率各做一次 (attention/cudnn kernel 按 shape 缓存)
"""

import os
import sys
import json
import time
import random
import gc
import argparse
import subprocess
from copy import deepcopy
from contextlib import contextmanager

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── 路径 ──
_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
import pandas as pd

# ── 常量 ──
MODEL_PATH = os.path.join(_proj_root, "BAGEL-7B-MoT")
OUTPUT_DIR = os.path.join(_proj_root, "experiments", "outputs", "think_res_outputs")

LATENT_DOWNSAMPLE = 16  # VAE 8x × latent_patch 2x → token 数 = (side/16)^2
IMAGE_SIDE_LIST = [1024, 768, 512, 256]  # 边长必须是 16 的倍数且 ≤1024 (max_latent_size=64)
NUM_TIMESTEPS_LIST = [50, 10, 5]
MAX_THINK_TOKEN_LIST = [1000, 256, 128, 64, 32]
FORCE_THINK_LENGTH = False  # budget forcing 开关, 见 experiments/BUDGET_FORCING.md
WAIT_INTERJECTION = " Wait,"
N_REPEATS = 2
SWEEP_SHUFFLE_SEED = 42
SEED_BASE = 1000
N_PROMPTS_PER_SOURCE = 8
PROMPT_SEED = 0

GENEVAL_PROMPTS_PATH = os.path.join(_proj_root, "eval/gen/geneval/prompts/evaluation_metadata.jsonl")
WISE_PROMPTS_PATH = os.path.join(_proj_root, "eval/gen/wise/final_data.json")


def _apply_overrides(caps: str = None, force_think: bool = False):
    """补跑用: 覆盖模块级常量。必须在 build_trials() 之前调用, 且 main 与 worker 两侧
    都要调用同样的覆盖, 否则 worker 重建的 trial 列表与主进程分配的 index 对不上。"""
    global MAX_THINK_TOKEN_LIST, FORCE_THINK_LENGTH
    if caps:
        MAX_THINK_TOKEN_LIST = [int(x) for x in caps.split(",")]
    if force_think:
        FORCE_THINK_LENGTH = True


def build_conditions():
    """构建 condition 列表 (R × N × cap 完全交叉)"""
    return [
        dict(
            image_side=side,
            num_timesteps=n, max_think_token_n=cap,
            min_think_token_n=cap if FORCE_THINK_LENGTH else 0,
            wait_interjection=WAIT_INTERJECTION if FORCE_THINK_LENGTH else None,
            cfg_text_scale=1.0, cfg_img_scale=1.0,
            cfg_interval=[0.4, 1.0], cfg_renorm_min=0.0, cfg_renorm_type="global",
            timestep_shift=3.0, enable_taylorseer=False,
        )
        for side in IMAGE_SIDE_LIST
        for n in NUM_TIMESTEPS_LIST
        for cap in MAX_THINK_TOKEN_LIST
    ]


def build_trials():
    """构建 trial 列表(prompt 采样与 think_cap 实验完全一致, 保证可比)"""
    with open(GENEVAL_PROMPTS_PATH, "r", encoding="utf-8") as f:
        geneval_all = [json.loads(line)["prompt"] for line in f]
    with open(WISE_PROMPTS_PATH, "r", encoding="utf-8") as f:
        wise_all = [d["Prompt"] for d in json.load(f)]

    _rng = random.Random(PROMPT_SEED)
    sampled_geneval = _rng.sample(geneval_all, N_PROMPTS_PER_SOURCE)
    sampled_wise = _rng.sample(wise_all, N_PROMPTS_PER_SOURCE)

    sampled_prompts = (
        [dict(source="geneval", prompt=p, prompt_id=i)
         for i, p in enumerate(sampled_geneval)]
        + [dict(source="wise", prompt=p, prompt_id=i + N_PROMPTS_PER_SOURCE)
           for i, p in enumerate(sampled_wise)]
    )

    all_conditions = build_conditions()

    trials = [
        (p["prompt"], p["source"], p["prompt_id"], cond, repeat)
        for p in sampled_prompts
        for cond in all_conditions
        for repeat in range(N_REPEATS)
    ]

    _shuffle_rng = random.Random(SWEEP_SHUFFLE_SEED)
    _shuffle_rng.shuffle(trials)

    return trials, all_conditions


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU parallel sweep for resolution_sweep_benchmark")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--gpus-per-worker", type=int, default=2,
                        help="GPUs per worker (default: 2; model ~28GB doesn't fit on single 24GB GPU)")
    parser.add_argument("--force-think", action="store_true",
                        help="开启 budget forcing (min=max=cap), 见 experiments/BUDGET_FORCING.md")
    parser.add_argument("--caps", type=str, default=None,
                        help="逗号分隔的 cap 列表, 覆盖 MAX_THINK_TOKEN_LIST (补跑用, 如 1000,256)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="输出目录 (补跑时换一个目录, 避免覆盖已有 trials.csv)")
    args = parser.parse_args()

    _apply_overrides(args.caps, args.force_think)
    output_dir = args.output_dir

    all_gpus = [int(x.strip()) for x in args.gpus.split(",")]
    gpus_per_worker = args.gpus_per_worker

    if len(all_gpus) % gpus_per_worker != 0:
        print(f"ERROR: {len(all_gpus)} GPUs cannot be evenly divided by {gpus_per_worker}")
        sys.exit(1)

    n_workers = len(all_gpus) // gpus_per_worker
    worker_gpus = [all_gpus[i * gpus_per_worker:(i + 1) * gpus_per_worker]
                   for i in range(n_workers)]

    os.makedirs(output_dir, exist_ok=True)

    trials, all_conditions = build_trials()
    total_trials = len(trials)
    print(f"Total trials: {total_trials} "
          f"({len(IMAGE_SIDE_LIST)}R × {len(NUM_TIMESTEPS_LIST)}N × {len(MAX_THINK_TOKEN_LIST)}cap "
          f"× {2 * N_PROMPTS_PER_SOURCE} prompts × {N_REPEATS} repeats)")
    print(f"Workers: {n_workers} (each using {gpus_per_worker} GPU(s))")
    for wid, gpus in enumerate(worker_gpus):
        print(f"  Worker {wid}: GPU(s) {gpus}")
    print(f"Trials per worker: ~{total_trials // n_workers}")
    print(f"Estimated time: ~{8.0 / n_workers:.1f}h\n")

    # 交错分配 trials
    trial_indices_per_worker = [[] for _ in range(n_workers)]
    for i in range(total_trials):
        trial_indices_per_worker[i % n_workers].append(i)

    # ── 用 subprocess 启动各 worker ──
    # 关键: CUDA_VISIBLE_DEVICES 通过 subprocess env 传入，在 import torch 前生效
    script_path = os.path.abspath(__file__)
    processes = []

    for wid in range(n_workers):
        gpu_str = ",".join(str(g) for g in worker_gpus[wid])
        trial_indices_json = json.dumps(trial_indices_per_worker[wid])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_str

        cmd = [
            sys.executable, script_path,
            "--worker",
            "--worker-id", str(wid),
            "--trial-indices", trial_indices_json,
            "--output-dir", output_dir,
        ]
        if args.force_think:
            cmd.append("--force-think")
        if args.caps:
            cmd += ["--caps", args.caps]

        p = subprocess.Popen(cmd, env=env)
        processes.append((wid, p))
        print(f"[Main] Spawned worker {wid} (PID {p.pid}) on GPU(s) {gpu_str}")

    for wid, p in processes:
        p.wait()
        if p.returncode != 0:
            print(f"[Main] Worker {wid} FAILED (exit code {p.returncode})")
        else:
            print(f"[Main] Worker {wid} DONE")

    # ── 合并结果 ──
    worker_files = [os.path.join(output_dir, f"worker_{i}.csv") for i in range(n_workers)]
    dfs = []
    for f in worker_files:
        if os.path.exists(f):
            dfs.append(pd.read_csv(f))

    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged["hardware"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
        merged["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        out_path = os.path.join(output_dir, "trials.csv")
        merged.to_csv(out_path, index=False)
        print(f"\nMerged {len(merged)} rows ({merged['ok'].sum()} ok) → {out_path}")
    else:
        print("\nNo worker results to merge!")


def run_worker():
    """--worker 模式: 由 subprocess 调用, CUDA_VISIBLE_DEVICES 已在环境变量中"""
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--trial-indices", type=str)  # JSON list
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--force-think", action="store_true")
    parser.add_argument("--caps", type=str, default=None)
    args = parser.parse_args()

    if not args.worker:
        return False

    _apply_overrides(args.caps, args.force_think)

    worker_id = args.worker_id
    trial_indices = json.loads(args.trial_indices)
    output_dir = args.output_dir

    # 延迟导入重量级模块 (CUDA_VISIBLE_DEVICES 已生效)
    from safetensors.torch import load_file
    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from modeling.bagel import (
        BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
        SiglipVisionConfig, SiglipVisionModel,
    )
    from modeling.qwen2 import Qwen2Tokenizer
    from modeling.autoencoder import load_ae
    from inferencer import InterleaveInferencer, GEN_THINK_SYSTEM_PROMPT

    gpu_count = torch.cuda.device_count()
    print(f"[Worker {worker_id}] Visible GPUs: {gpu_count} ({torch.cuda.get_device_name(0)})")
    print(f"[Worker {worker_id}] Trials: {len(trial_indices)}")

    # ── 加载模型 ──
    if gpu_count == 1:
        llm_config = Qwen2Config.from_json_file(os.path.join(MODEL_PATH, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(MODEL_PATH, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        vae_model, vae_config = load_ae(local_path=os.path.join(MODEL_PATH, "ae.safetensors"))

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

        tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        print(f"[Worker {worker_id}] Loading weights (single GPU)...")
        model_state_dict = load_file(os.path.join(MODEL_PATH, "ema.safetensors"), device="cpu")
        msg = model.load_state_dict(model_state_dict, strict=False)
        print(f"[Worker {worker_id}] load_state_dict: {msg}")
        del model_state_dict
        gc.collect()

        model = model.to(dtype=torch.bfloat16, device="cuda:0").eval()
        vae_model = vae_model.to(device="cuda:0").eval()
    else:
        # 多卡 accelerate 模式 (2 GPUs per worker)
        from accelerate import load_checkpoint_and_dispatch, init_empty_weights

        print(f"[Worker {worker_id}] Loading with accelerate device_map ({gpu_count} GPUs)...")

        llm_config = Qwen2Config.from_json_file(os.path.join(MODEL_PATH, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(MODEL_PATH, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        vae_model, vae_config = load_ae(local_path=os.path.join(MODEL_PATH, "ae.safetensors"))

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

        tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # ── 手动均衡 device_map (与 run_cap_sweep_mp.py 相同, 已验证 480/480 OK) ──
        num_layers = llm_config.num_hidden_layers  # 28
        split = num_layers // 2 - 1  # 13
        device_map = {}
        for layer_idx in range(num_layers):
            gpu = 0 if layer_idx < split else 1
            device_map[f"language_model.model.layers.{layer_idx}"] = gpu

        # same_device_modules (对齐官方 app.py:81-102): 这些模块的裸输出会在
        # bagel.py 的 prepare_* 方法中直接相加/拼接, 必须与 embed_tokens 同卡
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

        print(f"[Worker {worker_id}] device_map: GPU 0 = layers 0-{split-1} + embed + aux(same-device), "
              f"GPU 1 = layers {split}-{num_layers-1} + norm/head/vit")

        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(MODEL_PATH, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder=f"/tmp/offload_{worker_id}",
        )
        model = model.eval()

        primary_device = "cuda:0"
        vae_model = vae_model.to(primary_device).eval()

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    inferencer = InterleaveInferencer(
        model=model, vae_model=vae_model, tokenizer=tokenizer,
        vae_transform=vae_transform, vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    # ── 工具函数 ──
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

    def run_trial(prompt, cond, seed):
        reset_taylorseer_state(model)
        set_all_seeds(seed)

        record = dict(prompt=prompt, seed=seed, **cond)
        record["image_token_count"] = (cond["image_side"] // LATENT_DOWNSAMPLE) ** 2
        gen_context = cfg_text_context = cfg_img_context = None
        try:
            gen_context = inferencer.init_gen_context()
            cfg_text_context = deepcopy(gen_context)
            cfg_img_context = deepcopy(gen_context)

            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                with sync_timer() as t_prefill:
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
                        (cond["image_side"], cond["image_side"]), gen_context,
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

            record.update(
                t_prefill=t_prefill.elapsed,
                t_think=t_think.elapsed,
                t_image=t_image.elapsed,
                think_token_count=think_token_count,
                hit_cap=hit_cap,
                think_closed=think_closed,
                gen_text=gen_text,
                ok=True, error=None,
            )
        except Exception as e:
            record.update(
                t_prefill=None, t_think=None, t_image=None,
                think_token_count=None, hit_cap=None, think_closed=None,
                gen_text=None, ok=False, error=repr(e),
            )
        finally:
            del gen_context, cfg_text_context, cfg_img_context
            reset_taylorseer_state(model)
            gc.collect()
            torch.cuda.empty_cache()

        return record

    # ── 重建 trials (种子固定, 结果与主进程一致) ──
    all_trials, all_conditions = build_trials()

    # ── Warm-up: 每个分辨率各一次 (attention/cudnn kernel 按 shape 缓存, 只 warm 一档
    #    会让其余分辨率的首个 trial 计时偏慢) ──
    warmup_prompt = "a photo of a cat"
    base_cond = deepcopy(all_conditions[0])
    base_cond.update(num_timesteps=5, max_think_token_n=32)  # 用最便宜的组合 warm shape
    if base_cond.get("min_think_token_n", 0) > 32:  # forcing 开启时 min 跟着压到 max
        base_cond["min_think_token_n"] = 32
    for side in IMAGE_SIDE_LIST:
        wcond = deepcopy(base_cond)
        wcond["image_side"] = side
        print(f"[Worker {worker_id}] Warm-up @ {side}x{side}...")
        _ = run_trial(warmup_prompt, wcond, seed=999)
        if not _["ok"]:
            print(f"[Worker {worker_id}] WARM-UP FAILED @ {side}: {_['error']}")
            pd.DataFrame([{"worker": worker_id, "ok": False, "error": f"warmup@{side}: {_['error']}"}]
                         ).to_csv(os.path.join(output_dir, f"worker_{worker_id}.csv"), index=False)
            return
        print(f"[Worker {worker_id}] Warm-up @ {side} OK: "
              f"t_think={_['t_think']:.2f}s t_image={_['t_image']:.2f}s")

    # ── Sweep ──
    rows = []
    t_start = time.perf_counter()
    for i, trial_idx in enumerate(trial_indices):
        prompt, source, prompt_id, cond, repeat = all_trials[trial_idx]
        seed = SEED_BASE + prompt_id * 1000 + repeat
        row = run_trial(prompt, cond, seed=seed)
        row["source"] = source
        row["prompt_id"] = prompt_id
        row["repeat"] = repeat
        row["worker"] = worker_id
        rows.append(row)

        if not row["ok"]:
            print(f"[Worker {worker_id}] [{i + 1}/{len(trial_indices)}] FAILED: {row['error']}")

        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (i + 1) * (len(trial_indices) - i - 1)
            pd.DataFrame(rows).to_csv(
                os.path.join(output_dir, f"worker_{worker_id}_partial.csv"), index=False)
            n_ok = sum(r["ok"] for r in rows)
            print(f"[Worker {worker_id}] [{i + 1}/{len(trial_indices)}] "
                  f"ok={n_ok}/{i + 1}, elapsed={elapsed:.0f}s, ETA={eta:.0f}s")
            gc.collect()
            torch.cuda.empty_cache()

    out_path = os.path.join(output_dir, f"worker_{worker_id}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    elapsed = time.perf_counter() - t_start
    n_ok = sum(r["ok"] for r in rows)
    print(f"[Worker {worker_id}] DONE: {n_ok}/{len(rows)} ok in {elapsed:.0f}s → {out_path}")
    return True


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker()
    else:
        main()
