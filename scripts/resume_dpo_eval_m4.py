#!/usr/bin/env python3
"""Resume the M4 pipeline from NB3: DPO + NB4 eval.

SFT-mini and the preference parquet were produced by run_pipeline_m4.py.
This reuses adapters/sft-mini/ and data/pref/train.parquet, runs DPO, evals
SFT vs SFT+DPO, and merges results into data/run_metrics.json.
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

REPO = Path(__file__).resolve().parent.parent
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_LEN, MAX_PROMPT_LEN = 512, 256
GRAD_ACCUM = 8
BETA = float(os.environ.get("DPO_BETA", "0.1"))
DPO_LR = float(os.environ.get("DPO_LR", "5e-6"))
ADAPTERS = REPO / "adapters"
SFT_PATH = ADAPTERS / "sft-mini"
DPO_PATH = ADAPTERS / "dpo"
PREF_DIR = REPO / "data" / "pref"
DTYPE = torch.float32
DPO_PATH.mkdir(parents=True, exist_ok=True)

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

mfile = REPO / "data" / "run_metrics.json"
metrics = json.loads(mfile.read_text()) if mfile.exists() else {
    "dataset": "Intel/orca_dpo_pairs", "base_model": BASE_MODEL, "device": DEVICE.type,
    "beta": BETA, "dpo_lr": DPO_LR, "max_length": MAX_LEN,
}

tokenizer = AutoTokenizer.from_pretrained(str(SFT_PATH))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── NB3: DPO ─────────────────────────────────────────────────────────────────
print("=== NB3: DPO training (resume) ===", flush=True)
t0 = time.time()
policy_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=DTYPE)
policy_base.config.use_cache = False
policy = PeftModel.from_pretrained(policy_base, str(SFT_PATH), is_trainable=True)
policy.to(DEVICE)
pref_ds = Dataset.from_parquet(str(PREF_DIR / "train.parquet"))
metrics["pref_pairs"] = len(pref_ds)

dpo_cfg = DPOConfig(
    output_dir=str(ADAPTERS / "dpo-checkpoints"),
    per_device_train_batch_size=1, gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=1, learning_rate=DPO_LR, beta=BETA,
    max_length=MAX_LEN, max_prompt_length=MAX_PROMPT_LEN, warmup_ratio=0.1,
    lr_scheduler_type="cosine", logging_steps=5, save_strategy="no",
    optim="adamw_torch", bf16=False, fp16=False, seed=42,
    loss_type="sigmoid", report_to="none", dataloader_pin_memory=False,
)
dpo_trainer = DPOTrainer(model=policy, ref_model=None, args=dpo_cfg,
                         train_dataset=pref_ds, processing_class=tokenizer)
dpo_res = dpo_trainer.train()
logs = pd.DataFrame(dpo_trainer.state.log_history)
metrics["dpo_final_loss"] = float(dpo_res.training_loss)
metrics["dpo_seconds"] = round(time.time() - t0, 1)
if "rewards/chosen" in logs.columns and "rewards/rejected" in logs.columns:
    d = logs.dropna(subset=["rewards/chosen", "rewards/rejected"])
    n = min(5, len(d))
    metrics["end_chosen_reward"] = float(d["rewards/chosen"].iloc[-n:].mean())
    metrics["end_rejected_reward"] = float(d["rewards/rejected"].iloc[-n:].mean())
    metrics["end_reward_gap"] = metrics["end_chosen_reward"] - metrics["end_rejected_reward"]
    if "rewards/accuracies" in d.columns:
        metrics["end_reward_accuracy"] = float(d["rewards/accuracies"].iloc[-n:].mean())
    metrics["dpo_loss_curve"] = [float(x) for x in logs["loss"].dropna().tolist()]
dpo_trainer.model.save_pretrained(str(DPO_PATH))
tokenizer.save_pretrained(str(DPO_PATH))
print(f"DPO done in {metrics['dpo_seconds']}s · final loss {metrics['dpo_final_loss']:.4f}"
      + (f" · reward gap {metrics.get('end_reward_gap', float('nan')):+.3f}"
         if "end_reward_gap" in metrics else ""), flush=True)

del policy, policy_base, dpo_trainer
gc.collect()
if DEVICE.type == "mps":
    torch.mps.empty_cache()

# ── NB4: eval ────────────────────────────────────────────────────────────────
print("=== NB4: side-by-side eval ===", flush=True)
EVAL_PROMPTS = [
    "Explain in 3-4 sentences how the quicksort algorithm works.",
    "What are three practical tips for staying focused while studying?",
    "Write a short, polite reply declining a meeting invitation.",
    "Summarize the water cycle for a 10-year-old.",
    "Give two pros and two cons of remote work.",
    "Translate to French: 'The weather is nice today.'",
    "What should I check first if my laptop won't turn on?",
    "Suggest a healthy 3-item breakfast.",
]


def gen(adapter_path, prompt):
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=DTYPE)
    m = PeftModel.from_pretrained(base, str(adapter_path))
    m.config.use_cache = True
    m.to(DEVICE).eval()
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True
    ).to(DEVICE)
    with torch.no_grad():
        out = m.generate(input_ids=ids, max_new_tokens=120, do_sample=False)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    del base, m
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return text.strip()


comparison = []
for p in EVAL_PROMPTS:
    comparison.append({"prompt": p, "sft": gen(SFT_PATH, p), "dpo": gen(DPO_PATH, p)})
    print(f"  · {p[:45]}", flush=True)
metrics["comparison"] = comparison

mfile.write_text(json.dumps(metrics, indent=2))
print("Wrote data/run_metrics.json")
print("PIPELINE COMPLETE")
