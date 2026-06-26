#!/usr/bin/env python3
"""Full M4 (Apple Silicon / MPS) DPO pipeline driver on a *new* dataset.

Stages, mirroring colab/Lab22_DPO_M4.ipynb but as one headless run:
  NB1  SFT-mini   — LoRA r=16 on the dataset's `chosen` responses
  NB2  pref data  — format {prompt, chosen, rejected} -> Parquet
  NB3  DPO        — TRL DPOTrainer(beta) on SFT policy + frozen ref
  NB4  eval       — fixed prompts x {SFT, SFT+DPO} side by side

Writes data/run_metrics.json which build_report.py turns into report.md.

New dataset: Intel/orca_dpo_pairs  (cols: system, question, chosen, rejected)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig, DPOConfig, DPOTrainer

REPO = Path(__file__).resolve().parent.parent

# ── config ───────────────────────────────────────────────────────────────────
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
NEW_DATASET = os.environ.get("PREF_DATASET", "Intel/orca_dpo_pairs")
SFT_SLICE = int(os.environ.get("SFT_SLICE", "300"))
PREF_SLICE = int(os.environ.get("PREF_SLICE", "400"))
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
MAX_PROMPT_LEN = int(os.environ.get("MAX_PROMPT_LEN", "256"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
BETA = float(os.environ.get("DPO_BETA", "0.1"))
DPO_LR = float(os.environ.get("DPO_LR", "5e-6"))
SFT_LR = float(os.environ.get("SFT_LR", "2e-4"))

ADAPTERS = REPO / "adapters"
SFT_PATH = ADAPTERS / "sft-mini"
DPO_PATH = ADAPTERS / "dpo"
PREF_DIR = REPO / "data" / "pref"
DTYPE = torch.float32

for p in (SFT_PATH, DPO_PATH, PREF_DIR):
    p.mkdir(parents=True, exist_ok=True)


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = device()
metrics: dict = {
    "dataset": NEW_DATASET,
    "base_model": BASE_MODEL,
    "device": DEVICE.type,
    "sft_slice": SFT_SLICE,
    "pref_slice": PREF_SLICE,
    "beta": BETA,
    "dpo_lr": DPO_LR,
    "sft_lr": SFT_LR,
    "max_length": MAX_LEN,
}


def _text(x):
    if isinstance(x, list) and x and isinstance(x[-1], dict):
        return x[-1].get("content", "")
    return x if isinstance(x, str) else ""


def norm_row(row):
    """Map Intel/orca_dpo_pairs schema -> prompt/chosen/rejected."""
    prompt = row.get("question") or row.get("prompt") or ""
    if row.get("system"):
        prompt = row["system"].strip() + "\n\n" + prompt
    return {
        "prompt": prompt,
        "chosen": _text(row.get("chosen", "")),
        "rejected": _text(row.get("rejected", "")),
    }


print(f"Device: {DEVICE} | dataset: {NEW_DATASET} | base: {BASE_MODEL}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── NB1: SFT-mini ────────────────────────────────────────────────────────────
print("\n=== NB1: SFT-mini ===")
t0 = time.time()
raw = load_dataset(NEW_DATASET, split=f"train[:{SFT_SLICE}]").map(
    norm_row, remove_columns=load_dataset(NEW_DATASET, split="train[:1]").column_names
)
raw = raw.filter(lambda r: r["chosen"] and r["rejected"] and r["chosen"] != r["rejected"])


def to_sft_text(r):
    msgs = [
        {"role": "user", "content": r["prompt"]},
        {"role": "assistant", "content": r["chosen"]},
    ]
    return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}


sft_ds = raw.map(to_sft_text, remove_columns=raw.column_names)
metrics["sft_examples"] = len(sft_ds)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=DTYPE)
model.config.use_cache = False
model.to(DEVICE)
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_cfg)

sft_cfg = SFTConfig(
    output_dir=str(ADAPTERS / "sft-mini-checkpoints"),
    per_device_train_batch_size=1, gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=1, learning_rate=SFT_LR, warmup_ratio=0.03,
    lr_scheduler_type="cosine", logging_steps=5, save_strategy="no",
    optim="adamw_torch", bf16=False, fp16=False, seed=42,
    max_length=MAX_LEN, dataset_text_field="text", report_to="none",
    dataloader_pin_memory=False,
)
sft_trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=sft_ds, processing_class=tokenizer)
sft_res = sft_trainer.train()
sft_losses = [l["loss"] for l in sft_trainer.state.log_history if "loss" in l]
metrics["sft_final_loss"] = float(sft_res.training_loss)
metrics["sft_first_loss"] = float(sft_losses[0]) if sft_losses else None
metrics["sft_loss_curve"] = [float(x) for x in sft_losses]
metrics["sft_seconds"] = round(time.time() - t0, 1)
sft_trainer.model.save_pretrained(str(SFT_PATH))
tokenizer.save_pretrained(str(SFT_PATH))
print(f"SFT done in {metrics['sft_seconds']}s · final loss {metrics['sft_final_loss']:.4f}")

import gc
del model, sft_trainer
gc.collect()
if DEVICE.type == "mps":
    torch.mps.empty_cache()

# ── NB2: preference data ─────────────────────────────────────────────────────
print("\n=== NB2: preference data ===")
pref_raw = load_dataset(NEW_DATASET, split=f"train[:{PREF_SLICE}]").map(
    norm_row, remove_columns=load_dataset(NEW_DATASET, split="train[:1]").column_names
)
pref_raw = pref_raw.filter(lambda r: r["chosen"] and r["rejected"] and r["chosen"] != r["rejected"])


def fmt_pref(r):
    pt = tokenizer.apply_chat_template(
        [{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True
    )
    return {"prompt": pt, "chosen": r["chosen"], "rejected": r["rejected"]}


pref = pref_raw.map(fmt_pref, remove_columns=pref_raw.column_names)
pref.to_parquet(str(PREF_DIR / "train.parquet"))
pref.select(range(max(0, len(pref) - 50), len(pref))).to_parquet(str(PREF_DIR / "eval.parquet"))
metrics["pref_pairs"] = len(pref)
metrics["pref_examples_preview"] = [
    {
        "prompt": pref[i]["prompt"][:200],
        "chosen": pref[i]["chosen"][:200],
        "rejected": pref[i]["rejected"][:200],
    }
    for i in range(min(3, len(pref)))
]
print(f"Wrote {len(pref)} preference pairs")

# ── NB3: DPO ─────────────────────────────────────────────────────────────────
print("\n=== NB3: DPO training ===")
t0 = time.time()
policy_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=DTYPE)
policy_base.config.use_cache = False
policy = PeftModel.from_pretrained(policy_base, str(SFT_PATH), is_trainable=True)
policy.to(DEVICE)
pref_ds = Dataset.from_parquet(str(PREF_DIR / "train.parquet"))

dpo_cfg = DPOConfig(
    output_dir=str(ADAPTERS / "dpo-checkpoints"),
    per_device_train_batch_size=1, gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=1, learning_rate=DPO_LR, beta=BETA,
    max_length=MAX_LEN, max_prompt_length=MAX_PROMPT_LEN, warmup_ratio=0.1,
    lr_scheduler_type="cosine", logging_steps=5, save_strategy="no",
    optim="adamw_torch", bf16=False, fp16=False, seed=42,
    loss_type="sigmoid", report_to="none", dataloader_pin_memory=False,
)
dpo_trainer = DPOTrainer(
    model=policy, ref_model=None, args=dpo_cfg, train_dataset=pref_ds, processing_class=tokenizer
)
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
dpo_trainer.model.save_pretrained(str(DPO_PATH))
tokenizer.save_pretrained(str(DPO_PATH))
print(f"DPO done in {metrics['dpo_seconds']}s · final loss {metrics['dpo_final_loss']:.4f}"
      + (f" · reward gap {metrics.get('end_reward_gap', float('nan')):+.3f}" if "end_reward_gap" in metrics else ""))

# ── NB4: eval (SFT vs SFT+DPO) ───────────────────────────────────────────────
print("\n=== NB4: side-by-side eval ===")
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
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=DTYPE)
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
    print(f"  · prompt done: {p[:40]}")
metrics["comparison"] = comparison

(REPO / "data" / "run_metrics.json").write_text(json.dumps(metrics, indent=2))
print("\nWrote data/run_metrics.json")
print("PIPELINE COMPLETE")
