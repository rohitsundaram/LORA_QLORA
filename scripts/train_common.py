"""Shared LoRA / QLoRA training utilities."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def messages_to_text(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def load_jsonl_dataset(path: Path, tokenizer) -> Dataset:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = messages_to_text(tokenizer, obj["messages"])
            rows.append({"text": text})
    return Dataset.from_list(rows)


def create_model_and_tokenizer(
    cfg: dict[str, Any], qlora: bool
) -> tuple[Any, Any]:
    model_id = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if qlora:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    if cfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model, tokenizer


def train_run(cfg: dict[str, Any], qlora: bool, output_dir: str) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()

    model, tokenizer = create_model_and_tokenizer(cfg, qlora=qlora)
    train_ds = load_jsonl_dataset(Path(cfg["train_file"]), tokenizer)
    val_ds = load_jsonl_dataset(Path(cfg["val_file"]), tokenizer)

    args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        seed=cfg.get("seed", 42),
        load_best_model_at_end=False,
        dataset_text_field="text",
        max_length=cfg["max_seq_length"],
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    elapsed = time.perf_counter() - t0
    peak_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )
    samples = len(train_ds)
    steps = getattr(train_result, "global_step", None)
    throughput = samples / elapsed if elapsed > 0 else 0.0

    metrics = {
        "mode": "qlora" if qlora else "lora",
        "train_samples": samples,
        "train_runtime_s": round(elapsed, 3),
        "throughput_samples_per_s": round(throughput, 4),
        "peak_vram_gb": round(peak_gb, 4),
        "train_loss_final": round(float(train_result.training_loss), 6)
        if train_result.training_loss is not None
        else None,
        "global_step": steps,
        "output_dir": output_dir,
    }
    metrics_path = Path(output_dir) / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics
