#!/usr/bin/env python3
"""Primary metrics: JSON parse rate and schema compliance on val or spot-check set."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import jsonschema
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Match build_dataset.py
SYSTEM_PROMPT = (
    "You convert the user's natural-language request into a single JSON object "
    "that matches the schema: product_id (integer), quantity (integer), "
    'shipping (one of \"standard\", \"express\", \"overnight\"). '
    "Reply with only valid JSON, no markdown fences or commentary."
)


def load_schema(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def extract_json(text: str) -> str | None:
    text = text.strip()
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return None


def generate(model, tokenizer, user: str, max_new_tokens: int = 128) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return gen


def load_model_eval(
    base_model: str,
    adapter_dir: Path | None,
    qlora: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if qlora:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    if adapter_dir is not None and adapter_dir.exists():
        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.eval()
    return model, tokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config/train_config.yaml"))
    p.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="PEFT adapter dir (omit for base-only baseline)",
    )
    p.add_argument(
        "--spot-check",
        type=Path,
        default=Path("data/spot_check_human.jsonl"),
    )
    p.add_argument(
        "--val",
        type=Path,
        default=Path("data/instructions_val.jsonl"),
    )
    p.add_argument("--schema", type=Path, default=Path("schemas/order_payload.schema.json"))
    p.add_argument("--qlora", action="store_true", help="Load base in 4-bit (match QLoRA train)")
    args = p.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg["base_model"]
    schema = load_schema(args.schema)
    validator = jsonschema.Draft202012Validator(schema)

    model, tokenizer = load_model_eval(base, args.adapter, args.qlora)

    def eval_file(path: Path, user_key: str | None = None) -> dict:
        parse_ok = 0
        schema_ok = 0
        n = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if user_key:
                    user = row[user_key]
                else:
                    msgs = row["messages"]
                    user = next(m["content"] for m in msgs if m["role"] == "user")
                n += 1
                raw = generate(model, tokenizer, user)
                blob = extract_json(raw)
                if blob is None:
                    continue
                parse_ok += 1
                try:
                    obj = json.loads(blob)
                    validator.validate(obj)
                    schema_ok += 1
                except (json.JSONDecodeError, jsonschema.ValidationError):
                    pass
        return {
            "file": str(path),
            "n": n,
            "json_parse_rate": round(parse_ok / n, 4) if n else 0.0,
            "schema_compliance_rate": round(schema_ok / n, 4) if n else 0.0,
        }

    val_metrics = eval_file(args.val, user_key=None)
    spot_metrics = eval_file(args.spot_check, user_key="user")
    out = {"base_model": base, "adapter": str(args.adapter) if args.adapter else None, "val": val_metrics, "spot_check": spot_metrics}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
