#!/usr/bin/env python3
"""Optional Gradio UI: compare base vs LoRA vs QLoRA adapter generations side-by-side."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import gradio as gr
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "You convert the user's natural-language request into a single JSON object "
    "that matches the schema: product_id (integer), quantity (integer), "
    'shipping (one of \"standard\", \"express\", \"overnight\"). '
    "Reply with only valid JSON, no markdown fences or commentary."
)


def extract_json(text: str) -> str:
    text = text.strip()
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    return m.group(0) if m else text


def load_pair(base_model: str, adapter_dir: Path | None, qlora: bool):
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
    if adapter_dir is not None and Path(adapter_dir).exists():
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, user: str, max_new_tokens: int = 128) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return extract_json(gen)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config/train_config.yaml"))
    p.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = p.parse_args()
    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg["base_model"]
    lora_dir = Path(cfg.get("output_dir_lora", "artifacts/adapters_lora"))
    qlora_dir = Path(cfg.get("output_dir_qlora", "artifacts/adapters_qlora"))

    print("Loading base (bf16)...")
    base_model, base_tok = load_pair(base, None, qlora=False)
    print("Loading LoRA adapter...")
    lora_model, lora_tok = load_pair(base, lora_dir, qlora=False)
    print("Loading QLoRA adapter (4-bit base)...")
    qlora_model, qlora_tok = load_pair(base, qlora_dir, qlora=True)

    def run(user: str) -> tuple[str, str, str]:
        if not user.strip():
            return "", "", ""
        b = generate(base_model, base_tok, user)
        l = generate(lora_model, lora_tok, user)
        q = generate(qlora_model, qlora_tok, user)
        return b, l, q

    demo = gr.Interface(
        fn=run,
        inputs=gr.Textbox(label="User request (natural language)", lines=3),
        outputs=[
            gr.Textbox(label="Base model (no adapter)"),
            gr.Textbox(label="LoRA adapter"),
            gr.Textbox(label="QLoRA adapter"),
        ],
        title="Structured JSON order payload — base vs LoRA vs QLoRA",
        description=(
            f"Base: `{base}`. Loads adapters from `{lora_dir}` and `{qlora_dir}` if present."
        ),
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
