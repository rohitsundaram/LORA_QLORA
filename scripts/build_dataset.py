#!/usr/bin/env python3
"""Generate reproducible train/val JSONL for structured JSON (order payload) task."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "You convert the user's natural-language request into a single JSON object "
    "that matches the schema: product_id (integer), quantity (integer), "
    'shipping (one of \"standard\", \"express\", \"overnight\"). '
    "Reply with only valid JSON, no markdown fences or commentary."
)

SHIPPING = ["standard", "express", "overnight"]


def _utterance(pid: int, qty: int, ship: str) -> str:
    templates = [
        f"Place an order for product {pid}, quantity {qty}, {ship} shipping.",
        f"Buy {qty} units of item {pid}; use {ship} delivery.",
        f"I need {qty} of product id {pid} shipped {ship}.",
        f"Order: SKU {pid}, qty {qty}, shipping speed {ship}.",
    ]
    return random.choice(templates)


def build_example(seed: int) -> dict:
    rng = random.Random(seed)
    pid = rng.randint(1, 99999)
    qty = rng.randint(1, 50)
    ship = rng.choice(SHIPPING)
    user = _utterance(pid, qty, ship)
    assistant = json.dumps(
        {"product_id": pid, "quantity": qty, "shipping": ship},
        separators=(",", ":"),
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=800, help="Number of train examples")
    p.add_argument("--val", type=int, default=100, help="Number of val examples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    train_path = args.out_dir / "instructions_train.jsonl"
    val_path = args.out_dir / "instructions_val.jsonl"

    with train_path.open("w", encoding="utf-8") as ft, val_path.open(
        "w", encoding="utf-8"
    ) as fv:
        for i in range(args.train):
            ft.write(json.dumps(build_example(args.seed + i), ensure_ascii=False) + "\n")
        for j in range(args.val):
            fv.write(
                json.dumps(build_example(args.seed + 10_000 + j), ensure_ascii=False)
                + "\n"
            )

    print(f"Wrote {train_path} ({args.train} lines)")
    print(f"Wrote {val_path} ({args.val} lines)")


if __name__ == "__main__":
    main()
