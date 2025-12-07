#!/usr/bin/env python3
"""QLoRA (4-bit NF4 base + LoRA adapters) training run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_common import load_config, train_run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path("config/train_config.yaml"),
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    out = cfg.get("output_dir_qlora", "artifacts/adapters_qlora")
    Path(out).mkdir(parents=True, exist_ok=True)
    train_run(cfg, qlora=True, output_dir=out)


if __name__ == "__main__":
    main()
