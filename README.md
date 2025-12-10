# LoRA vs QLoRA — structured JSON order payloads

Fine-tune **Qwen2.5-7B-Instruct** with **LoRA** (bf16 base + adapters) and **QLoRA** (4-bit NF4 base + same LoRA setup) on the same data and hyperparameters, then compare VRAM, throughput, automatic JSON metrics, and generations.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/build_dataset.py
python scripts/train_lora.py --config config/train_config.yaml
python scripts/train_qlora.py --config config/train_config.yaml
```

Evaluate (GPU recommended):

```bash
python scripts/eval_json.py --adapter artifacts/adapters_lora
python scripts/eval_json.py --adapter artifacts/adapters_qlora --qlora
```

Optional UI (needs trained adapters in default paths):

```bash
python scripts/demo_gradio.py
```

Run all commands from the **repository root**.

## What this project does

| Item | Choice |
|------|--------|
| **Task** | Natural language → one JSON object: `product_id` (int), `quantity` (int), `shipping` (`standard` \| `express` \| `overnight`). |
| **Base model** | `Qwen/Qwen2.5-7B-Instruct` (see `config/train_config.yaml`). |

JSON parse rate and schema validation are easy to score automatically, which makes LoRA vs QLoRA comparisons concrete.

## Repository layout

| Path | Purpose |
|------|---------|
| `config/train_config.yaml` | Shared training settings for both runs |
| `scripts/build_dataset.py` | Builds train/val JSONL |
| `scripts/train_lora.py` / `train_qlora.py` | Entry points; logic in `train_common.py` |
| `scripts/eval_json.py` | Metrics on val + spot-check file |
| `scripts/demo_gradio.py` | Side-by-side base / LoRA / QLoRA |
| `data/instructions_*.jsonl` | Generated chat data |
| `data/spot_check_human.jsonl` | Manual review prompts |
| `schemas/order_payload.schema.json` | JSON Schema for eval |
| `artifacts/adapters_lora/` / `adapters_qlora/` | Adapter checkpoints + `train_metrics.json` after training |

## Data and splits

- **Source:** Synthetic, seeded templates from `scripts/build_dataset.py` (default **800 train**, **100 val**).
- **Split:** Validation examples use seeds `42 + 10_000 + j`; training uses `42 + i`, so there is no overlap.
- **Format:** Each line is a JSON object with a `messages` array (`system`, `user`, `assistant`).

Regenerate data:

```bash
python scripts/build_dataset.py --train 800 --val 100 --seed 42 --out-dir data
```

## Evaluation metrics

1. **JSON parse rate** — share of outputs where a JSON object is extracted and `json.loads` succeeds.
2. **Schema compliance rate** — share of parsed objects valid under `schemas/order_payload.schema.json`.

`eval_json.py` reports these on `data/instructions_val.jsonl` and on `data/spot_check_human.jsonl` (use the `user` field; `notes` is for human reviewers).

Baseline without adapters:

```bash
python scripts/eval_json.py
```

## Prompts (keep in sync across train / eval / demo)

**System prompt:**

```text
You convert the user's natural-language request into a single JSON object that matches the schema: product_id (integer), quantity (integer), shipping (one of "standard", "express", "overnight"). Reply with only valid JSON, no markdown fences or commentary.
```

The same string is used in `scripts/build_dataset.py`, `scripts/eval_json.py`, and `scripts/demo_gradio.py`.

## Matched LoRA vs QLoRA

Use one `config/train_config.yaml` for both runs. LoRA loads the base in bf16; QLoRA loads it in 4-bit with bitsandbytes. LoRA targets, learning rate, epochs, batch size, gradient accumulation, and data paths should stay identical.

After each training run, read **`train_metrics.json`** in the output folder for `peak_vram_gb`, `throughput_samples_per_s`, runtime, and final loss.

For evaluation, use **`--qlora`** when loading the QLoRA adapter so the base matches how it was trained.

## Requirements

- Python 3.10+ recommended  
- NVIDIA GPU with enough VRAM for 7B LoRA (24 GB+ comfortable; QLoRA lower)  
- CUDA toolchain compatible with PyTorch and `bitsandbytes` for QLoRA  

## License

Follow the **Qwen2.5** model license and terms on Hugging Face. Synthetic data you generate locally is yours.
