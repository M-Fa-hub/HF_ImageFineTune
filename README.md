# Image LoRA Trainer

**Professional LoRA / QLoRA-style fine-tuning for Hugging Face Diffusers text-to-image models.**

Train personalized concepts — characters, products, styles, clothing, environments — on Stable Diffusion 1.5/2.1, SDXL, and FLUX-family checkpoints using parameter-efficient adapters. Switch between LoRA and QLoRA-style training through configuration alone. Built for reproducibility, GPU efficiency, and portfolio-quality ML engineering.

```bash
# Quick start (after install)
python scripts/train.py --config configs/lora_sdxl.yaml
python scripts/inference.py \
  --base-model stabilityai/stable-diffusion-xl-base-1.0 \
  --lora-path outputs/lora_sdxl/final \
  --prompt "portrait of sks_person in a studio" \
  --output generated.png
```

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [LoRA explained](#2-lora-explained)
3. [QLoRA explained](#3-qlora-explained)
4. [QLoRA limitations for diffusion models](#4-qlora-limitations-for-diffusion-models)
5. [Supported architectures](#5-supported-architectures)
6. [Installation](#6-installation)
7. [CUDA prerequisites](#7-cuda-prerequisites)
8. [Dataset format](#8-dataset-format)
9. [Image preprocessing & captions](#9-image-preprocessing--captions)
10. [Configuration reference](#10-configuration-reference)
11. [Training examples](#11-training-examples)
12. [Inference examples](#12-inference-examples)
13. [Hugging Face Hub usage](#13-hugging-face-hub-usage)
14. [Resume training](#14-resume-training)
15. [GPU memory optimization](#15-gpu-memory-optimization)
16. [Expected VRAM](#16-expected-vram)
17. [CLI reference](#17-cli-reference)
18. [Training outputs](#18-training-outputs)
19. [Logging & experiment tracking](#19-logging--experiment-tracking)
20. [Repository structure](#20-repository-structure)
21. [Development & testing](#21-development--testing)
22. [Troubleshooting](#22-troubleshooting)
23. [Model licensing & responsibility](#23-model-licensing--responsibility)
24. [Known limitations](#24-known-limitations)
25. [License](#25-license)

---

## 1. What this project does

`image-lora-trainer` is a modular Python package for fine-tuning pretrained **text-to-image diffusion models** from the Hugging Face Hub (or local Diffusers folders).

### Goals

| Goal | How it is achieved |
|---|---|
| Parameter-efficient training | LoRA adapters via Hugging Face **PEFT** |
| Memory-efficient training | Optional **QLoRA-style** 4/8-bit base quantization + gradient checkpointing |
| Architecture flexibility | Model factory + capability detection (SD / SDXL / FLUX) |
| Reproducibility | Seeded RNGs, saved configs, environment snapshots, training summaries |
| Production hygiene | Typed YAML configs, structured logging, secret redaction, tests |
| Easy switching | LoRA ↔ QLoRA via `quantization.enabled` — same trainer code |

### What you get after training

Small adapter files (typically tens of MB), **not** a full duplicate of the base model:

```text
outputs/my_run/final/denoiser/
  adapter_model.safetensors
  adapter_config.json
```

Load them at inference time on top of the original base checkpoint.

---

## 2. LoRA explained

**LoRA (Low-Rank Adaptation)** freezes the pretrained base weights and injects small trainable matrices into selected layers (usually attention projections such as `to_q`, `to_k`, `to_v`, `to_out.0`).

For a weight matrix \(W\), LoRA approximates the update as:

\[
W' = W + \frac{\alpha}{r} BA
\]

where \(B\) and \(A\) are low-rank matrices of rank \(r\), and \(\alpha\) scales the update.

### Why LoRA is useful for image models

- **Much less VRAM** than full fine-tuning
- **Fast iteration** on small personal datasets
- **Portable adapters** you can share without redistributing the base model
- **Composable** — multiple LoRAs can be mixed at inference (scale controls strength)

### Typical knobs

```yaml
lora:
  enabled: true
  rank: 16          # higher = more capacity, more VRAM
  alpha: 16         # scaling; often set equal to rank
  dropout: 0.05
  bias: none
  train_text_encoder: false
  train_text_encoder_2: false
  target_modules:
    - to_q
    - to_k
    - to_v
    - to_out.0
```

If `target_modules` is omitted, the trainer selects sensible defaults for the detected architecture (and can discover Linear module suffixes when needed).

---

## 3. QLoRA explained

**QLoRA** was introduced for large language models. The recipe is:

1. Load the **frozen base model in 4-bit** (commonly NF4 + double quantization)
2. Keep **LoRA adapters in higher precision** (bf16 / fp16)
3. Train only the adapters

This repository supports a **QLoRA-style** mode for Diffusers denoisers using:

- `bitsandbytes` quantization configs
- Hugging Face `transformers` / `diffusers` `quantization_config`
- PEFT LoRA on top of the quantized base

Enable it with:

```yaml
lora:
  enabled: true

quantization:
  enabled: true
  bits: 4
  quant_type: nf4
  double_quant: true
  compute_dtype: bfloat16
  quantize_denoiser: true
  quantize_text_encoder: false
  allow_quantization_fallback: false
```

Use the ready-made recipe:

```bash
accelerate launch scripts/train.py --config configs/qlora_sdxl.yaml
```

---

## 4. QLoRA limitations for diffusion models

> **Correctness over marketing.** This project will not pretend that LLM QLoRA maps 1:1 onto UNets and DiT/FLUX transformers.

| Topic | Behavior here |
|---|---|
| What can be quantized | Primarily **denoiser** Linear layers (UNet / Transformer), when bitsandbytes supports it |
| What is never quantized for training correctness | **VAE** (encode/decode must stay numerically stable) |
| Text encoders | Full/mixed precision by default; quantization is opt-in and experimental |
| Trainable parameters | LoRA adapters remain floating-point |
| Silent fallback | **Never.** If 4-bit is requested but unsupported/unavailable, training fails unless you set `allow_quantization_fallback: true` |
| Naming | We call this **QLoRA-style PEFT**, not “identical to LLM QLoRA” |

If true 4-bit training is unavailable, the closest valid memory-efficient mode is usually:

- standard LoRA
- gradient checkpointing
- batch size 1 + gradient accumulation
- optional `adamw_8bit`
- VAE slicing / tiling

…and the capability checks / logs will say so explicitly.

---

## 5. Supported architectures

| Family | Denoiser | Text encoders | Notes |
|---|---|---|---|
| **Stable Diffusion 1.5** | UNet | CLIP | Great starter; 512² typical |
| **Stable Diffusion 2.1** | UNet | CLIP | Similar LoRA path to SD1.5 |
| **SDXL** | UNet | CLIP + CLIP-WithProjection | Dual tokenizers; SDXL time-ids conditioning |
| **FLUX-family** | Transformer | CLIP + T5 | Flow-matching objective; verify **license** before fine-tune/share |
| **Other Diffusers T2I** | Via adapters | Varies | Set `model.architecture` explicitly; unknown arches disable 4-bit until identified |

Inspect before you spend GPU time:

```bash
image-lora-trainer inspect-model --model stabilityai/stable-diffusion-xl-base-1.0
```

This reports detected architecture, LoRA target modules, quantization support, and rough trainable-parameter estimates.

---

## 6. Installation

### Requirements

- **Python ≥ 3.11** (3.12 recommended)
- NVIDIA GPU + CUDA for practical training
- Git (optional, for environment commit recording)

### Create an environment

```bash
# from repository root
python -m venv .venv

# Windows (PowerShell / cmd)
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

python -m pip install -U pip
```

### Install this package

```bash
# core + developer tools
pip install -e ".[dev]"

# optional extras
pip install -e ".[qlora]"       # bitsandbytes (4/8-bit)
pip install -e ".[xformers]"    # memory-efficient attention (where compatible)
pip install -e ".[wandb]"       # Weights & Biases
# or everything:
pip install -e ".[all]"
```

### Hugging Face authentication

Needed for gated models (and for Hub uploads):

```bash
hf auth login
```

Or copy `.env.example` → `.env` and set:

```bash
HF_TOKEN=hf_xxxxxxxx
# optional
WANDB_API_KEY=
```

Never commit real tokens. Logs attempt to redact token-like strings.

---

## 7. CUDA prerequisites

1. Install a recent NVIDIA driver.
2. Install a **CUDA-matched PyTorch** build *before* (or carefully alongside) project deps:

```bash
# Example: CUDA 12.4 wheels — pick the index that matches your driver/CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

3. Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

### Platform notes

| Component | Notes |
|---|---|
| **bitsandbytes** | Best supported on Linux. Windows support varies by wheel/build. |
| **xFormers** | Optional; mainly helpful for some UNet attention paths. |
| **bf16** | Prefer Ampere+ GPUs (RTX 30/40, A100, etc.). Use `fp16` on older cards. |
| **CPU-only** | Package installs and unit tests run; real SDXL/FLUX training is not practical. |

---

## 8. Dataset format

### A. Local folder + `metadata.jsonl` (recommended)

```text
my_dataset/
├── 0001.jpg
├── 0002.jpg
├── 0003.png
└── metadata.jsonl
```

`metadata.jsonl` (one JSON object per line):

```json
{"file_name": "0001.jpg", "text": "portrait of sks_person standing outside"}
{"file_name": "0002.jpg", "text": "sks_person wearing a black jacket"}
{"file_name": "0003.png", "text": "sks_person in a sunny park"}
```

Supported image extensions: **`.jpg` / `.jpeg` / `.png` / `.webp`**.

A tiny example corpus ships at [`examples/dataset/`](examples/dataset/).

### B. Hugging Face Hub datasets

```yaml
dataset:
  dataset_name: owner/dataset
  dataset_config_name: null   # optional subset
  image_column: image
  caption_column: text
```

### C. Image folder + default / instance caption

```yaml
dataset:
  dataset_path: ./my_images
  default_caption: "a photo of sks_person"
  # or
  instance_prompt: "a photo of sks_person"
  trigger_word: sks_person
```

You can also place a sibling `.txt` caption next to each image (`0001.txt`).

### Validate before training

The validator checks missing files, corrupted images, missing captions, bad extensions, and empty folders:

```bash
image-lora-trainer validate-dataset --dataset ./examples/dataset
# or
python scripts/validate_dataset.py --dataset ./examples/dataset
```

Example success output:

```json
{
  "ok": true,
  "num_images": 3,
  "num_captions": 3,
  "issues": []
}
```

---

## 9. Image preprocessing & captions

### Preprocessing principles

- Resize by **short side**, then **center/random crop** (or pad) — images are not naively stretched to square.
- Configurable resolution, flip, interpolation, normalization.
- SDXL-style conditioning metadata (`original_sizes`, `crop_top_lefts`, `target_sizes`) is produced when needed.

```yaml
preprocess:
  resolution: 1024
  center_crop: true
  random_crop: false
  horizontal_flip: true
  keep_aspect_ratio: true
  interpolation: lanczos
```

### Caption controls

Useful for character / product / style concepts:

```yaml
dataset:
  trigger_word: sks_person
  caption_prefix: "photo of"
  caption_suffix: "highly detailed"
  caption_dropout: 0.05
  shuffle_captions: false
```

| Option | Purpose |
|---|---|
| `trigger_word` | Rare token the model should associate with your concept |
| `instance_prompt` / `default_caption` | Fallback when per-image text is missing |
| `caption_dropout` | Occasionally drop captions (regularization) |
| `shuffle_captions` | Shuffle comma-separated fragments |

**Tip:** Keep captions consistent. Include the trigger word in most captions. Prefer 10–50 strong images over hundreds of noisy ones for personal concepts.

---

## 10. Configuration reference

Configs live in [`configs/`](configs/):

| File | Purpose |
|---|---|
| `configs/lora_sdxl.yaml` | Standard SDXL LoRA |
| `configs/qlora_sdxl.yaml` | SDXL QLoRA-style (4-bit UNet + LoRA) |
| `configs/example.yaml` | Smaller SD1.5-style starter |

### Minimal annotated example

```yaml
model:
  pretrained_model_name_or_path: "stabilityai/stable-diffusion-xl-base-1.0"
  architecture: sdxl          # auto | sd15 | sd21 | sdxl | flux
  prediction_type: auto       # epsilon | v_prediction | flow_matching | auto

dataset:
  dataset_path: "./examples/dataset"
  trigger_word: "sks_person"

preprocess:
  resolution: 1024
  center_crop: true

training:
  output_dir: "./outputs"
  run_name: "lora_sdxl"
  seed: 42
  train_batch_size: 1
  gradient_accumulation_steps: 4
  max_train_steps: 1000
  mixed_precision: bf16       # no | fp16 | bf16
  gradient_checkpointing: true
  checkpointing_steps: 250

lora:
  enabled: true
  rank: 16
  alpha: 16
  dropout: 0.05
  train_text_encoder: false
  train_text_encoder_2: false

quantization:
  enabled: false              # true => QLoRA-style

optimizer:
  name: adamw                 # adamw | adamw_8bit
  learning_rate: 1.0e-4

lr_scheduler:
  name: constant_with_warmup  # constant | linear | cosine | ...
  warmup_steps: 100

validation:
  enabled: true
  every_n_steps: 250
  prompts:
    - "portrait of sks_person in a professional studio"

logging:
  backend: tensorboard        # tensorboard | wandb | both | none

hub:
  push_to_hub: false
  private: true
  repo_id: null
```

### CLI overrides

Any dotted path can be overridden:

```bash
python scripts/train.py --config configs/lora_sdxl.yaml \
  --set training.max_train_steps=200 \
  --set lora.rank=8 \
  --set dataset.trigger_word=sks_person \
  --set quantization.enabled=false
```

Invalid combinations (e.g. quantization without LoRA, text-embedding cache while training text encoders) are rejected **before** expensive model loads when possible.

---

## 11. Training examples

### Beginner: SDXL LoRA on the example dataset

1. Install deps and authenticate if needed.
2. Validate data:

```bash
python scripts/validate_dataset.py --dataset ./examples/dataset
```

3. Train:

```bash
python scripts/train.py --config configs/lora_sdxl.yaml
```

4. Watch TensorBoard:

```bash
tensorboard --logdir outputs/lora_sdxl/logs/tensorboard
```

### QLoRA-style SDXL (lower VRAM when bitsandbytes works)

```bash
accelerate launch scripts/train.py --config configs/qlora_sdxl.yaml
```

### Multi-GPU

```bash
accelerate config   # once
accelerate launch --multi_gpu scripts/train.py --config configs/lora_sdxl.yaml
```

### SD 1.5 starter

```bash
python scripts/train.py --config configs/example.yaml
```

### Train your own concept

1. Collect 10–30 clean images of one subject/style.
2. Write captions that include a rare trigger (`sks_person`, `sks_product`, …).
3. Point `dataset.dataset_path` at your folder.
4. Start with rank 8–16, LR `1e-4`, 500–2000 steps, batch size 1.
5. Generate validation images periodically and stop when quality plateaus (overfitting is easy).

---

## 12. Inference examples

### Script

```bash
python scripts/inference.py \
  --base-model stabilityai/stable-diffusion-xl-base-1.0 \
  --lora-path outputs/lora_sdxl/final \
  --prompt "portrait of sks_person in a laboratory, soft light" \
  --negative-prompt "blurry, low quality, deformed" \
  --output generated.png \
  --width 1024 \
  --height 1024 \
  --steps 30 \
  --guidance-scale 7.5 \
  --lora-scale 1.0 \
  --seed 42 \
  --num-images 1 \
  --dtype bfloat16
```

### CLI

```bash
image-lora-trainer infer \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --lora outputs/lora_sdxl/final \
  --prompt "sks_person walking through a forest" \
  --output forest.png \
  --steps 30 \
  --lora-scale 0.9
```

`lora-scale` controls adapter strength (`1.0` = full, lower = subtler).

### Optional: merge LoRA into a full pipeline

```bash
image-lora-trainer merge-lora \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --lora outputs/lora_sdxl/final \
  --output outputs/lora_sdxl/merged
```

**Warnings**

- Merged pipelines use **much more disk** than adapters.
- Precision can change after fuse/merge.
- Quantized bases generally **cannot** be merged cleanly — keep adapters separate.

---

## 13. Hugging Face Hub usage

```yaml
hub:
  repo_id: your-username/my-sdxl-lora
  private: true
  push_to_hub: true
```

On training completion (main process), adapters under `final/` are uploaded.

Authentication sources (in order of typical use):

1. `hf auth login` cached token
2. `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` environment variables

Tokens are **never** hard-coded. Do not print them. Prefer private repos until you verify licenses and dataset rights.

---

## 14. Resume training

Latest checkpoint:

```yaml
training:
  resume_from_checkpoint: latest
```

Specific checkpoint:

```yaml
training:
  resume_from_checkpoint: ./outputs/lora_sdxl/checkpoints/checkpoint-500
```

CLI override:

```bash
python scripts/train.py --config configs/lora_sdxl.yaml \
  --set training.resume_from_checkpoint=latest
```

Checkpoints store:

- LoRA adapter weights
- optimizer / scheduler state (best effort)
- trainer step / epoch
- RNG states where practical
- training config snapshot

---

## 15. GPU memory optimization

| Technique | Config / action |
|---|---|
| Smaller micro-batch | `training.train_batch_size: 1` |
| Simulate larger batch | raise `gradient_accumulation_steps` |
| Activation savings | `training.gradient_checkpointing: true` |
| Lower precision | `mixed_precision: bf16` or `fp16` |
| QLoRA-style base | `quantization.enabled: true` (when supported) |
| 8-bit optimizer | `optimizer.name: adamw_8bit` |
| VAE memory | `enable_vae_slicing` / `enable_vae_tiling` |
| Attention | `enable_xformers` / `enable_sliced_attention` |
| Less trainable params | disable text-encoder LoRA; lower rank |
| TF32 on Ampere+ | `enable_tf32: true` |

### OOM behavior

On CUDA out-of-memory the trainer prints concrete suggestions and **does not silently change** your hyperparameters. You decide what to adjust.

---

## 16. Expected VRAM

Indicative only — resolution, TE training, attention backend, and CUDA fragmentation dominate.

| Setup | Rough ballpark |
|---|---|
| SD1.5 LoRA @ 512, bs=1 | ~6–10 GB |
| SDXL LoRA @ 1024, bs=1 + grad checkpoint | ~12–20 GB |
| SDXL QLoRA-style 4-bit UNet | Often lower than full bf16 UNet; still needs VAE/activation headroom |
| FLUX LoRA | Commonly 24 GB+ depending on settings |

If you are under 12 GB VRAM, start with SD1.5 @ 512, rank 8, batch size 1, gradient checkpointing, and consider 8-bit AdamW.

---

## 17. CLI reference

Installed entrypoint: `image-lora-trainer`

```bash
# Train
image-lora-trainer train --config configs/lora_sdxl.yaml
image-lora-trainer train --config configs/qlora_sdxl.yaml --set training.max_train_steps=100

# Validate dataset
image-lora-trainer validate-dataset --dataset ./examples/dataset

# Inspect model capabilities (no multi-GB requirement for metadata-style inspect)
image-lora-trainer inspect-model --model stabilityai/stable-diffusion-xl-base-1.0

# Inference
image-lora-trainer infer \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --lora outputs/lora_sdxl/final \
  --prompt "portrait of sks_person" \
  --output out.png

# Merge adapters into a full pipeline (optional)
image-lora-trainer merge-lora \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --lora outputs/lora_sdxl/final \
  --output outputs/lora_sdxl/merged
```

Equivalent scripts (handy without installing the console script):

| Script | Role |
|---|---|
| `scripts/train.py` | Training |
| `scripts/inference.py` | Sampling |
| `scripts/validate_dataset.py` | Dataset checks |

---

## 18. Training outputs

```text
outputs/<run_name>/
├── config.yaml                 # exact resolved config used
├── environment.json            # model id, seed, package versions, CUDA info, git commit
├── training_summary.json       # steps, loss, peak VRAM, trainable params, adapter path
├── logs/
│   ├── train.log
│   └── tensorboard/
├── checkpoints/
│   ├── checkpoint-250/
│   │   ├── lora/
│   │   ├── trainer_state.json
│   │   └── training_state.pt
│   └── checkpoint-500/
├── validation/
│   └── step_000250/
│       └── *.png
└── final/
    ├── denoiser/
    │   ├── adapter_model.safetensors
    │   └── adapter_config.json
    ├── text_encoder/           # only if trained
    └── text_encoder_2/         # only if trained
```

### Startup summary

Training prints a concise summary covering model, architecture, dataset size, method (lora/qlora), quantization, precision, LoRA rank, trainable vs total parameters, effective batch size, LR, max steps, and output path.

### `training_summary.json` (example shape)

```json
{
  "model": "stabilityai/stable-diffusion-xl-base-1.0",
  "architecture": "sdxl",
  "method": "lora",
  "lora_rank": 16,
  "quantization": null,
  "steps": 1000,
  "training_time_seconds": 0,
  "final_loss": 0.0,
  "peak_gpu_memory_gb": 0.0,
  "trainable_parameters": 0,
  "total_parameters": 0,
  "output_adapter": "outputs/lora_sdxl/final"
}
```

---

## 19. Logging & experiment tracking

```yaml
logging:
  backend: tensorboard   # tensorboard | wandb | both | none
  log_every_n_steps: 10
  project: image-lora-trainer
  entity: null           # W&B entity/team if used
```

Logged when available:

- training loss
- learning rate
- step / epoch
- GPU memory
- validation images

W&B is **optional** (`pip install -e ".[wandb]"` + `WANDB_API_KEY`).

```bash
tensorboard --logdir outputs/lora_sdxl/logs/tensorboard
```

---

## 20. Repository structure

```text
HF_ImageFineTune/
├── README.md
├── LICENSE
├── pyproject.toml
├── .env.example
├── .gitignore
├── configs/
│   ├── lora_sdxl.yaml
│   ├── qlora_sdxl.yaml
│   └── example.yaml
├── examples/
│   └── dataset/
│       ├── 0001.jpg
│       ├── 0002.jpg
│       ├── 0003.png
│       └── metadata.jsonl
├── scripts/
│   ├── train.py
│   ├── inference.py
│   └── validate_dataset.py
├── src/
│   └── image_lora_trainer/
│       ├── cli.py
│       ├── config.py
│       ├── logging_utils.py
│       ├── data/           # dataset, preprocessing, validation
│       ├── models/         # factory, adapters, quantization, capabilities
│       ├── training/       # trainer, losses, optimizer, scheduler, checkpointing
│       ├── inference/      # pipeline + merge
│       └── utils/          # device, seed, memory
└── tests/                  # unit + smoke tests (no multi-GB downloads)
```

Design principle: architecture-specific behavior lives in **adapters / factory / capabilities**, not scattered through a giant `trainer.py`.

---

## 21. Development & testing

```bash
pip install -e ".[dev]"

# lint
ruff check .

# unit + smoke tests
pytest

# static typing
mypy src/image_lora_trainer
```

Tests cover:

- config validation (bad ranks, illegal quantization combos, cache conflicts)
- dataset loading / corruption / missing captions
- capability detection & target-module discovery
- checkpoint naming / latest discovery
- CLI parsing

Tests **intentionally avoid** downloading multi-gigabyte production models.

---

## 22. Troubleshooting

| Symptom | Likely cause | What to try |
|---|---|---|
| `CUDA out of memory` | Batch/resolution too high | bs=1, lower res, grad checkpoint, QLoRA-style, 8-bit AdamW, disable TE LoRA |
| `bitsandbytes` import / load errors | Platform wheel mismatch | Install `[qlora]` on Linux CUDA, or set `quantization.enabled: false` |
| “4-bit not supported” | Architecture/capability check | Set `model.architecture` correctly, or consciously enable `allow_quantization_fallback` |
| Hub 401 / 403 | Auth or license gate | `hf auth login`; accept model license on the Hub page |
| Empty / invalid dataset | Metadata mismatch | `validate-dataset`; fix filenames/captions |
| Loss is NaN | Unstable LR / bad data | Lower LR, check images, disable TE training, try bf16↔fp16 carefully |
| Validation images fail mid-train | Pipeline edge cases (esp. FLUX) | Prefer `scripts/inference.py` on saved adapters |
| Trigger word ignored | Captions lack the token | Put `sks_…` in captions; raise `lora_scale` at inference |
| Overfitting / identity collapse | Too many steps / tiny data | Early stop; more caption variety; lower rank |

### Useful diagnostics

```bash
image-lora-trainer inspect-model --model <MODEL_ID>
python -c "from image_lora_trainer.utils.device import collect_device_info; print(collect_device_info())"
```

---

## 23. Model licensing & responsibility

This software does **not** grant rights to any upstream model or dataset.

**You** must verify:

- base model license and fine-tuning / redistribution terms
- dataset ownership, consent, and privacy constraints
- commercial-use restrictions
- generated-content policies for your jurisdiction and use case

Defaults save **LoRA adapters only**. Base weights are not automatically redistributed.

Gated models (including some FLUX checkpoints) may require accepting terms on Hugging Face before download.

---

## 24. Known limitations

- FLUX training uses a simplified forward path relative to the fullest packed-latent community recipes; treat FLUX as supported-but-evolving.
- Mid-training FLUX validation is best-effort; use post-train inference for reliable samples.
- Latent / text-embedding **caching flags** are validated for safety; a full high-throughput cache pipeline is a planned enhancement.
- bitsandbytes quality varies by OS; Linux + CUDA is the reliable QLoRA-style path.
- Not every Diffusers architecture supports the same LoRA targets or quantization mechanism — capability checks exist for that reason.

---

## 25. License

This repository’s code is released under the **MIT License** (see [`LICENSE`](LICENSE)).

Upstream model weights, tokenizers, and datasets remain under their own licenses.

---

## Quick command cheat sheet

```bash
# setup
pip install -e ".[dev]"
hf auth login

# data
python scripts/validate_dataset.py --dataset ./examples/dataset

# train
python scripts/train.py --config configs/lora_sdxl.yaml
accelerate launch scripts/train.py --config configs/qlora_sdxl.yaml

# monitor
tensorboard --logdir outputs/lora_sdxl/logs/tensorboard

# sample
python scripts/inference.py \
  --base-model stabilityai/stable-diffusion-xl-base-1.0 \
  --lora-path outputs/lora_sdxl/final \
  --prompt "portrait of sks_person" \
  --output generated.png

# quality
ruff check .
pytest
mypy src/image_lora_trainer
```

If you are building a portfolio demo: start with `configs/example.yaml` or `configs/lora_sdxl.yaml`, a clean 15-image concept dataset, and document your trigger word, step count, and sample grid in the run’s `validation/` folder.
