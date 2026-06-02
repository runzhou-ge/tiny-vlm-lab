# tiny-vlm

A minimal LLaVA-style Vision-Language Model built from scratch for learning purposes.

## What this is

A stripped-down implementation of a Vision-Language Model (VLM) that connects a frozen CLIP vision encoder to a small causal language model via a single trainable linear projector. The goal is clarity over performance.

## What this is NOT

- Not a production system
- Not state-of-the-art
- Not distributed or multi-GPU
- Not evaluated on benchmarks
- Not a framework — just a few plain Python files

## Architecture

```
Image → CLIP (frozen) → patch embeddings → Linear Projector (trainable) ─┐
                                                                           ↓
Question → Tokenizer → Token Embeddings ──────────────────────────────────┤
                                                                           ↓
                                                              Causal LM (frozen)
                                                                           ↓
                                                                       Answer
```

- **Vision encoder**: `openai/clip-vit-base-patch32` — outputs 49 patch tokens of dim 768
- **Language model**: `HuggingFaceTB/SmolLM2-135M-Instruct` — 135M param causal LM
- **Projector**: single `nn.Linear(768, lm_hidden_dim)` — the only trained component
- **Training**: loss only on answer tokens; image tokens and prompt tokens masked with -100

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Train

```bash
uv run python train.py
```

Trains for 3 epochs on `examples/toy_data.json`. Saves checkpoint to `checkpoints/tiny-vlm.pt`.

## Inference

```bash
uv run python infer.py --image examples/example.jpg --question "What is in the image?"
```

## Toy data format

`examples/toy_data.json` is a list of objects:

```json
[
  {
    "image": "example.jpg",
    "question": "What is in the image?",
    "answer": "A red circle."
  }
]
```

Image paths are relative to the JSON file's directory.

## Limitations

- Toy dataset of 3 samples — the model memorizes rather than generalizes
- No instruction tuning, RLHF, or alignment
- No evaluation metrics
- Answer quality depends heavily on the base LM's priors; the projector only trains for a few steps
- Not suitable for any real task — educational use only
