# heet — high-entropy English translator

Dense-English text autoencoder: a Qwen3 LoRA *compressor* rewrites English into shorter, still-readable dense English; a *decompressor* LoRA reconstructs the original.

Design: `docs/superpowers/specs/2026-10-14-dense-english-autoencoder-design.md`

## Usage

```bash
ollama pull qwen3:14b
uv sync
uv run heet gen-data --limit 20000          # data/chunks.jsonl, data/pairs.jsonl, data/rejected.jsonl
uv run heet sft --task decompress --run r1  # runs/r1/adapters/decompressor
uv run heet sft --task compress --run r1    # runs/r1/adapters/compressor
uv run heet eval --run r1                   # runs/r1/metrics.jsonl, runs/r1/samples/
uv run tensorboard --logdir runs/
```

Override any config value with `--set section.key=value`, e.g. `--set model.name=Qwen/Qwen3-1.7B --set reward.w_char=0.8`.

Tests: `uv run pytest` (fast) and `uv run pytest -m slow` (downloads tiny models).
