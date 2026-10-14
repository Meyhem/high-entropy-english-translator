# CLAUDE.md

HEET — dense-English text autoencoder. Qwen3 base (frozen) + two LoRA adapters: `compressor` (plain → dense) and `decompressor` (dense → plain). See `README.md` for usage and `docs/superpowers/specs/` for the design.

## Git workflow

- After every completed feature or fix: run `uv run pytest`, then commit and **push to `origin`** without asking. Work happens directly on `main`.
- One commit per feature, conventional-commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
- Never commit generated artifacts: `data/`, `data_*/`, `runs/` (gitignored).

## Commands

- `uv run pytest` — fast tests; `uv run pytest -m slow` downloads tiny models.
- `uv run heet gen-data | sft | eval | compress | ei` — see `README.md`.
- Stage order: `gen-data` → `sft --task decompress` → `sft --task compress` → `eval` → `ei --run r2 --init r1` → `eval --run r2`.

## Code layout

- `src/heet/cli.py` — argparse entry; heavy imports are lazy inside each command branch.
- `src/heet/models/loading.py` — base model, named adapters, `use_adapter`, batched `generate`.
- `src/heet/rewards/` — pure scoring (no training imports); reused as the stage 3 reward (`training/expert_iter.py`).
- Config lives in `configs/default.yaml`; override with `--set section.key=value`.
- Tests use stubs in `tests/stubs.py` instead of real models where possible.

## Environment

- Single RTX 5080 (16 GB). Teacher LLM runs locally in Ollama (`qwen3:14b`); stop it before training (`ollama stop qwen3:14b`).
