# HEET Dense-English Autoencoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `heet`, a uv-managed PyTorch prototype that generates dense-English training pairs with an Ollama teacher, fine-tunes Qwen3 LoRA compressor/decompressor adapters, and evaluates round-trip compression, fidelity and readability.

**Architecture:** Three independently runnable stages behind one CLI (`heet gen-data | sft | eval`). A frozen Qwen3 base carries two named LoRA adapters (`compressor`, `decompressor`); the same base with adapters disabled is the reference LM. A pure `rewards` package scores (original, dense, reconstruction) triples and is shared by the data filter, the SFT eval callback and the eval CLI.

**Tech Stack:** Python 3.12, uv, torch 2.14 (cu130 wheels), transformers 5.19, peft 0.21, trl 1.15, sentence-transformers 6.1, sacrebleu 2.6, datasets 5.1, httpx 0.28, tensorboard 2.21, pyyaml, pytest.

**Spec:** `docs/superpowers/specs/2026-10-14-dense-english-autoencoder-design.md`

## Global Constraints

- Package `heet` under `src/heet/`; console script `heet = "heet.cli:main"`.
- Python dependency management with uv only (`uv sync`, `uv run ...`).
- torch from `https://download.pytorch.org/whl/cu130` (RTX 5080 / sm_120; host driver 580 supports CUDA 13.0).
- Base model default `Qwen/Qwen3-0.6B`, configurable to `Qwen/Qwen3-1.7B`; chat template always rendered with `enable_thinking=False`.
- Adapter names are exactly `compressor` and `decompressor`; saved at `runs/<run>/adapters/<name>/`.
- Teacher: Ollama `/api/chat`, default model `qwen3:14b`, `"think": false`, default base URL `http://localhost:11434`.
- Chunks: 1–3 sentences, 40–300 chars; split 98/1/1 train/val/test from id hash.
- Filter defaults: `min_similarity=0.90`, `min_ratio=0.30`, `max_ratio=0.80`, near-copy threshold `0.9`.
- LoRA: r=16, alpha=32, dropout 0.05, targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
- SFT defaults: lr 2e-4, cosine, 2 epochs, effective batch 32, max length 256, bf16, gradient checkpointing.
- Reward defaults: `w_sim=1.0, w_exact=0.2, w_chrf=0.0, w_char=0.5, w_bits=0.25, w_read=0.5, margin=ln 2`.
- Success gate: `char_ratio ≤ 0.55`, `sim ≥ 0.90`, `ppl_log_ratio ≤ margin`.
- Logging: JSONL + TensorBoard only. No hosted trackers, no hosted LLM APIs, no RL code.
- `uv run pytest` must not require network or GPU; tests needing either are marked `@pytest.mark.slow`.

## Review Focus

1. **Teacher output wrapped in noise** (quotes, `Dense:` prefix, code fences, trailing explanation lines) — expected: wrappers stripped before filtering; test in Task 5 (`test_clean_teacher_output`).
2. **Ollama not running or model not pulled** — expected: `gen-data` fails fast with a message naming the base URL and `ollama pull <model>`; test in Task 5 (`test_preflight_unreachable`).
3. **`gen-data` killed mid-write leaving a truncated last JSONL line** — expected: resume ignores the broken line and continues; test in Task 5 (`test_resume_tolerates_truncated_line`).
4. **Empty or degenerate dense/reconstruction strings** (model emits nothing) — expected: scoring returns finite numbers (ratio 0, sim from embedder, ppl 1.0 for empty text), never NaN/ZeroDivision; tests in Task 3 (`test_score_handles_empty_strings`) and Task 4 (`test_ref_lm_empty_text`).
5. **`heet eval` / compress-SFT with a missing adapter directory** — expected: eval raises a clear `FileNotFoundError` naming the path and the `heet sft --task ...` command; compress-SFT warns once and logs compressor-side metrics only; tests in Task 7 (`test_eval_missing_adapter`) and Task 8 (`test_callback_without_decompressor`).

---

## File map

| File | Responsibility |
|---|---|
| `pyproject.toml` | uv project, deps, torch index, script entry, pytest markers |
| `configs/default.yaml` | all defaults (mirrors dataclasses) |
| `src/heet/config.py` | dataclasses, YAML load, `--set` overrides, dump |
| `src/heet/jsonl.py` | robust JSONL read/append helpers |
| `src/heet/cli.py` | argparse CLI, subcommand dispatch |
| `src/heet/prompts.py` | task messages, teacher prompts, chat rendering, output cleaning |
| `src/heet/data/corpus.py` | cleaning, sentence split, chunking, splits |
| `src/heet/data/llm_client.py` | async Ollama client + preflight |
| `src/heet/data/generate.py` | stage-1 pipeline and filter |
| `src/heet/rewards/fidelity.py` | `Embedder` protocol, `SentenceEmbedder`, exact match, chrF |
| `src/heet/rewards/ref_lm.py` | `TextScore`, `RefLM` |
| `src/heet/rewards/length.py` | char ratio, bits ratio |
| `src/heet/rewards/readability.py` | perplexity-ratio penalty |
| `src/heet/rewards/combined.py` | `RewardBreakdown`, `Scorer` |
| `src/heet/models/loading.py` | tokenizer/base/adapters loading, batched generation |
| `src/heet/eval/logger.py` | JSONL + TensorBoard metrics logger |
| `src/heet/eval/samples.py` | sample-triple dumps |
| `src/heet/eval/evaluate.py` | aggregation, round trip, teacher baseline, CLI entry |
| `src/heet/training/sft.py` | dataset building, SFT run, eval callback |
| `tests/...` | one test module per source module |

---

### Task 1: Project scaffold, config and CLI skeleton

**Files:**
- Create: `pyproject.toml`, `configs/default.yaml`, `src/heet/__init__.py`, `src/heet/config.py`, `src/heet/jsonl.py`, `src/heet/cli.py`, `src/heet/data/__init__.py`, `src/heet/rewards/__init__.py`, `src/heet/models/__init__.py`, `src/heet/eval/__init__.py`, `src/heet/training/__init__.py`, `tests/__init__.py`, `tests/conftest.py`
- Test: `tests/test_config.py`, `tests/test_jsonl.py`

**Interfaces:**
- Produces:
  - `heet.config.Config` (dataclass with fields `paths, corpus, teacher, filter, model, lora, sft, generation, reward, eval`), and section dataclasses `PathsConfig, CorpusConfig, TeacherConfig, FilterConfig, ModelConfig, LoraConfig, SFTConfig, GenerationConfig, RewardConfig, EvalConfig`.
  - `heet.config.load_config(path: str | Path | None, overrides: list[str] = ()) -> Config`
  - `heet.config.dump_config(cfg: Config, path: Path) -> None`
  - `heet.jsonl.read_jsonl(path: Path) -> list[dict]` (skips blank and malformed lines, returns `[]` if missing)
  - `heet.jsonl.append_jsonl(path: Path, record: dict) -> None` (append + flush, creates parent dir)
  - `heet.jsonl.write_jsonl(path: Path, records: Iterable[dict]) -> None`
  - `heet.cli.main(argv: list[str] | None = None) -> int` with subparsers `gen-data`, `sft`, `eval`; each has `--config` (default `configs/default.yaml`) and repeatable `--set KEY=VALUE`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "heet"
version = "0.1.0"
description = "High-entropy English translator: dense-English text autoencoder"
requires-python = ">=3.12,<3.13"
dependencies = [
    "torch>=2.14",
    "transformers>=5.19,<6",
    "peft>=0.21,<0.22",
    "trl>=1.15,<2",
    "accelerate>=1.15",
    "datasets>=5.1",
    "sentence-transformers>=6.1",
    "sacrebleu>=2.6",
    "httpx>=0.28",
    "tensorboard>=2.21",
    "pyyaml>=6.0",
]

[project.scripts]
heet = "heet.cli:main"

[dependency-groups]
dev = ["pytest>=9.1"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/heet"]

[tool.uv.sources]
torch = [{ index = "pytorch-cu130" }]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: needs network (HF downloads) and/or several minutes"]
addopts = "-m 'not slow'"
```

Run: `uv sync`
Expected: resolves and installs; `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` prints `2.14.x+cu130 True`.

- [ ] **Step 2: Write failing config + jsonl tests**

`tests/conftest.py`:
```python
import pytest

from heet.config import Config


@pytest.fixture
def cfg() -> Config:
    return Config()
```

`tests/test_config.py`:
```python
import math
from pathlib import Path

import pytest

from heet.config import Config, dump_config, load_config


def test_defaults_match_spec():
    c = Config()
    assert c.model.name == "Qwen/Qwen3-0.6B"
    assert c.teacher.model == "qwen3:14b"
    assert c.filter.min_similarity == 0.90
    assert (c.filter.min_ratio, c.filter.max_ratio) == (0.30, 0.80)
    assert c.lora.r == 16 and c.lora.alpha == 32
    assert c.reward.w_sim == 1.0 and c.reward.w_char == 0.5
    assert math.isclose(c.reward.margin, math.log(2))
    assert c.eval.target_char_ratio == 0.55


def test_load_yaml_and_overrides(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("sft:\n  lr: 0.001\nmodel:\n  name: Qwen/Qwen3-1.7B\n")
    c = load_config(p, ["sft.eval_steps=50", "reward.w_chrf=0.3", "corpus.source=/tmp/x.txt"])
    assert c.sft.lr == 0.001
    assert c.model.name == "Qwen/Qwen3-1.7B"
    assert c.sft.eval_steps == 50 and isinstance(c.sft.eval_steps, int)
    assert c.reward.w_chrf == 0.3
    assert c.corpus.source == "/tmp/x.txt"


def test_unknown_key_rejected(tmp_path: Path):
    with pytest.raises(KeyError, match="sft.nope"):
        load_config(None, ["sft.nope=1"])
    p = tmp_path / "c.yaml"
    p.write_text("bogus:\n  a: 1\n")
    with pytest.raises(KeyError, match="bogus"):
        load_config(p)


def test_dump_roundtrip(tmp_path: Path):
    c = load_config(None, ["sft.lr=0.0003"])
    dump_config(c, tmp_path / "out.yaml")
    assert load_config(tmp_path / "out.yaml").sft.lr == 0.0003
```

`tests/test_jsonl.py`:
```python
from pathlib import Path

from heet.jsonl import append_jsonl, read_jsonl, write_jsonl


def test_append_and_read(tmp_path: Path):
    p = tmp_path / "sub" / "a.jsonl"
    append_jsonl(p, {"id": "1"})
    append_jsonl(p, {"id": "2"})
    assert [r["id"] for r in read_jsonl(p)] == ["1", "2"]


def test_read_skips_truncated_and_blank(tmp_path: Path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"id": "1"}\n\n{"id": "2"}\n{"id": "3", "te')
    assert [r["id"] for r in read_jsonl(p)] == ["1", "2"]


def test_missing_file_is_empty(tmp_path: Path):
    assert read_jsonl(tmp_path / "none.jsonl") == []


def test_write_overwrites(tmp_path: Path):
    p = tmp_path / "a.jsonl"
    write_jsonl(p, [{"a": 1}])
    write_jsonl(p, [{"a": 2}])
    assert read_jsonl(p) == [{"a": 2}]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py tests/test_jsonl.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'heet.config'`

- [ ] **Step 4: Implement `config.py`, `jsonl.py`, empty `__init__.py` files**

`src/heet/__init__.py`, `src/heet/{data,rewards,models,eval,training}/__init__.py`, `tests/__init__.py`: empty files.

`src/heet/config.py`:
```python
"""Typed configuration: dataclass defaults, YAML file, `--set a.b=c` overrides."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    data_dir: str = "data"
    runs_dir: str = "runs"


@dataclass
class CorpusConfig:
    # "wikitext" or a path to a local .txt file
    source: str = "wikitext"
    hf_dataset: str = "Salesforce/wikitext"
    hf_config: str = "wikitext-103-raw-v1"
    hf_split: str = "train"
    min_chars: int = 40
    max_chars: int = 300
    max_sentences: int = 3
    max_chunks: int = 200_000
    val_frac: float = 0.01
    test_frac: float = 0.01


@dataclass
class TeacherConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:14b"
    concurrency: int = 4
    max_retries: int = 3
    timeout_s: float = 120.0
    rewrite_temperature: float = 0.3
    recon_temperature: float = 0.0


@dataclass
class FilterConfig:
    min_similarity: float = 0.90
    min_ratio: float = 0.30
    max_ratio: float = 0.80
    near_copy: float = 0.9


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"
    device: str = "auto"  # "auto" -> cuda if available else cpu


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class SFTConfig:
    lr: float = 2e-4
    epochs: float = 2.0
    per_device_batch_size: int = 16
    grad_accum: int = 2
    max_length: int = 256
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"
    gradient_checkpointing: bool = True
    eval_steps: int = 200
    logging_steps: int = 10
    save_steps: int = 1000
    seed: int = 0
    # adapter dir of a trained decompressor used by the compress-task eval callback;
    # empty -> runs/<run>/adapters/decompressor if it exists
    decompressor_path: str = ""


@dataclass
class GenerationConfig:
    batch_size: int = 32
    # max_new_tokens = factor * input_tokens + extra
    max_new_tokens_factor: float = 2.0
    max_new_tokens_extra: int = 16


@dataclass
class RewardConfig:
    embedder: str = "sentence-transformers/all-mpnet-base-v2"
    w_sim: float = 1.0
    w_exact: float = 0.2
    w_chrf: float = 0.0
    w_char: float = 0.5
    w_bits: float = 0.25
    w_read: float = 0.5
    margin: float = math.log(2)
    ref_batch_size: int = 32


@dataclass
class EvalConfig:
    callback_n: int = 64
    n_samples: int = 32
    worst_k: int = 10
    target_char_ratio: float = 0.55
    target_sim: float = 0.90


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _coerce(current: Any, value: Any, key: str) -> Any:
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise TypeError(f"{key}: expected bool, got {value!r}")
        return value
    if isinstance(current, int) and isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    if isinstance(current, float) and isinstance(value, (int, float)):
        return float(value)
    if isinstance(current, str):
        return str(value)
    return value


def _set(cfg: Config, dotted: str, value: Any) -> None:
    section_name, _, field_name = dotted.partition(".")
    if not field_name or not hasattr(cfg, section_name):
        raise KeyError(f"unknown config key: {dotted}")
    section = getattr(cfg, section_name)
    if field_name not in {f.name for f in dataclasses.fields(section)}:
        raise KeyError(f"unknown config key: {dotted}")
    setattr(section, field_name, _coerce(getattr(section, field_name), value, dotted))


def load_config(path: str | Path | None, overrides: list[str] | tuple[str, ...] = ()) -> Config:
    cfg = Config()
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
        for section_name, values in data.items():
            if not hasattr(cfg, section_name) or not isinstance(values, dict):
                raise KeyError(f"unknown config section: {section_name}")
            for k, v in values.items():
                _set(cfg, f"{section_name}.{k}", v)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must be KEY=VALUE, got {item!r}")
        _set(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def dump_config(cfg: Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))
```

`src/heet/jsonl.py`:
```python
"""JSONL helpers that tolerate interrupted writes."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # truncated line from an interrupted run
    return out


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        # a previous crash may have left a partial line without newline
        if f.tell() > 0:
            with path.open("rb") as r:
                r.seek(-1, 2)
                if r.read(1) != b"\n":
                    f.write("\n")
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

- [ ] **Step 5: Write `configs/default.yaml`** (mirrors defaults so users can edit one file)

```yaml
paths:
  data_dir: data
  runs_dir: runs
corpus:
  source: wikitext
  min_chars: 40
  max_chars: 300
  max_sentences: 3
  max_chunks: 200000
  val_frac: 0.01
  test_frac: 0.01
teacher:
  base_url: http://localhost:11434
  model: qwen3:14b
  concurrency: 4
  max_retries: 3
filter:
  min_similarity: 0.90
  min_ratio: 0.30
  max_ratio: 0.80
  near_copy: 0.9
model:
  name: Qwen/Qwen3-0.6B
  dtype: bfloat16
lora:
  r: 16
  alpha: 32
  dropout: 0.05
sft:
  lr: 0.0002
  epochs: 2
  per_device_batch_size: 16
  grad_accum: 2
  max_length: 256
  eval_steps: 200
reward:
  embedder: sentence-transformers/all-mpnet-base-v2
  w_sim: 1.0
  w_exact: 0.2
  w_chrf: 0.0
  w_char: 0.5
  w_bits: 0.25
  w_read: 0.5
  margin: 0.6931471805599453
eval:
  callback_n: 64
  n_samples: 32
  worst_k: 10
  target_char_ratio: 0.55
  target_sim: 0.90
```

Add to `tests/test_config.py`:
```python
def test_default_yaml_loads():
    c = load_config(Path(__file__).parent.parent / "configs" / "default.yaml")
    assert c == Config()
```

- [ ] **Step 6: Write CLI skeleton** `src/heet/cli.py`

```python
"""`heet` command-line entry point."""

from __future__ import annotations

import argparse
import sys

from heet.config import load_config


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heet", description="Dense-English autoencoder")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gen-data", help="stage 1: chunk corpus and generate filtered pairs")
    _common(g)
    g.add_argument("--limit", type=int, default=None, help="max new chunks to send to the teacher")
    g.add_argument("--rechunk", action="store_true", help="rebuild chunks.jsonl even if present")

    s = sub.add_parser("sft", help="stage 2: train one LoRA adapter")
    _common(s)
    s.add_argument("--task", choices=["compress", "decompress"], required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--limit", type=int, default=None, help="max training pairs")

    e = sub.add_parser("eval", help="round-trip evaluation")
    _common(e)
    e.add_argument("--run", required=True)
    e.add_argument("--split", choices=["val", "test"], default="test")
    e.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if args.command == "gen-data":
        from heet.data.generate import run_gen_data

        run_gen_data(cfg, limit=args.limit, rechunk=args.rechunk)
    elif args.command == "sft":
        from heet.training.sft import run_sft

        run_sft(cfg, task=args.task, run=args.run, limit=args.limit)
    elif args.command == "eval":
        from heet.eval.evaluate import run_eval

        run_eval(cfg, run=args.run, split=args.split, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Add `tests/test_cli.py`:
```python
import pytest

from heet.cli import build_parser


def test_parser_subcommands():
    p = build_parser()
    a = p.parse_args(["sft", "--task", "compress", "--run", "r1", "--set", "sft.lr=1e-4"])
    assert (a.command, a.task, a.run, a.overrides) == ("sft", "compress", "r1", ["sft.lr=1e-4"])
    assert p.parse_args(["eval", "--run", "r1"]).split == "test"
    with pytest.raises(SystemExit):
        p.parse_args(["sft", "--task", "bogus", "--run", "r1"])
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest -v`
Expected: all tests in `test_config.py`, `test_jsonl.py`, `test_cli.py` PASS. `uv run heet --help` lists `gen-data`, `sft`, `eval`.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock configs src tests
git commit -m "feat: project scaffold, typed config, jsonl helpers, CLI skeleton"
```

---

### Task 2: Prompts and output cleaning

**Files:**
- Create: `src/heet/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `TASK_SYSTEM: dict[str, str]` with keys `"compress"`, `"decompress"`.
  - `task_messages(task: str, text: str) -> list[dict[str, str]]` (system + user)
  - `render_prompt(tokenizer, task: str, text: str) -> str` — chat template, `add_generation_prompt=True`, `enable_thinking=False`, `tokenize=False`.
  - `render_completion(tokenizer, text: str) -> str` — `text + tokenizer.eos_token` (Qwen3 eos is `<|im_end|>`).
  - `teacher_rewrite_messages(text: str) -> list[dict]`, `teacher_reconstruct_messages(dense: str) -> list[dict]`
  - `clean_output(text: str) -> str` — strips `<think>…</think>`, code fences, surrounding quotes, leading labels (`Dense:`, `Rewrite:`, `Reconstruction:`, `Output:`, `Original:`), keeps only the first non-empty paragraph.

- [ ] **Step 1: Write failing tests** `tests/test_prompts.py`

```python
import pytest

from heet.prompts import (
    clean_output,
    render_completion,
    render_prompt,
    task_messages,
    teacher_reconstruct_messages,
    teacher_rewrite_messages,
)


def test_task_messages_shape():
    m = task_messages("compress", "Hello world.")
    assert [x["role"] for x in m] == ["system", "user"]
    assert m[1]["content"] == "Hello world."
    assert task_messages("decompress", "x")[0]["content"] != m[0]["content"]
    with pytest.raises(ValueError):
        task_messages("nope", "x")


def test_teacher_messages_contain_rules_and_text():
    m = teacher_rewrite_messages("The cat sat on the mat.")
    assert m[0]["role"] == "system" and "number" in m[0]["content"].lower()
    assert m[-1]["content"].endswith("The cat sat on the mat.")
    r = teacher_reconstruct_messages("cat sat on mat")
    assert r[-1]["content"].endswith("cat sat on mat")
    assert "The cat sat on the mat." not in str(r)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Dense: cat sat on mat", "cat sat on mat"),
        ('"cat sat on mat"', "cat sat on mat"),
        ("```\ncat sat on mat\n```", "cat sat on mat"),
        ("<think>\nhmm\n</think>\n\ncat sat on mat", "cat sat on mat"),
        ("cat sat on mat\n\nNote: I removed articles.", "cat sat on mat"),
        ("Reconstruction: The cat sat.", "The cat sat."),
        ("   ", ""),
    ],
)
def test_clean_output(raw, expected):
    assert clean_output(raw) == expected


class FakeTok:
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return body + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_render_prompt_and_completion():
    p = render_prompt(FakeTok(), "compress", "Hi there.")
    assert p.endswith("</think>\n\n") and "Hi there." in p
    assert render_completion(FakeTok(), "hi") == "hi<|im_end|>"


@pytest.mark.slow
def test_render_prompt_real_qwen3_template():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM")
    p = render_prompt(tok, "decompress", "cat sat on mat")
    assert "cat sat on mat" in p and p.rstrip().endswith("</think>")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'heet.prompts'`

- [ ] **Step 3: Implement** `src/heet/prompts.py`

```python
"""Prompt construction shared by data generation, SFT and evaluation."""

from __future__ import annotations

import re

TASK_SYSTEM: dict[str, str] = {
    "compress": (
        "Rewrite the user's English text as dense English: as short as possible while a "
        "careful reader can still recover the full meaning. Abbreviate, drop function words, "
        "use telegraphic phrasing and symbols. Keep names and numbers exact. Output only the rewrite."
    ),
    "decompress": (
        "The user's text is dense English (abbreviated, telegraphic). Expand it back into the "
        "original fluent English sentence(s). Output only the expanded text."
    ),
}

TEACHER_REWRITE_SYSTEM = """You compress English into "dense English": shorter, information-dense text that a careful human reader can still understand and expand back to the original.

Rules:
- Abbreviate common words (e.g. "government" -> "gov", "approximately" -> "~", "with" -> "w/", "because" -> "b/c", "and" -> "&", "leads to"/"becomes" -> "→").
- Drop articles, copulas and auxiliaries when meaning survives ("the", "a", "is", "was", "has been").
- Use telegraphic phrasing; merge clauses with ";" or ",".
- Keep EVERY proper name and EVERY number exactly as written.
- Do not add information. Do not explain. Output only the rewrite on a single line."""

TEACHER_FEWSHOT: list[tuple[str, str]] = [
    (
        "The government announced that the new bridge, which cost approximately 40 million dollars, "
        "would be opened to traffic in March.",
        "Gov announced new bridge (~$40M) opens to traffic in March.",
    ),
    (
        "After the war, Smith returned to London, where he worked as a journalist for the Times "
        "until his retirement in 1952.",
        "Post-war Smith → London, journalist @ Times till retiring 1952.",
    ),
]

TEACHER_RECON_SYSTEM = """The user's text is "dense English": abbreviated, telegraphic English with dropped function words and symbols such as & w/ → ~ b/c.
Expand it back into complete, fluent English sentence(s) with the same meaning. Keep every name and number exactly.
Do not add information. Do not explain. Output only the expanded text."""


def task_messages(task: str, text: str) -> list[dict[str, str]]:
    if task not in TASK_SYSTEM:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASK_SYSTEM)}")
    return [{"role": "system", "content": TASK_SYSTEM[task]}, {"role": "user", "content": text}]


def render_prompt(tokenizer, task: str, text: str) -> str:
    return tokenizer.apply_chat_template(
        task_messages(task, text),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def render_completion(tokenizer, text: str) -> str:
    return text + tokenizer.eos_token


def teacher_rewrite_messages(text: str) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": TEACHER_REWRITE_SYSTEM}]
    for orig, dense in TEACHER_FEWSHOT:
        msgs.append({"role": "user", "content": f"Text: {orig}"})
        msgs.append({"role": "assistant", "content": dense})
    msgs.append({"role": "user", "content": f"Text: {text}"})
    return msgs


def teacher_reconstruct_messages(dense: str) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": TEACHER_RECON_SYSTEM}]
    for orig, d in TEACHER_FEWSHOT:
        msgs.append({"role": "user", "content": f"Dense: {d}"})
        msgs.append({"role": "assistant", "content": orig})
    msgs.append({"role": "user", "content": f"Dense: {dense}"})
    return msgs


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*$", re.MULTILINE)
_LABEL = re.compile(r"^(dense|rewrite|reconstruction|output|original|text|expanded)\s*:\s*", re.I)
_QUOTES = ('"', "'", "“", "”")


def clean_output(text: str) -> str:
    text = _THINK.sub("", text).strip()
    text = _FENCE.sub("", text).strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    text = paragraphs[0] if paragraphs else ""
    text = " ".join(text.split())
    text = _LABEL.sub("", text)
    if len(text) >= 2 and text[0] in _QUOTES and text[-1] in _QUOTES:
        text = text[1:-1].strip()
    return text
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: PASS (slow test deselected).

- [ ] **Step 5: Commit**

```bash
git add src/heet/prompts.py tests/test_prompts.py
git commit -m "feat: task and teacher prompts with output cleaning"
```

---

### Task 3: Reward components with stub models

**Files:**
- Create: `src/heet/rewards/fidelity.py`, `src/heet/rewards/length.py`, `src/heet/rewards/readability.py`, `src/heet/rewards/ref_lm.py` (dataclass + protocol only in this task), `src/heet/rewards/combined.py`
- Test: `tests/test_rewards.py`, `tests/stubs.py`

**Interfaces:**
- Consumes: `heet.config.RewardConfig`.
- Produces:
  - `fidelity.Embedder` (Protocol): `similarity(a: list[str], b: list[str]) -> list[float]`
  - `fidelity.normalize(text: str) -> str`; `fidelity.exact_match(a: str, b: str) -> float`; `fidelity.chrf(hyp: str, ref: str) -> float` (0–1)
  - `ref_lm.TextScore` dataclass: `nll_nats: float, n_tokens: int`, properties `bits -> float`, `ppl -> float` (1.0 when `n_tokens == 0`)
  - `ref_lm.ReferenceScorer` (Protocol): `score(texts: list[str]) -> list[TextScore]`
  - `length.char_ratio(dense: str, original: str) -> float` (0.0 if original empty); `length.bits_ratio(dense: TextScore, original: TextScore) -> float` (0.0 if original bits are 0)
  - `readability.readability_penalty(dense: TextScore, original: TextScore, margin: float) -> float`; `readability.ppl_log_ratio(dense, original) -> float`
  - `combined.RewardBreakdown` dataclass: `sim, exact, chrf, char_ratio, bits_ratio, ppl_dense, ppl_orig, ppl_log_ratio, read_pen, bits_dense, bits_orig, total` (all float); `to_dict() -> dict`
  - `combined.Scorer(cfg: RewardConfig, embedder: Embedder, ref: ReferenceScorer | None)` with `score(originals, denses, recons, compute_all: bool = True) -> list[RewardBreakdown]`

- [ ] **Step 1: Write stubs** `tests/stubs.py`

```python
"""Deterministic stand-ins for the embedding model and reference LM."""

from difflib import SequenceMatcher

from heet.rewards.ref_lm import TextScore


class StubEmbedder:
    def similarity(self, a: list[str], b: list[str]) -> list[float]:
        return [SequenceMatcher(None, x.lower(), y.lower()).ratio() for x, y in zip(a, b, strict=True)]


class StubRef:
    """2 nats per whitespace token; texts containing 'zzz' are 'unreadable' (8 nats/token)."""

    def __init__(self):
        self.calls = 0

    def score(self, texts: list[str]) -> list[TextScore]:
        self.calls += 1
        out = []
        for t in texts:
            n = len(t.split())
            per = 8.0 if "zzz" in t else 2.0
            out.append(TextScore(nll_nats=per * n, n_tokens=n))
        return out
```

- [ ] **Step 2: Write failing tests** `tests/test_rewards.py`

```python
import math

import pytest

from heet.config import RewardConfig
from heet.rewards.combined import Scorer
from heet.rewards.fidelity import chrf, exact_match, normalize
from heet.rewards.length import bits_ratio, char_ratio
from heet.rewards.readability import ppl_log_ratio, readability_penalty
from heet.rewards.ref_lm import TextScore
from tests.stubs import StubEmbedder, StubRef


def test_textscore_bits_and_ppl():
    s = TextScore(nll_nats=4 * math.log(2), n_tokens=2)
    assert math.isclose(s.bits, 4.0)
    assert math.isclose(s.ppl, 4.0)
    assert TextScore(0.0, 0).ppl == 1.0 and TextScore(0.0, 0).bits == 0.0


def test_normalize_and_exact():
    assert normalize("  The Cat  sat. ") == "the cat sat"
    assert exact_match("The cat sat.", "the cat sat") == 1.0
    assert exact_match("The cat sat.", "the dog sat") == 0.0


def test_chrf_range():
    assert math.isclose(chrf("the cat sat", "the cat sat"), 1.0)
    assert 0.0 <= chrf("xyz", "the cat sat") < 0.3


def test_length_ratios():
    assert char_ratio("abc", "abcdef") == 0.5
    assert char_ratio("abc", "") == 0.0
    assert bits_ratio(TextScore(math.log(2), 1), TextScore(2 * math.log(2), 2)) == 0.5
    assert bits_ratio(TextScore(1.0, 1), TextScore(0.0, 0)) == 0.0


def test_readability_margin():
    orig = TextScore(nll_nats=2.0 * 10, n_tokens=10)       # ppl e^2
    same = TextScore(nll_nats=2.0 * 5, n_tokens=5)          # ppl e^2
    worse = TextScore(nll_nats=4.0 * 5, n_tokens=5)         # ppl e^4
    assert ppl_log_ratio(same, orig) == 0.0
    assert readability_penalty(same, orig, margin=math.log(2)) == 0.0
    assert math.isclose(readability_penalty(worse, orig, margin=math.log(2)), 2.0 - math.log(2))


def test_combined_total_matches_formula():
    cfg = RewardConfig(w_chrf=0.5)
    s = Scorer(cfg, StubEmbedder(), StubRef())
    [b] = s.score(["the big cat sat on the mat"], ["big cat sat on mat"], ["the big cat sat on the mat"])
    assert b.sim == 1.0 and b.exact == 1.0 and math.isclose(b.chrf, 1.0)
    expected = (
        cfg.w_sim * b.sim + cfg.w_exact * b.exact + cfg.w_chrf * b.chrf
        - cfg.w_char * b.char_ratio - cfg.w_bits * b.bits_ratio - cfg.w_read * b.read_pen
    )
    assert math.isclose(b.total, expected)
    assert math.isclose(b.bits_ratio, 5 / 7)
    assert b.read_pen == 0.0


def test_unreadable_dense_is_penalized():
    s = Scorer(RewardConfig(), StubEmbedder(), StubRef())
    good, bad = s.score(["the cat sat"] * 2, ["cat sat", "zzz qq"], ["the cat sat"] * 2)
    assert bad.read_pen > 0 and bad.total < good.total


def test_ref_skipped_when_weights_zero_and_not_compute_all():
    ref = StubRef()
    s = Scorer(RewardConfig(w_bits=0.0, w_read=0.0), StubEmbedder(), ref)
    s.score(["a b"], ["a"], ["a b"], compute_all=False)
    assert ref.calls == 0
    s.score(["a b"], ["a"], ["a b"], compute_all=True)
    assert ref.calls == 1


def test_score_handles_empty_strings():
    s = Scorer(RewardConfig(), StubEmbedder(), StubRef())
    [b] = s.score(["the cat sat"], [""], [""])
    assert b.char_ratio == 0.0 and b.ppl_dense == 1.0
    assert all(math.isfinite(v) for v in b.to_dict().values())


def test_scorer_requires_ref_for_bits():
    with pytest.raises(ValueError, match="reference"):
        Scorer(RewardConfig(), StubEmbedder(), None).score(["a"], ["a"], ["a"])
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_rewards.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'heet.rewards.ref_lm'`

- [ ] **Step 4: Implement components**

`src/heet/rewards/ref_lm.py` (this task: dataclass + protocol; `RefLM` added in Task 4):
```python
"""Frozen reference-LM scoring: total NLL and token counts per text."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TextScore:
    nll_nats: float
    n_tokens: int

    @property
    def bits(self) -> float:
        return self.nll_nats / math.log(2)

    @property
    def ppl(self) -> float:
        return math.exp(self.nll_nats / self.n_tokens) if self.n_tokens else 1.0


class ReferenceScorer(Protocol):
    def score(self, texts: list[str]) -> list[TextScore]: ...
```

`src/heet/rewards/fidelity.py`:
```python
"""Reconstruction-fidelity measures."""

from __future__ import annotations

import re
from typing import Protocol

from sacrebleu.metrics import CHRF

_CHRF = CHRF()


class Embedder(Protocol):
    def similarity(self, a: list[str], b: list[str]) -> list[float]: ...


def normalize(text: str) -> str:
    text = " ".join(text.lower().split())
    return re.sub(r"^[\W_]+|[\W_]+$", "", text)


def exact_match(a: str, b: str) -> float:
    return 1.0 if normalize(a) == normalize(b) else 0.0


def chrf(hyp: str, ref: str) -> float:
    return _CHRF.sentence_score(hyp, [ref]).score / 100.0
```

`src/heet/rewards/length.py`:
```python
"""Length measures: characters and reference-LM bits."""

from heet.rewards.ref_lm import TextScore


def char_ratio(dense: str, original: str) -> float:
    return len(dense) / len(original) if original else 0.0


def bits_ratio(dense: TextScore, original: TextScore) -> float:
    return dense.bits / original.bits if original.bits > 0 else 0.0
```

`src/heet/rewards/readability.py`:
```python
"""Readability penalty: how much less predictable the dense text is than the original."""

import math

from heet.rewards.ref_lm import TextScore


def ppl_log_ratio(dense: TextScore, original: TextScore) -> float:
    return math.log(dense.ppl) - math.log(original.ppl)


def readability_penalty(dense: TextScore, original: TextScore, margin: float) -> float:
    return max(0.0, ppl_log_ratio(dense, original) - margin)
```

`src/heet/rewards/combined.py`:
```python
"""Weighted combination of fidelity, length and readability terms."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from heet.config import RewardConfig
from heet.rewards.fidelity import Embedder, chrf, exact_match
from heet.rewards.length import bits_ratio, char_ratio
from heet.rewards.readability import ppl_log_ratio, readability_penalty
from heet.rewards.ref_lm import ReferenceScorer, TextScore


@dataclass
class RewardBreakdown:
    sim: float = 0.0
    exact: float = 0.0
    chrf: float = 0.0
    char_ratio: float = 0.0
    bits_ratio: float = 0.0
    bits_dense: float = 0.0
    bits_orig: float = 0.0
    ppl_dense: float = 1.0
    ppl_orig: float = 1.0
    ppl_log_ratio: float = 0.0
    read_pen: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


class Scorer:
    def __init__(self, cfg: RewardConfig, embedder: Embedder, ref: ReferenceScorer | None):
        self.cfg = cfg
        self.embedder = embedder
        self.ref = ref

    def score(
        self,
        originals: list[str],
        denses: list[str],
        recons: list[str],
        compute_all: bool = True,
    ) -> list[RewardBreakdown]:
        c = self.cfg
        n = len(originals)
        assert len(denses) == n and len(recons) == n
        sims = self.embedder.similarity(originals, recons)
        need_ref = compute_all or c.w_bits != 0 or c.w_read != 0
        if need_ref and self.ref is None:
            raise ValueError("a reference LM is required for bits/readability terms")
        if need_ref:
            ref_scores = self.ref.score(list(denses) + list(originals))
            dense_s, orig_s = ref_scores[:n], ref_scores[n:]
        else:
            dense_s = orig_s = [TextScore(0.0, 0)] * n

        out = []
        for i in range(n):
            b = RewardBreakdown(sim=float(sims[i]), exact=exact_match(originals[i], recons[i]))
            if compute_all or c.w_chrf != 0:
                b.chrf = chrf(recons[i], originals[i])
            b.char_ratio = char_ratio(denses[i], originals[i])
            if need_ref:
                b.bits_dense, b.bits_orig = dense_s[i].bits, orig_s[i].bits
                b.bits_ratio = bits_ratio(dense_s[i], orig_s[i])
                b.ppl_dense, b.ppl_orig = dense_s[i].ppl, orig_s[i].ppl
                b.ppl_log_ratio = ppl_log_ratio(dense_s[i], orig_s[i])
                b.read_pen = readability_penalty(dense_s[i], orig_s[i], c.margin)
            b.total = (
                c.w_sim * b.sim + c.w_exact * b.exact + c.w_chrf * b.chrf
                - c.w_char * b.char_ratio - c.w_bits * b.bits_ratio - c.w_read * b.read_pen
            )
            out.append(b)
        return out
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_rewards.py -v`
Expected: PASS (10 tests).

- [ ] **Step 6: Commit**

```bash
git add src/heet/rewards tests/test_rewards.py tests/stubs.py
git commit -m "feat: reward components and weighted scorer"
```

---

### Task 4: Model loading, generation, real embedder and reference LM

**Files:**
- Create: `src/heet/models/loading.py`
- Modify: `src/heet/rewards/ref_lm.py` (add `RefLM`), `src/heet/rewards/fidelity.py` (add `SentenceEmbedder`)
- Test: `tests/test_models.py` (slow), `tests/test_ref_lm.py`

**Interfaces:**
- Consumes: `ModelConfig`, `LoraConfig`, `GenerationConfig`, `prompts.render_prompt`, `prompts.clean_output`, `ref_lm.TextScore`.
- Produces:
  - `loading.resolve_device(cfg: ModelConfig) -> str`
  - `loading.load_tokenizer(name: str)` — sets `pad_token = eos_token` if missing
  - `loading.load_base(cfg: ModelConfig)` — `AutoModelForCausalLM.from_pretrained(name, dtype=...)` moved to device, `eval()`
  - `loading.new_lora(base, cfg: LoraConfig, adapter_name: str) -> PeftModel`
  - `loading.check_adapter_dir(name: str, path: Path) -> None` — raises `FileNotFoundError` naming the path and `heet sft --task <task> --run <run>`
  - `loading.load_adapters(base, adapters: dict[str, Path]) -> PeftModel` — first loaded with `PeftModel.from_pretrained(..., is_trainable=False)`, others via `load_adapter`; raises `FileNotFoundError` with a hint if a dir is missing.
  - `loading.adapter_dir(cfg: Config, run: str, name: str) -> Path` → `runs/<run>/adapters/<name>`
  - `loading.use_adapter(model, name: str)` — context manager: sets adapter `name` (inference mode), restores the previously active adapter on exit (re-enabling `requires_grad` only if it was trainable before).
  - `loading.generate(model, tokenizer, task: str, texts: list[str], gen: GenerationConfig) -> list[str]` — greedy, left padding, batches of `gen.batch_size`, decodes new tokens, `clean_output`.
  - `ref_lm.RefLM(model, tokenizer, batch_size: int = 32)` implementing `ReferenceScorer`; disables adapters if `model` is a `PeftModel`; prefixes each text with `<|endoftext|>`.
  - `fidelity.SentenceEmbedder(name: str, device: str)` implementing `Embedder`.

- [ ] **Step 1: Write failing tests**

`tests/test_ref_lm.py` (fast; uses a tiny fake causal LM so no download):
```python
import math

import torch

from heet.rewards.ref_lm import RefLM


class UniformLM(torch.nn.Module):
    """Every next-token distribution is uniform over V tokens -> nll = log V per token."""

    def __init__(self, vocab: int = 8):
        super().__init__()
        self.vocab = vocab
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None):
        b, t = input_ids.shape
        return type("Out", (), {"logits": torch.zeros(b, t, self.vocab)})()


class CharTok:
    """Each character is one token id (mod 7) + 1; id 0 is <|endoftext|>."""

    pad_token_id = 0

    def convert_tokens_to_ids(self, tok):
        assert tok == "<|endoftext|>"
        return 0

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[(ord(c) % 7) + 1 for c in t] for t in texts]}


def test_ref_lm_uniform_nll():
    ref = RefLM(UniformLM(8), CharTok(), batch_size=2)
    a, b, c = ref.score(["abc", "hello", "x"])
    assert a.n_tokens == 3 and math.isclose(a.nll_nats, 3 * math.log(8), rel_tol=1e-5)
    assert b.n_tokens == 5 and math.isclose(b.ppl, 8.0, rel_tol=1e-5)
    assert c.n_tokens == 1


def test_ref_lm_empty_text():
    [s] = RefLM(UniformLM(8), CharTok()).score([""])
    assert s.n_tokens == 0 and s.nll_nats == 0.0 and s.ppl == 1.0
```

`tests/test_models.py` (slow; downloads tiny Qwen3):
```python
import pytest
import torch

from heet.config import Config, GenerationConfig, LoraConfig, ModelConfig
from heet.models.loading import generate, load_adapters, load_base, load_tokenizer, new_lora, use_adapter
from heet.rewards.ref_lm import RefLM

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(TINY)


def test_generate_returns_one_string_per_input(tok):
    base = load_base(ModelConfig(name=TINY, dtype="float32", device="cpu"))
    out = generate(base, tok, "compress", ["Hello there.", "A second, longer sentence."],
                   GenerationConfig(batch_size=1, max_new_tokens_factor=0.0, max_new_tokens_extra=4))
    assert len(out) == 2 and all(isinstance(o, str) for o in out)


def test_two_adapters_switch_and_save(tmp_path, tok):
    base = load_base(ModelConfig(name=TINY, dtype="float32", device="cpu"))
    m = new_lora(base, LoraConfig(r=2, alpha=4, target_modules=["q_proj", "v_proj"]), "decompressor")
    m.save_pretrained(str(tmp_path / "adapters"), selected_adapters=["decompressor"])
    assert (tmp_path / "adapters" / "decompressor" / "adapter_config.json").exists()

    base2 = load_base(ModelConfig(name=TINY, dtype="float32", device="cpu"))
    m2 = new_lora(base2, LoraConfig(r=2, alpha=4, target_modules=["q_proj", "v_proj"]), "compressor")
    m2.load_adapter(str(tmp_path / "adapters" / "decompressor"), adapter_name="decompressor", is_trainable=False)
    assert m2.active_adapter == "compressor"
    with use_adapter(m2, "decompressor"):
        assert m2.active_adapter == "decompressor"
    assert m2.active_adapter == "compressor"
    assert any(p.requires_grad for n, p in m2.named_parameters() if "compressor" in n)
    assert not any(p.requires_grad for n, p in m2.named_parameters() if "decompressor" in n)


def test_load_adapters_missing_dir(tmp_path):
    base = load_base(ModelConfig(name=TINY, dtype="float32", device="cpu"))
    with pytest.raises(FileNotFoundError, match="heet sft --task compress"):
        load_adapters(base, {"compressor": tmp_path / "nope"})


def test_ref_lm_on_peft_model_matches_base(tok):
    base = load_base(ModelConfig(name=TINY, dtype="float32", device="cpu"))
    plain = RefLM(base, tok).score(["The cat sat on the mat."])[0]
    m = new_lora(base, LoraConfig(r=2, alpha=4, target_modules=["q_proj", "v_proj"]), "compressor")
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.fill_(0.5)  # make the adapter non-identity
    peft_score = RefLM(m, tok).score(["The cat sat on the mat."])[0]
    assert peft_score.n_tokens == plain.n_tokens
    assert abs(peft_score.nll_nats - plain.nll_nats) < 1e-4
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ref_lm.py -v` and `uv run pytest -m slow tests/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'RefLM'` / `No module named 'heet.models.loading'`

- [ ] **Step 3: Add `RefLM` to `src/heet/rewards/ref_lm.py`**

Append:
```python
import contextlib

import torch


class RefLM:
    """Scores texts unconditionally under the frozen base model (adapters disabled)."""

    BOS = "<|endoftext|>"

    def __init__(self, model, tokenizer, batch_size: int = 32):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.bos_id = tokenizer.convert_tokens_to_ids(self.BOS)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.bos_id

    def _context(self):
        return self.model.disable_adapter() if hasattr(self.model, "disable_adapter") else contextlib.nullcontext()

    @torch.no_grad()
    def score(self, texts: list[str]) -> list[TextScore]:
        ids = self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        out: list[TextScore] = []
        device = next(self.model.parameters()).device
        with self._context():
            for start in range(0, len(ids), self.batch_size):
                chunk = ids[start : start + self.batch_size]
                seqs = [[self.bos_id] + s for s in chunk]
                width = max(len(s) for s in seqs)
                input_ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
                mask = torch.zeros((len(seqs), width), dtype=torch.long)
                for i, s in enumerate(seqs):
                    input_ids[i, : len(s)] = torch.tensor(s)
                    mask[i, : len(s)] = 1
                input_ids, mask = input_ids.to(device), mask.to(device)
                logits = self.model(input_ids=input_ids, attention_mask=mask).logits.float()
                logp = torch.log_softmax(logits[:, :-1], dim=-1)
                target = input_ids[:, 1:]
                tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1) * mask[:, 1:]
                nll = (-tok_lp.sum(dim=1)).tolist()
                counts = mask[:, 1:].sum(dim=1).tolist()
                out.extend(TextScore(float(n), int(c)) for n, c in zip(nll, counts))
        return out
```
(Move the `import contextlib` / `import torch` lines to the top of the module with the other imports.)

- [ ] **Step 4: Add `SentenceEmbedder` to `src/heet/rewards/fidelity.py`**

```python
class SentenceEmbedder:
    def __init__(self, name: str, device: str = "cpu", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, device=device)
        self.batch_size = batch_size

    def similarity(self, a: list[str], b: list[str]) -> list[float]:
        if not a:
            return []
        ea = self.model.encode(a, batch_size=self.batch_size, normalize_embeddings=True, convert_to_tensor=True)
        eb = self.model.encode(b, batch_size=self.batch_size, normalize_embeddings=True, convert_to_tensor=True)
        return (ea * eb).sum(dim=-1).clamp(-1.0, 1.0).tolist()
```

- [ ] **Step 5: Implement** `src/heet/models/loading.py`

```python
"""Base model, named LoRA adapters and batched generation."""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from heet.config import Config, GenerationConfig, LoraConfig, ModelConfig
from heet.prompts import clean_output, render_prompt

_TASK_FOR_ADAPTER = {"compressor": "compress", "decompressor": "decompress"}


def resolve_device(cfg: ModelConfig) -> str:
    if cfg.device != "auto":
        return cfg.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(cfg: ModelConfig):
    model = AutoModelForCausalLM.from_pretrained(cfg.name, dtype=getattr(torch, cfg.dtype))
    return model.to(resolve_device(cfg)).eval()


def new_lora(base, cfg: LoraConfig, adapter_name: str) -> PeftModel:
    peft_cfg = PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base, peft_cfg, adapter_name=adapter_name)


def adapter_dir(cfg: Config, run: str, name: str) -> Path:
    return Path(cfg.paths.runs_dir) / run / "adapters" / name


def check_adapter_dir(name: str, path: Path) -> None:
    if not (Path(path) / "adapter_config.json").exists():
        task = _TASK_FOR_ADAPTER.get(name, name)
        raise FileNotFoundError(
            f"adapter '{name}' not found at {path}; train it with `heet sft --task {task} --run <run>`"
        )


def load_adapters(base, adapters: dict[str, Path]) -> PeftModel:
    items = list(adapters.items())
    for name, path in items:
        check_adapter_dir(name, path)
    first_name, first_path = items[0]
    model = PeftModel.from_pretrained(base, str(first_path), adapter_name=first_name, is_trainable=False)
    for name, path in items[1:]:
        model.load_adapter(str(path), adapter_name=name, is_trainable=False)
    return model.eval()


@contextlib.contextmanager
def use_adapter(model: PeftModel, name: str):
    previous = model.active_adapter
    was_trainable = any(p.requires_grad for n, p in model.named_parameters() if f".{previous}." in n)
    model.set_adapter(name, inference_mode=True)
    try:
        yield model
    finally:
        model.set_adapter(previous, inference_mode=not was_trainable)


@torch.no_grad()
def generate(model, tokenizer, task: str, texts: list[str], gen: GenerationConfig) -> list[str]:
    was_training = model.training
    model.eval()
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    outputs: list[str] = []
    try:
        for start in range(0, len(texts), gen.batch_size):
            batch = texts[start : start + gen.batch_size]
            prompts = [render_prompt(tokenizer, task, t) for t in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            longest_input = max(len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in batch)
            max_new = int(gen.max_new_tokens_factor * longest_input) + gen.max_new_tokens_extra
            out = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            for seq in tokenizer.batch_decode(new_tokens, skip_special_tokens=True):
                outputs.append(clean_output(seq))
    finally:
        tokenizer.padding_side = old_side
        if was_training:
            model.train()
    return outputs
```

Note on `use_adapter`: peft names LoRA params like `...q_proj.lora_A.compressor.weight`, hence the `f".{previous}."` match.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_ref_lm.py -v` then `uv run pytest -m slow tests/test_models.py -v`
Expected: all PASS. If `model.generate` warns about `temperature`/`top_p` being set with `do_sample=False` (Qwen3 generation_config), that is harmless.

- [ ] **Step 7: Commit**

```bash
git add src/heet/models src/heet/rewards tests/test_ref_lm.py tests/test_models.py
git commit -m "feat: model/adapter loading, batched generation, reference LM and embedder"
```

---

### Task 5: Stage 1 — corpus chunking, Ollama client, generation pipeline

**Files:**
- Create: `src/heet/data/corpus.py`, `src/heet/data/llm_client.py`, `src/heet/data/generate.py`
- Test: `tests/test_corpus.py`, `tests/test_generate.py`

**Interfaces:**
- Consumes: `CorpusConfig`, `TeacherConfig`, `FilterConfig`, `Config`, `jsonl.*`, `prompts.teacher_rewrite_messages`, `prompts.teacher_reconstruct_messages`, `prompts.clean_output`, `fidelity.Embedder`, `fidelity.SentenceEmbedder`.
- Produces:
  - `corpus.clean_wikitext(line: str) -> str`
  - `corpus.split_sentences(paragraph: str) -> list[str]`
  - `corpus.chunk_paragraph(paragraph: str, cfg: CorpusConfig) -> list[str]`
  - `corpus.assign_split(chunk_id: str, cfg: CorpusConfig) -> str` (`"train" | "val" | "test"`)
  - `corpus.chunk_id(text: str) -> str` (`sha1[:16]`)
  - `corpus.iter_paragraphs(cfg: CorpusConfig) -> Iterator[str]`
  - `corpus.build_chunks(cfg: CorpusConfig, out_path: Path) -> int`
  - `llm_client.ChatClient` (Protocol): `async chat(messages: list[dict], temperature: float) -> str`
  - `llm_client.OllamaClient(cfg: TeacherConfig)` with `async chat(...)`, `async preflight() -> None`, `async aclose()`
  - `llm_client.TeacherUnavailable(RuntimeError)`
  - `generate.cheap_checks(original: str, dense: str, cfg: FilterConfig) -> str | None` (reason or None)
  - `generate.process_chunk(chunk: dict, client: ChatClient, embedder: Embedder, cfg: Config) -> tuple[dict, str | None]` (record, reject reason)
  - `async generate.generate_pairs(chunks: list[dict], client, embedder, cfg: Config, limit: int | None) -> dict[str, int]` (counts `accepted`, `rejected`, `skipped`)
  - `generate.run_gen_data(cfg: Config, limit: int | None, rechunk: bool) -> None`
  - Files: `data/chunks.jsonl` `{id, text, split}`; `data/pairs.jsonl` `{id, split, original, dense, teacher_recon, similarity, ratio}`; `data/rejected.jsonl` same + `reason`.

- [ ] **Step 1: Write failing corpus tests** `tests/test_corpus.py`

```python
from collections import Counter

from heet.config import CorpusConfig
from heet.data.corpus import assign_split, build_chunks, chunk_id, chunk_paragraph, clean_wikitext, split_sentences
from heet.jsonl import read_jsonl


def test_clean_wikitext():
    assert clean_wikitext(" = Valkyria Chronicles = \n") == ""
    assert clean_wikitext(" It cost 1 @,@ 000 dollars , a well @-@ known fact . \n") == (
        "It cost 1,000 dollars, a well-known fact."
    )
    assert clean_wikitext(" Pi is 3 @.@ 14 ( roughly ) . ") == "Pi is 3.14 (roughly)."


def test_split_sentences_guards_abbreviations():
    s = split_sentences("Dr. Smith met Mr. Jones in the U.S. on Monday. They talked. Was it fun? Yes!")
    assert s == ["Dr. Smith met Mr. Jones in the U.S. on Monday.", "They talked.", "Was it fun?", "Yes!"]


def test_chunk_bounds():
    cfg = CorpusConfig(min_chars=40, max_chars=120, max_sentences=3)
    para = " ".join(f"Sentence number {i} is about twenty-five chars." for i in range(10))
    chunks = chunk_paragraph(para, cfg)
    assert chunks
    for c in chunks:
        assert 40 <= len(c) <= 120
        assert 1 <= len(split_sentences(c)) <= 3


def test_long_sentence_dropped():
    cfg = CorpusConfig(min_chars=40, max_chars=60)
    assert chunk_paragraph("x" * 100 + ".", cfg) == []


def test_split_deterministic_and_proportional():
    cfg = CorpusConfig(val_frac=0.1, test_frac=0.1)
    ids = [chunk_id(f"text {i}") for i in range(5000)]
    assert [assign_split(i, cfg) for i in ids] == [assign_split(i, cfg) for i in ids]
    c = Counter(assign_split(i, cfg) for i in ids)
    assert 350 < c["val"] < 650 and 350 < c["test"] < 650


def test_build_chunks_from_text_file(tmp_path):
    src = tmp_path / "corpus.txt"
    src.write_text(
        "The first paragraph has a sentence. And another sentence follows it here.\n\n"
        "Second paragraph sentence one is here. Sentence two is also here today.\n"
    )
    cfg = CorpusConfig(source=str(src), min_chars=20, max_chars=300, max_chunks=10)
    n = build_chunks(cfg, tmp_path / "chunks.jsonl")
    rows = read_jsonl(tmp_path / "chunks.jsonl")
    assert n == len(rows) == 2
    assert {"id", "text", "split"} <= rows[0].keys()
    assert len({r["id"] for r in rows}) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'heet.data.corpus'`

- [ ] **Step 3: Implement** `src/heet/data/corpus.py`

```python
"""Corpus → cleaned 1-3 sentence chunks with deterministic splits."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

from heet.config import CorpusConfig
from heet.jsonl import write_jsonl

_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "inc", "ltd", "co",
    "gen", "col", "lt", "sgt", "capt", "gov", "sen", "rep", "no", "vol", "fig", "jan", "feb",
    "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "e.g", "i.e", "u.s",
    "u.k", "a.m", "p.m",
}
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def clean_wikitext(line: str) -> str:
    s = line.strip()
    if not s or (s.startswith("=") and s.endswith("=")):
        return ""
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r"\s+([.,;:!?%)\]])", r"\1", s)
    s = re.sub(r"([(\[])\s+", r"\1", s)
    s = re.sub(r"\s+('s|n't|'re|'ve|'ll|'d)\b", r"\1", s)
    return " ".join(s.split())


def split_sentences(paragraph: str) -> list[str]:
    parts: list[str] = []
    buf = ""
    last = 0
    for m in _SENT_END.finditer(paragraph):
        candidate = paragraph[last : m.start()].strip()
        last_word = candidate.rsplit(" ", 1)[-1].rstrip(".").lower()
        buf = f"{buf} {candidate}".strip() if buf else candidate
        last = m.end()
        if last_word in _ABBREV or (len(last_word) == 1 and last_word.isalpha()):
            continue
        parts.append(buf)
        buf = ""
    tail = paragraph[last:].strip()
    if tail:
        buf = f"{buf} {tail}".strip() if buf else tail
    if buf:
        parts.append(buf)
    return parts


def chunk_paragraph(paragraph: str, cfg: CorpusConfig) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []

    def flush():
        if cur:
            text = " ".join(cur)
            if cfg.min_chars <= len(text) <= cfg.max_chars:
                chunks.append(text)
            cur.clear()

    for sent in split_sentences(paragraph):
        if len(sent) > cfg.max_chars:
            flush()
            continue
        if cur and (len(cur) >= cfg.max_sentences or len(" ".join(cur + [sent])) > cfg.max_chars):
            flush()
        cur.append(sent)
        if len(" ".join(cur)) >= cfg.min_chars and len(cur) >= cfg.max_sentences:
            flush()
    flush()
    return chunks


def chunk_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def assign_split(cid: str, cfg: CorpusConfig) -> str:
    u = int(cid[:8], 16) / 0xFFFFFFFF
    if u < cfg.test_frac:
        return "test"
    if u < cfg.test_frac + cfg.val_frac:
        return "val"
    return "train"


def iter_paragraphs(cfg: CorpusConfig) -> Iterator[str]:
    if cfg.source == "wikitext":
        from datasets import load_dataset

        ds = load_dataset(cfg.hf_dataset, cfg.hf_config, split=cfg.hf_split)
        for row in ds:
            text = clean_wikitext(row["text"])
            if text:
                yield text
        return
    raw = Path(cfg.source).read_text(encoding="utf-8")
    for block in re.split(r"\n\s*\n", raw):
        text = clean_wikitext(" ".join(block.split()))
        if text:
            yield text


def build_chunks(cfg: CorpusConfig, out_path: Path) -> int:
    seen: set[str] = set()
    rows: list[dict] = []
    for para in iter_paragraphs(cfg):
        for text in chunk_paragraph(para, cfg):
            cid = chunk_id(text)
            if cid in seen:
                continue
            seen.add(cid)
            rows.append({"id": cid, "text": text, "split": assign_split(cid, cfg)})
            if len(rows) >= cfg.max_chunks:
                write_jsonl(out_path, rows)
                return len(rows)
    write_jsonl(out_path, rows)
    return len(rows)
```

Run: `uv run pytest tests/test_corpus.py -v` → PASS. Fix the splitter against the test cases if needed (the tests are the contract).

- [ ] **Step 4: Write failing generation tests** `tests/test_generate.py`

```python
import asyncio
import json

import httpx
import pytest

from heet.config import Config, TeacherConfig
from heet.data.generate import cheap_checks, generate_pairs, process_chunk
from heet.data.llm_client import OllamaClient, TeacherUnavailable
from heet.jsonl import read_jsonl
from tests.stubs import StubEmbedder

ORIG = "The government announced that the bridge, costing 40 million dollars, would open in March 2021."


class FakeClient:
    """Returns scripted rewrites; reconstructions echo a mapping or the original."""

    def __init__(self, rewrite: str, recon: str | None = None, fail: bool = False):
        self.rewrite, self.recon, self.fail = rewrite, recon, fail
        self.calls: list[str] = []

    async def chat(self, messages, temperature):
        if self.fail:
            raise httpx.ConnectError("boom")
        last = messages[-1]["content"]
        if last.startswith("Text: "):
            self.calls.append("rewrite")
            return self.rewrite
        self.calls.append("recon")
        return self.recon if self.recon is not None else ORIG


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.paths.data_dir = str(tmp_path)
    c.filter.min_similarity = 0.9
    return c


def test_cheap_checks(cfg):
    f = cfg.filter
    assert cheap_checks(ORIG, "Gov: bridge ($40M) opens March 2021.", f) is None
    assert cheap_checks(ORIG, "", f) == "empty"
    assert cheap_checks(ORIG, "Gov: bridge opens March 2021, about 40M.", f) is None
    assert cheap_checks(ORIG, "Gov: bridge (big cost) opens in March.", f) == "missing_number"
    assert cheap_checks(ORIG, "x 40 2021", f) == "ratio_low"
    assert cheap_checks(ORIG, ORIG.replace("The ", ""), f) in {"ratio_high", "near_copy"}


def test_process_chunk_accept(cfg):
    chunk = {"id": "a", "text": ORIG, "split": "train"}
    rec, reason = asyncio.run(
        process_chunk(chunk, FakeClient("Dense: Gov: bridge ($40M) opens March 2021."), StubEmbedder(), cfg)
    )
    assert reason is None
    assert rec["dense"] == "Gov: bridge ($40M) opens March 2021."
    assert rec["similarity"] == 1.0 and 0.3 <= rec["ratio"] <= 0.8


def test_process_chunk_low_similarity(cfg):
    chunk = {"id": "a", "text": ORIG, "split": "train"}
    rec, reason = asyncio.run(
        process_chunk(chunk, FakeClient("Gov: bridge ($40M) opens March 2021.", recon="Cats like milk."),
                      StubEmbedder(), cfg)
    )
    assert reason == "low_similarity" and rec["teacher_recon"] == "Cats like milk."


def test_cheap_reject_skips_reconstruction(cfg):
    client = FakeClient("bridge")
    rec, reason = asyncio.run(process_chunk({"id": "a", "text": ORIG, "split": "train"}, client, StubEmbedder(), cfg))
    assert reason is not None and client.calls == ["rewrite"]


def test_clean_teacher_output(cfg):
    noisy = '```\n"Gov: bridge ($40M) opens March 2021."\n```\n\nNote: removed articles.'
    rec, reason = asyncio.run(
        process_chunk({"id": "a", "text": ORIG, "split": "train"}, FakeClient(noisy), StubEmbedder(), cfg)
    )
    assert reason is None and rec["dense"] == "Gov: bridge ($40M) opens March 2021."


def test_teacher_error_rejected(cfg):
    rec, reason = asyncio.run(
        process_chunk({"id": "a", "text": ORIG, "split": "train"}, FakeClient("", fail=True), StubEmbedder(), cfg)
    )
    assert reason == "teacher_error"


def _chunks(n):
    return [{"id": f"c{i}", "text": ORIG, "split": "train"} for i in range(n)]


def test_generate_pairs_writes_and_resumes(cfg, tmp_path):
    client = FakeClient("Gov: bridge ($40M) opens March 2021.")
    counts = asyncio.run(generate_pairs(_chunks(3), client, StubEmbedder(), cfg, limit=2))
    assert counts == {"accepted": 2, "rejected": 0, "skipped": 0}
    counts = asyncio.run(generate_pairs(_chunks(3), client, StubEmbedder(), cfg, limit=None))
    assert counts == {"accepted": 1, "rejected": 0, "skipped": 2}
    assert [r["id"] for r in read_jsonl(tmp_path / "pairs.jsonl")] == ["c0", "c1", "c2"]


def test_resume_tolerates_truncated_line(cfg, tmp_path):
    good = {"id": "c0", "split": "train", "original": ORIG, "dense": "d", "teacher_recon": ORIG,
            "similarity": 1.0, "ratio": 0.5}
    (tmp_path / "pairs.jsonl").write_text(json.dumps(good) + "\n" + '{"id": "c1", "spl')
    client = FakeClient("Gov: bridge ($40M) opens March 2021.")
    counts = asyncio.run(generate_pairs(_chunks(2), client, StubEmbedder(), cfg, limit=None))
    assert counts["skipped"] == 1 and counts["accepted"] == 1
    assert [r["id"] for r in read_jsonl(tmp_path / "pairs.jsonl")] == ["c0", "c1"]


def test_rejected_written_with_reason(cfg, tmp_path):
    asyncio.run(generate_pairs(_chunks(1), FakeClient("bridge"), StubEmbedder(), cfg, limit=None))
    [r] = read_jsonl(tmp_path / "rejected.jsonl")
    assert r["id"] == "c0" and r["reason"] in {"ratio_low", "missing_number"}


def test_preflight_unreachable():
    client = OllamaClient(TeacherConfig(base_url="http://127.0.0.1:9", timeout_s=2))
    with pytest.raises(TeacherUnavailable, match="127.0.0.1:9"):
        asyncio.run(client.preflight())


def test_preflight_model_missing():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})

    client = OllamaClient(TeacherConfig(model="qwen3:14b"), transport=httpx.MockTransport(handler))
    with pytest.raises(TeacherUnavailable, match="ollama pull qwen3:14b"):
        asyncio.run(client.preflight())


def test_chat_sends_think_false():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})

    client = OllamaClient(TeacherConfig(), transport=httpx.MockTransport(handler))
    assert asyncio.run(client.chat([{"role": "user", "content": "hi"}], temperature=0.3)) == "ok"
    assert seen["think"] is False and seen["stream"] is False
    assert seen["options"]["temperature"] == 0.3
```

- [ ] **Step 5: Run to verify failure**

Run: `uv run pytest tests/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'heet.data.generate'`

- [ ] **Step 6: Implement** `src/heet/data/llm_client.py`

```python
"""Minimal async Ollama /api/chat client."""

from __future__ import annotations

import asyncio
from typing import Protocol

import httpx

from heet.config import TeacherConfig


class TeacherUnavailable(RuntimeError):
    pass


class ChatClient(Protocol):
    async def chat(self, messages: list[dict], temperature: float) -> str: ...


class OllamaClient:
    def __init__(self, cfg: TeacherConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self._client = httpx.AsyncClient(base_url=cfg.base_url, timeout=cfg.timeout_s, transport=transport)

    async def preflight(self) -> None:
        try:
            r = await self._client.get("/api/tags")
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise TeacherUnavailable(
                f"cannot reach Ollama at {self.cfg.base_url} ({e!r}); start it with `ollama serve`"
            ) from e
        names = {m.get("name") for m in r.json().get("models", [])} | {
            m.get("model") for m in r.json().get("models", [])
        }
        wanted = self.cfg.model if ":" in self.cfg.model else f"{self.cfg.model}:latest"
        if wanted not in names:
            raise TeacherUnavailable(
                f"model {self.cfg.model!r} not available at {self.cfg.base_url}; run `ollama pull {self.cfg.model}`"
            )

    async def chat(self, messages: list[dict], temperature: float) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature},
        }
        delay = 1.0
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = await self._client.post("/api/chat", json=payload)
                r.raise_for_status()
                return r.json()["message"]["content"]
            except (httpx.HTTPError, KeyError):
                if attempt == self.cfg.max_retries:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 7: Implement** `src/heet/data/generate.py`

```python
"""Stage 1: teacher rewrites → blind reconstruction → fidelity filter → pairs.jsonl."""

from __future__ import annotations

import asyncio
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path

import httpx

from heet.config import Config, FilterConfig
from heet.data.corpus import build_chunks
from heet.data.llm_client import ChatClient, OllamaClient
from heet.jsonl import append_jsonl, read_jsonl
from heet.prompts import clean_output, teacher_reconstruct_messages, teacher_rewrite_messages
from heet.rewards.fidelity import Embedder, normalize

log = logging.getLogger(__name__)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def cheap_checks(original: str, dense: str, cfg: FilterConfig) -> str | None:
    if not dense:
        return "empty"
    if "<think>" in dense:
        return "think_tag"
    ratio = len(dense) / len(original)
    if ratio < cfg.min_ratio:
        return "ratio_low"
    if ratio > cfg.max_ratio:
        return "ratio_high"
    for num in _NUMBER.findall(original):
        if num not in dense:
            return "missing_number"
    if SequenceMatcher(None, normalize(original), normalize(dense)).ratio() >= cfg.near_copy:
        return "near_copy"
    return None


async def process_chunk(chunk: dict, client: ChatClient, embedder: Embedder, cfg: Config):
    original = chunk["text"]
    rec = {
        "id": chunk["id"], "split": chunk["split"], "original": original,
        "dense": None, "teacher_recon": None, "similarity": None, "ratio": None,
    }
    try:
        dense = clean_output(await client.chat(teacher_rewrite_messages(original), cfg.teacher.rewrite_temperature))
        rec["dense"] = dense
        rec["ratio"] = len(dense) / len(original)
        reason = cheap_checks(original, dense, cfg.filter)
        if reason:
            return rec, reason
        recon = clean_output(
            await client.chat(teacher_reconstruct_messages(dense), cfg.teacher.recon_temperature)
        )
    except (httpx.HTTPError, KeyError, ValueError) as e:
        log.warning("teacher error on %s: %r", chunk["id"], e)
        return rec, "teacher_error"
    rec["teacher_recon"] = recon
    rec["similarity"] = float(embedder.similarity([original], [recon])[0])
    if rec["similarity"] < cfg.filter.min_similarity:
        return rec, "low_similarity"
    return rec, None


async def generate_pairs(chunks, client: ChatClient, embedder: Embedder, cfg: Config, limit: int | None):
    data = Path(cfg.paths.data_dir)
    pairs_path, rejected_path = data / "pairs.jsonl", data / "rejected.jsonl"
    done = {r["id"] for r in read_jsonl(pairs_path)} | {r["id"] for r in read_jsonl(rejected_path)}
    todo = [c for c in chunks if c["id"] not in done]
    skipped = len(chunks) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    counts = {"accepted": 0, "rejected": 0, "skipped": skipped}
    sem = asyncio.Semaphore(cfg.teacher.concurrency)

    async def one(chunk):
        async with sem:
            rec, reason = await process_chunk(chunk, client, embedder, cfg)
        if reason is None:
            append_jsonl(pairs_path, rec)
            counts["accepted"] += 1
        else:
            append_jsonl(rejected_path, {**rec, "reason": reason})
            counts["rejected"] += 1
        done_n = counts["accepted"] + counts["rejected"]
        if done_n % 100 == 0:
            log.info("processed %d/%d (accepted %d)", done_n, len(todo), counts["accepted"])

    await asyncio.gather(*(one(c) for c in todo))
    return counts


def run_gen_data(cfg: Config, limit: int | None, rechunk: bool) -> None:
    from heet.models.loading import resolve_device
    from heet.rewards.fidelity import SentenceEmbedder

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    chunks_path = Path(cfg.paths.data_dir) / "chunks.jsonl"
    if rechunk or not chunks_path.exists():
        n = build_chunks(cfg.corpus, chunks_path)
        log.info("wrote %d chunks to %s", n, chunks_path)
    chunks = read_jsonl(chunks_path)
    embedder = SentenceEmbedder(cfg.reward.embedder, device=resolve_device(cfg.model))

    async def main():
        client = OllamaClient(cfg.teacher)
        try:
            await client.preflight()
            return await generate_pairs(chunks, client, embedder, cfg, limit)
        finally:
            await client.aclose()

    counts = asyncio.run(main())
    log.info("done: %s", counts)
```

Note: `test_cheap_checks` expects `"x 40 2021"` → `ratio_low` (checked before numbers) and `"Gov: bridge (big cost) opens in March."` → `missing_number`; the check order above is the contract.

- [ ] **Step 8: Run tests**

Run: `uv run pytest tests/test_corpus.py tests/test_generate.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/heet/data tests/test_corpus.py tests/test_generate.py
git commit -m "feat: stage 1 corpus chunking, Ollama teacher client and filtered pair generation"
```

---

### Task 6: Metrics logger and sample writer

**Files:**
- Create: `src/heet/eval/logger.py`, `src/heet/eval/samples.py`
- Test: `tests/test_logging.py`

**Interfaces:**
- Consumes: `jsonl.append_jsonl`, `jsonl.write_jsonl`.
- Produces:
  - `logger.MetricsLogger(run_dir: Path)` with `log(step: int, metrics: dict[str, float | bool], prefix: str = "", **meta) -> None` (appends `{step, **meta, **prefixed metrics}` to `run_dir/metrics.jsonl`, writes numeric scalars to TensorBoard under `run_dir/tb`) and `close()`.
  - `samples.write_samples(run_dir: Path, step: int, records: list[dict], n_samples: int, worst_k: int, tag: str = "") -> Path` — keeps the first `n_samples` records in input order (fixed ids since inputs are fixed), plus the `worst_k` lowest `total` not already included, each with `"worst": bool`; writes `run_dir/samples/{tag}step_{step:06d}.jsonl`.

- [ ] **Step 1: Write failing tests** `tests/test_logging.py`

```python
from heet.eval.logger import MetricsLogger
from heet.eval.samples import write_samples
from heet.jsonl import read_jsonl


def test_metrics_logger_jsonl_and_tb(tmp_path):
    lg = MetricsLogger(tmp_path)
    lg.log(10, {"sim": 0.9, "pass": True}, prefix="student/", split="val")
    lg.log(20, {"sim": 0.95}, prefix="teacher/", split="val")
    lg.close()
    rows = read_jsonl(tmp_path / "metrics.jsonl")
    assert rows[0] == {"step": 10, "split": "val", "student/sim": 0.9, "student/pass": True}
    assert rows[1]["teacher/sim"] == 0.95
    assert any((tmp_path / "tb").iterdir())


def test_write_samples_fixed_plus_worst(tmp_path):
    recs = [{"id": str(i), "total": float(i)} for i in range(20)]
    recs[15]["total"] = -5.0
    p = write_samples(tmp_path, 7, recs, n_samples=5, worst_k=2, tag="val_")
    assert p.name == "val_step_000007.jsonl"
    rows = read_jsonl(p)
    assert [r["id"] for r in rows[:5]] == ["0", "1", "2", "3", "4"]
    worst = [r["id"] for r in rows if r["worst"]]
    assert worst == ["15", "5"]
```

Note `"0"` has total 0.0 (would be second worst) but is already in the fixed set, so the next worst `"5"` is used.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_logging.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'heet.eval.logger'`

- [ ] **Step 3: Implement**

`src/heet/eval/logger.py`:
```python
"""Metrics → JSONL + TensorBoard."""

from __future__ import annotations

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from heet.jsonl import append_jsonl


class MetricsLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))

    def log(self, step: int, metrics: dict, prefix: str = "", **meta) -> None:
        row = {"step": step, **meta, **{f"{prefix}{k}": v for k, v in metrics.items()}}
        append_jsonl(self.run_dir / "metrics.jsonl", row)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._tb.add_scalar(f"{prefix}{k}", float(v), step)
        self._tb.flush()

    def close(self) -> None:
        self._tb.close()
```

`src/heet/eval/samples.py`:
```python
"""Dump (input, dense, reconstruction) triples for manual inspection."""

from __future__ import annotations

from pathlib import Path

from heet.jsonl import write_jsonl


def write_samples(run_dir: Path, step: int, records: list[dict], n_samples: int, worst_k: int, tag: str = "") -> Path:
    fixed = [{**r, "worst": False} for r in records[:n_samples]]
    tail = sorted(records[n_samples:], key=lambda r: r["total"])[:worst_k]
    worst = [{**r, "worst": True} for r in tail]
    path = Path(run_dir) / "samples" / f"{tag}step_{step:06d}.jsonl"
    write_jsonl(path, fixed + worst)
    return path
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_logging.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/heet/eval/logger.py src/heet/eval/samples.py tests/test_logging.py
git commit -m "feat: JSONL/TensorBoard metrics logger and sample dumps"
```

---

### Task 7: Evaluation (aggregation, round trip, teacher baseline, `heet eval`)

**Files:**
- Create: `src/heet/eval/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `Scorer`, `RewardBreakdown`, `MetricsLogger`, `write_samples`, `loading.{load_tokenizer, load_base, load_adapters, adapter_dir, use_adapter, generate, resolve_device}`, `RefLM`, `SentenceEmbedder`, `read_jsonl`, `EvalConfig`, `RewardConfig`.
- Produces:
  - `evaluate.aggregate(breakdowns: list[RewardBreakdown], token_ratios: list[float], orig_chars: list[int], cfg: Config) -> dict[str, float | bool]` with keys: `char_ratio, token_ratio, bpc_dense, bpc_orig, bits_ratio, sim, sim_p10, exact_rate, chrf, ppl_dense, ppl_orig, ppl_log_ratio, reward, pass, n`.
    - `bpc_dense = Σ bits_dense / Σ orig_chars`, `bpc_orig = Σ bits_orig / Σ orig_chars`.
    - `ppl_dense`/`ppl_orig` are means; `ppl_log_ratio` is the mean of per-item log ratios.
    - `pass = char_ratio <= eval.target_char_ratio and sim >= eval.target_sim and ppl_log_ratio <= reward.margin`.
  - `evaluate._mean(xs) -> float`, `evaluate._p10(xs) -> float` (also used by Task 8)
  - `evaluate.build_scorer(model, tokenizer, cfg) -> Scorer` (SentenceEmbedder + RefLM)
  - `evaluate.score_triples(originals, denses, recons, tokenizer, scorer, cfg) -> tuple[dict, list[dict]]` — returns (aggregate metrics, per-item records `{original, dense, recon, **breakdown}`).
  - `evaluate.round_trip(model, tokenizer, originals: list[str], cfg) -> tuple[list[str], list[str]]` — requires adapters `compressor` and `decompressor` loaded on `model`; returns (denses, recons).
  - `evaluate.load_split_pairs(cfg, split: str, limit: int | None) -> list[dict]`
  - `evaluate.run_eval(cfg: Config, run: str, split: str, limit: int | None) -> dict` — logs `student/` and `teacher/` metrics at step `0` (standalone eval), writes `samples/{split}_student_step_...` and `{split}_teacher_step_...`, prints a summary, returns `{"student": ..., "teacher": ...}`.

- [ ] **Step 1: Write failing tests** `tests/test_evaluate.py`

```python
import math

import pytest

from heet.config import Config
from heet.eval.evaluate import aggregate, run_eval, score_triples
from heet.rewards.combined import RewardBreakdown, Scorer
from tests.stubs import StubEmbedder, StubRef


class WsTok:
    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [t.split() for t in texts]}


def test_aggregate_values_and_pass():
    cfg = Config()
    bs = [
        RewardBreakdown(sim=1.0, char_ratio=0.5, bits_dense=10, bits_orig=20, bits_ratio=0.5,
                        ppl_dense=4, ppl_orig=2, ppl_log_ratio=math.log(2), total=1.0, exact=1.0),
        RewardBreakdown(sim=0.9, char_ratio=0.4, bits_dense=10, bits_orig=20, bits_ratio=0.5,
                        ppl_dense=2, ppl_orig=2, ppl_log_ratio=0.0, total=0.5),
    ]
    m = aggregate(bs, token_ratios=[0.5, 0.5], orig_chars=[50, 50], cfg=cfg)
    assert math.isclose(m["sim"], 0.95) and math.isclose(m["char_ratio"], 0.45)
    assert math.isclose(m["bpc_dense"], 0.2) and math.isclose(m["bpc_orig"], 0.4)
    assert m["exact_rate"] == 0.5 and m["n"] == 2
    assert m["pass"] is True
    cfg.eval.target_char_ratio = 0.3
    assert aggregate(bs, [0.5, 0.5], [50, 50], cfg)["pass"] is False


def test_score_triples_records():
    cfg = Config()
    scorer = Scorer(cfg.reward, StubEmbedder(), StubRef())
    metrics, recs = score_triples(["the cat sat down"], ["cat sat"], ["the cat sat down"], WsTok(), scorer, cfg)
    assert metrics["token_ratio"] == 0.5 and metrics["sim"] == 1.0
    assert recs[0]["original"] == "the cat sat down" and "total" in recs[0]


def test_eval_missing_adapter(tmp_path):
    cfg = Config()
    cfg.paths.runs_dir = str(tmp_path / "runs")
    cfg.paths.data_dir = str(tmp_path / "data")
    with pytest.raises(FileNotFoundError, match="heet sft --task"):
        run_eval(cfg, run="r1", split="test", limit=2)
```

`run_eval` must check adapter dirs **before** loading any model, so `test_eval_missing_adapter` needs no download.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'heet.eval.evaluate'`

- [ ] **Step 3: Implement** `src/heet/eval/evaluate.py`

```python
"""Round-trip evaluation: original → compressor → dense → decompressor → reconstruction."""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

from heet.config import Config
from heet.eval.logger import MetricsLogger
from heet.eval.samples import write_samples
from heet.jsonl import read_jsonl
from heet.rewards.combined import RewardBreakdown, Scorer

log = logging.getLogger(__name__)


def _mean(xs) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else 0.0


def _p10(xs) -> float:
    xs = sorted(xs)
    return float(xs[int(0.1 * (len(xs) - 1))]) if xs else 0.0


def aggregate(breakdowns: list[RewardBreakdown], token_ratios: list[float], orig_chars: list[int], cfg: Config) -> dict:
    total_chars = sum(orig_chars) or 1
    m = {
        "n": len(breakdowns),
        "char_ratio": _mean(b.char_ratio for b in breakdowns),
        "token_ratio": _mean(token_ratios),
        "bpc_dense": sum(b.bits_dense for b in breakdowns) / total_chars,
        "bpc_orig": sum(b.bits_orig for b in breakdowns) / total_chars,
        "bits_ratio": _mean(b.bits_ratio for b in breakdowns),
        "sim": _mean(b.sim for b in breakdowns),
        "sim_p10": _p10([b.sim for b in breakdowns]),
        "exact_rate": _mean(b.exact for b in breakdowns),
        "chrf": _mean(b.chrf for b in breakdowns),
        "ppl_dense": _mean(b.ppl_dense for b in breakdowns),
        "ppl_orig": _mean(b.ppl_orig for b in breakdowns),
        "ppl_log_ratio": _mean(b.ppl_log_ratio for b in breakdowns),
        "reward": _mean(b.total for b in breakdowns),
    }
    m["pass"] = bool(
        m["char_ratio"] <= cfg.eval.target_char_ratio
        and m["sim"] >= cfg.eval.target_sim
        and m["ppl_log_ratio"] <= cfg.reward.margin
    )
    return m


def score_triples(originals, denses, recons, tokenizer, scorer: Scorer, cfg: Config):
    breakdowns = scorer.score(originals, denses, recons, compute_all=True)
    o_ids = tokenizer(list(originals), add_special_tokens=False)["input_ids"]
    d_ids = tokenizer(list(denses), add_special_tokens=False)["input_ids"]
    token_ratios = [len(d) / len(o) if o else 0.0 for d, o in zip(d_ids, o_ids)]
    metrics = aggregate(breakdowns, token_ratios, [len(o) for o in originals], cfg)
    records = [
        {"original": o, "dense": d, "recon": r, **b.to_dict()}
        for o, d, r, b in zip(originals, denses, recons, breakdowns)
    ]
    return metrics, records


def round_trip(model, tokenizer, originals: list[str], cfg: Config):
    from heet.models.loading import generate, use_adapter

    with use_adapter(model, "compressor"):
        denses = generate(model, tokenizer, "compress", originals, cfg.generation)
    with use_adapter(model, "decompressor"):
        recons = generate(model, tokenizer, "decompress", denses, cfg.generation)
    return denses, recons


def load_split_pairs(cfg: Config, split: str, limit: int | None) -> list[dict]:
    path = Path(cfg.paths.data_dir) / "pairs.jsonl"
    rows = [r for r in read_jsonl(path) if r["split"] == split]
    if not rows:
        raise FileNotFoundError(f"no '{split}' pairs in {path}; run `heet gen-data` first")
    return rows[:limit] if limit else rows


def build_scorer(model, tokenizer, cfg: Config) -> Scorer:
    from heet.models.loading import resolve_device
    from heet.rewards.fidelity import SentenceEmbedder
    from heet.rewards.ref_lm import RefLM

    embedder = SentenceEmbedder(cfg.reward.embedder, device=resolve_device(cfg.model))
    return Scorer(cfg.reward, embedder, RefLM(model, tokenizer, cfg.reward.ref_batch_size))


def run_eval(cfg: Config, run: str, split: str, limit: int | None) -> dict:
    from heet.models.loading import (
        adapter_dir, check_adapter_dir, generate, load_adapters, load_base, load_tokenizer, use_adapter,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dirs = {name: adapter_dir(cfg, run, name) for name in ("compressor", "decompressor")}
    for name, path in dirs.items():
        check_adapter_dir(name, path)
    pairs = load_split_pairs(cfg, split, limit)

    tokenizer = load_tokenizer(cfg.model.name)
    model = load_adapters(load_base(cfg.model), dirs)
    scorer = build_scorer(model, tokenizer, cfg)
    run_dir = Path(cfg.paths.runs_dir) / run
    logger = MetricsLogger(run_dir)

    originals = [p["original"] for p in pairs]
    denses, recons = round_trip(model, tokenizer, originals, cfg)
    student, s_recs = score_triples(originals, denses, recons, tokenizer, scorer, cfg)

    teacher_denses = [p["dense"] for p in pairs]
    with use_adapter(model, "decompressor"):
        t_recons = generate(model, tokenizer, "decompress", teacher_denses, cfg.generation)
    teacher, t_recs = score_triples(originals, teacher_denses, t_recons, tokenizer, scorer, cfg)

    for recs, src in ((s_recs, "student"), (t_recs, "teacher")):
        for rec, p in zip(recs, pairs):
            rec["id"] = p["id"]
        write_samples(run_dir, 0, recs, cfg.eval.n_samples, cfg.eval.worst_k, tag=f"{split}_{src}_")
    logger.log(0, student, prefix="student/", split=split, source="eval")
    logger.log(0, teacher, prefix="teacher/", split=split, source="eval")
    logger.close()
    summary = {"student": student, "teacher": teacher}
    print(json.dumps(summary, indent=2))
    return summary
```


- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/heet/eval/evaluate.py src/heet/models/loading.py tests/test_evaluate.py
git commit -m "feat: round-trip evaluation with teacher baseline and success gate"
```

---

### Task 8: Stage 2 — SFT with eval callback, plus end-to-end smoke test

**Files:**
- Create: `src/heet/training/sft.py`
- Test: `tests/test_sft.py`, `tests/test_smoke.py` (slow)

**Interfaces:**
- Consumes: everything above. TRL: `trl.SFTTrainer(model=PeftModel, args=trl.SFTConfig, train_dataset=datasets.Dataset, processing_class=tokenizer, callbacks=[...])`, standard (non-conversational) prompt–completion format `{"prompt": str, "completion": str}`; with that format TRL sets completion-only loss and appends `eos_token` only if missing.
- Produces:
  - `sft.TASK_ADAPTER = {"compress": "compressor", "decompress": "decompressor"}`
  - `sft.build_examples(pairs: list[dict], task: str, tokenizer) -> list[dict]` — `prompt = render_prompt(tok, task, src)`, `completion = render_completion(tok, tgt)`; compress: src=original, tgt=dense; decompress: src=dense, tgt=original.
  - `sft.EvalCallback(transformers.TrainerCallback)` — `__init__(self, cfg, task, tokenizer, pairs: list[dict], scorer: Scorer, logger: MetricsLogger, run_dir: Path, has_decompressor: bool)`; runs at `on_step_end` when `global_step % cfg.sft.eval_steps == 0` and at `on_train_end`; method `evaluate(model, step) -> dict`.
  - `sft.run_sft(cfg: Config, task: str, run: str, limit: int | None) -> Path` — returns the saved adapter dir.

- [ ] **Step 1: Write failing fast tests** `tests/test_sft.py`

```python
from pathlib import Path

from heet.config import Config
from heet.eval.logger import MetricsLogger
from heet.jsonl import read_jsonl
from heet.rewards.combined import Scorer
from heet.training.sft import EvalCallback, build_examples
from tests.stubs import StubEmbedder, StubRef
from tests.test_prompts import FakeTok

PAIRS = [{"id": "a", "split": "val", "original": "The cat sat on the mat.", "dense": "cat sat on mat"}]


def test_build_examples_directions():
    [c] = build_examples(PAIRS, "compress", FakeTok())
    assert "The cat sat on the mat." in c["prompt"] and c["completion"] == "cat sat on mat<|im_end|>"
    [d] = build_examples(PAIRS, "decompress", FakeTok())
    assert "cat sat on mat" in d["prompt"] and d["completion"] == "The cat sat on the mat.<|im_end|>"


class EchoModel:
    """Stands in for the PeftModel: generation is patched below, adapters are tracked by name."""

    active_adapter = "compressor"


def test_callback_without_decompressor(tmp_path, monkeypatch, caplog):
    import heet.training.sft as sft

    monkeypatch.setattr(sft, "generate", lambda model, tok, task, texts, gen: [t.lower()[:10] for t in texts])
    cfg = Config()
    ws = type("WsTok", (), {"__call__": lambda self, t, add_special_tokens=False: {"input_ids": [x.split() for x in t]}})()
    cb = EvalCallback(cfg, "compress", ws, PAIRS, Scorer(cfg.reward, StubEmbedder(), StubRef()),
                      MetricsLogger(tmp_path), tmp_path, has_decompressor=False)
    m1 = cb.evaluate(EchoModel(), step=5)
    cb.evaluate(EchoModel(), step=10)
    assert "char_ratio" in m1 and "ppl_log_ratio" in m1 and "sim" not in m1
    assert sum("decompressor" in r.message for r in caplog.records) == 1
    rows = read_jsonl(tmp_path / "metrics.jsonl")
    assert rows[0]["step"] == 5 and "compress/char_ratio" in rows[0]
    assert (tmp_path / "samples" / "compress_step_000005.jsonl").exists()


def test_callback_decompress_task(tmp_path, monkeypatch):
    import heet.training.sft as sft

    monkeypatch.setattr(sft, "generate", lambda model, tok, task, texts, gen: ["The cat sat on the mat."] * len(texts))
    cfg = Config()
    ws = type("WsTok", (), {"__call__": lambda self, t, add_special_tokens=False: {"input_ids": [x.split() for x in t]}})()
    cb = EvalCallback(cfg, "decompress", ws, PAIRS, Scorer(cfg.reward, StubEmbedder(), StubRef()),
                      MetricsLogger(tmp_path), tmp_path, has_decompressor=True)
    m = cb.evaluate(EchoModel(), step=1)
    assert m["sim"] == 1.0 and m["exact_rate"] == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_sft.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'heet.training.sft'`

- [ ] **Step 3: Implement** `src/heet/training/sft.py`

```python
"""Stage 2: supervised fine-tuning of one named LoRA adapter."""

from __future__ import annotations

import logging
import random
from pathlib import Path

from transformers import TrainerCallback

from heet.config import Config, dump_config
from heet.eval.evaluate import _mean, _p10, load_split_pairs, score_triples
from heet.eval.logger import MetricsLogger
from heet.eval.samples import write_samples
from heet.models.loading import generate, use_adapter
from heet.prompts import render_completion, render_prompt
from heet.rewards.combined import Scorer
from heet.rewards.fidelity import chrf, exact_match

log = logging.getLogger(__name__)
TASK_ADAPTER = {"compress": "compressor", "decompress": "decompressor"}


def build_examples(pairs: list[dict], task: str, tokenizer) -> list[dict]:
    out = []
    for p in pairs:
        src, tgt = (p["original"], p["dense"]) if task == "compress" else (p["dense"], p["original"])
        out.append({"prompt": render_prompt(tokenizer, task, src), "completion": render_completion(tokenizer, tgt)})
    return out


class EvalCallback(TrainerCallback):
    def __init__(self, cfg: Config, task: str, tokenizer, pairs: list[dict], scorer: Scorer,
                 logger: MetricsLogger, run_dir: Path, has_decompressor: bool):
        self.cfg, self.task, self.tokenizer, self.pairs = cfg, task, tokenizer, pairs
        self.scorer, self.logger, self.run_dir = scorer, logger, Path(run_dir)
        self.has_decompressor = has_decompressor
        self._warned = False

    def evaluate(self, model, step: int) -> dict:
        originals = [p["original"] for p in self.pairs]
        gen = self.cfg.generation
        if self.task == "decompress":
            recons = generate(model, self.tokenizer, "decompress", [p["dense"] for p in self.pairs], gen)
            sims = self.scorer.embedder.similarity(originals, recons)
            metrics = {
                "sim": _mean(sims),
                "sim_p10": _p10(sims),
                "exact_rate": _mean(exact_match(o, r) for o, r in zip(originals, recons)),
                "chrf": _mean(chrf(r, o) for o, r in zip(originals, recons)),
            }
            records = [{"id": p["id"], "original": o, "dense": p["dense"], "recon": r, "sim": s, "total": s}
                       for p, o, r, s in zip(self.pairs, originals, recons, sims)]
        else:
            denses = generate(model, self.tokenizer, "compress", originals, gen)
            if self.has_decompressor:
                with use_adapter(model, "decompressor"):
                    recons = generate(model, self.tokenizer, "decompress", denses, gen)
                metrics, records = score_triples(originals, denses, recons, self.tokenizer, self.scorer, self.cfg)
            else:
                if not self._warned:
                    log.warning("no decompressor adapter loaded; logging compressor-side metrics only")
                    self._warned = True
                full, records = score_triples(originals, denses, originals, self.tokenizer, self.scorer, self.cfg)
                keep = ("char_ratio", "token_ratio", "bpc_dense", "bpc_orig", "bits_ratio",
                        "ppl_dense", "ppl_orig", "ppl_log_ratio", "n")
                metrics = {k: full[k] for k in keep}
                for r in records:
                    r["recon"] = None
            for rec, p in zip(records, self.pairs):
                rec["id"] = p["id"]
        self.logger.log(step, metrics, prefix=f"{self.task}/", split="val", source="callback")
        write_samples(self.run_dir, step, records, self.cfg.eval.n_samples, self.cfg.eval.worst_k, tag=f"{self.task}_")
        return metrics

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.cfg.sft.eval_steps == 0:
            self.evaluate(model, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.cfg.sft.eval_steps != 0:
            self.evaluate(model, state.global_step)


def run_sft(cfg: Config, task: str, run: str, limit: int | None) -> Path:
    from datasets import Dataset
    from trl import SFTConfig as TrlSFTConfig
    from trl import SFTTrainer

    from heet.eval.evaluate import build_scorer
    from heet.models.loading import adapter_dir, check_adapter_dir, load_base, load_tokenizer, new_lora

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    adapter = TASK_ADAPTER[task]
    run_dir = Path(cfg.paths.runs_dir) / run
    dump_config(cfg, run_dir / f"config_{task}.yaml")

    train_pairs = load_split_pairs(cfg, "train", limit)
    val_pairs = load_split_pairs(cfg, "val", None)
    random.Random(cfg.sft.seed).shuffle(val_pairs)
    val_pairs = val_pairs[: cfg.eval.callback_n]

    tokenizer = load_tokenizer(cfg.model.name)
    model = new_lora(load_base(cfg.model), cfg.lora, adapter)

    has_decompressor = False
    if task == "compress":
        dpath = Path(cfg.sft.decompressor_path) if cfg.sft.decompressor_path else adapter_dir(cfg, run, "decompressor")
        try:
            check_adapter_dir("decompressor", dpath)
            model.load_adapter(str(dpath), adapter_name="decompressor", is_trainable=False)
            model.set_adapter(adapter)  # keep the compressor active and trainable
            has_decompressor = True
        except FileNotFoundError as e:
            log.warning("%s", e)

    logger = MetricsLogger(run_dir)
    callback = EvalCallback(cfg, task, tokenizer, val_pairs, build_scorer(model, tokenizer, cfg),
                            logger, run_dir, has_decompressor)
    s = cfg.sft
    args = TrlSFTConfig(
        output_dir=str(run_dir / f"checkpoints_{task}"),
        learning_rate=s.lr,
        num_train_epochs=s.epochs,
        per_device_train_batch_size=s.per_device_batch_size,
        gradient_accumulation_steps=s.grad_accum,
        lr_scheduler_type=s.scheduler,
        warmup_ratio=s.warmup_ratio,
        max_length=s.max_length,
        bf16=cfg.model.dtype == "bfloat16",
        gradient_checkpointing=s.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=s.logging_steps,
        save_steps=s.save_steps,
        save_total_limit=2,
        seed=s.seed,
        report_to=["tensorboard"],
        logging_dir=str(run_dir / "tb" / f"trainer_{task}"),
        completion_only_loss=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(build_examples(train_pairs, task, tokenizer)),
        processing_class=tokenizer,
        callbacks=[callback],
    )
    trainer.train()
    out_root = run_dir / "adapters"
    model.save_pretrained(str(out_root), selected_adapters=[adapter])
    tokenizer.save_pretrained(str(out_root / adapter))
    logger.close()
    log.info("saved adapter %s to %s", adapter, out_root / adapter)
    return out_root / adapter
```

Before running, verify the installed TRL accepts every `TrlSFTConfig` kwarg:
Run: `uv run python -c "import inspect, trl; p=inspect.signature(trl.SFTConfig).parameters; print([k for k in ['max_length','completion_only_loss','gradient_checkpointing_kwargs','warmup_ratio','logging_dir'] if k not in p])"`
Expected: `[]`. If a name is missing (e.g. `logging_dir` deprecated in transformers 5), drop that kwarg; do not rename others.

- [ ] **Step 4: Run fast tests**

Run: `uv run pytest tests/test_sft.py -v`
Expected: PASS.

- [ ] **Step 5: Write slow end-to-end smoke test** `tests/test_smoke.py`

```python
import pytest

from heet.config import load_config
from heet.eval.evaluate import run_eval
from heet.jsonl import write_jsonl
from heet.training.sft import run_sft

pytestmark = pytest.mark.slow

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
ORIG = "The government announced that the new bridge would open in March 2021."
DENSE = "Gov: new bridge opens March 2021."


def test_sft_then_eval_end_to_end(tmp_path):
    rows = [{"id": f"{s}{i}", "split": s, "original": ORIG, "dense": DENSE, "teacher_recon": ORIG,
             "similarity": 1.0, "ratio": 0.5} for s in ("train", "val", "test") for i in range(4)]
    write_jsonl(tmp_path / "data" / "pairs.jsonl", rows)
    cfg = load_config(None, [
        f"paths.data_dir={tmp_path / 'data'}", f"paths.runs_dir={tmp_path / 'runs'}",
        f"model.name={TINY}", "model.dtype=float32", "model.device=cpu",
        "sft.epochs=1", "sft.per_device_batch_size=2", "sft.grad_accum=1", "sft.eval_steps=1",
        "sft.gradient_checkpointing=false", "sft.max_length=64",
        "generation.max_new_tokens_factor=0.0", "generation.max_new_tokens_extra=4",
        "eval.callback_n=2", "eval.n_samples=2", "eval.worst_k=1",
        "reward.embedder=sentence-transformers/all-MiniLM-L6-v2",
    ])
    cfg.lora.target_modules = ["q_proj", "v_proj"]
    run_sft(cfg, task="decompress", run="r", limit=None)
    run_sft(cfg, task="compress", run="r", limit=None)
    out = run_eval(cfg, run="r", split="test", limit=4)

    run_dir = tmp_path / "runs" / "r"
    assert (run_dir / "adapters" / "compressor" / "adapter_config.json").exists()
    assert (run_dir / "adapters" / "decompressor" / "adapter_config.json").exists()
    assert (run_dir / "metrics.jsonl").exists() and any((run_dir / "tb").iterdir())
    assert list((run_dir / "samples").glob("compress_step_*.jsonl"))
    assert list((run_dir / "samples").glob("test_student_step_*.jsonl"))
    assert {"char_ratio", "sim", "bpc_dense", "ppl_log_ratio", "pass"} <= out["student"].keys()
```

- [ ] **Step 6: Run the smoke test**

Run: `uv run pytest -m slow tests/test_smoke.py -v -s`
Expected: PASS within a few minutes on CPU. If TRL logs "tokenized prompt is not a prefix of the tokenized prompt+completion", inspect one example with `tokenizer.decode` and fix `render_prompt`/`render_completion` so the boundary is clean (the prompt ends with `\n\n` after `</think>`, which should tokenize cleanly).

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -v` then `uv run pytest -m slow -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/heet/training tests/test_sft.py tests/test_smoke.py
git commit -m "feat: stage 2 LoRA SFT with in-training round-trip eval callback"
```

---

### Task 9: Real-hardware verification run

**Files:**
- Create: `README.md` (usage section only: the workflow from spec §11 plus where outputs land)

**Interfaces:** none new.

- [ ] **Step 1: Write `README.md`**

````markdown
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
````

- [ ] **Step 2: Small real run on the GPU** (requires `ollama pull qwen3:14b`)

Run:
```bash
uv run heet gen-data --limit 300 --set corpus.max_chunks=5000
uv run heet sft --task decompress --run smoke --set sft.epochs=1 --set sft.eval_steps=20
uv run heet sft --task compress --run smoke --set sft.epochs=1 --set sft.eval_steps=20
uv run heet eval --run smoke --split val --limit 50
```
Expected: pairs accepted (inspect `data/rejected.jsonl` reason counts with `uv run python -c "import collections,json;print(collections.Counter(json.loads(l)['reason'] for l in open('data/rejected.jsonl')))"`); both SFT runs finish without OOM (`nvidia-smi` peak < 16 GB); eval prints student and teacher metrics; `runs/smoke/samples/` contains readable triples. With only ~300 pairs, the metrics do not need to pass the success gate.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README usage"
```
