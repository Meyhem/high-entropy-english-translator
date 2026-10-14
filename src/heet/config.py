"""Typed configuration: dataclass defaults, YAML file, `--set a.b=c` overrides."""

from __future__ import annotations

import dataclasses
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
    # readability penalty fires when log(ppl_dense / ppl_orig) exceeds this;
    # 1.7 ≈ the teacher's own level on WikiText (ln 2 penalized every rewrite)
    margin: float = 1.7
    ref_batch_size: int = 32


@dataclass
class EvalConfig:
    callback_n: int = 64
    n_samples: int = 32
    worst_k: int = 10
    target_char_ratio: float = 0.55
    target_sim: float = 0.90


@dataclass
class EIConfig:
    rounds: int = 3
    n_per_round: int = 1000
    k: int = 8
    temperature: float = 0.8
    top_p: float = 0.95
    min_sim: float = 0.90
    min_gain: float = 0.02
    lr: float = 5e-5
    epochs: float = 1.0
    seed: int = 0


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
    ei: EIConfig = field(default_factory=EIConfig)


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
