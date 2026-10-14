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
    # peft names LoRA params like `...q_proj.lora_A.compressor.weight`
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
            longest_input = max(len(ids) for ids in tokenizer(batch, add_special_tokens=False)["input_ids"])
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
