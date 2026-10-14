"""Stage 2: supervised fine-tuning of one named LoRA adapter."""

from __future__ import annotations

import logging
import os
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
_COMPRESSOR_ONLY = ("char_ratio", "token_ratio", "bpc_dense", "bpc_orig", "bits_ratio",
                    "ppl_dense", "ppl_orig", "ppl_log_ratio", "n")


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
                metrics = {k: full[k] for k in _COMPRESSOR_ONLY}
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
    # transformers 5 reads the TensorBoard dir for trainer loss curves from this env var
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(run_dir / "tb" / f"trainer_{task}")
    s = cfg.sft
    args = TrlSFTConfig(
        output_dir=str(run_dir / f"checkpoints_{task}"),
        learning_rate=s.lr,
        num_train_epochs=s.epochs,
        per_device_train_batch_size=s.per_device_batch_size,
        gradient_accumulation_steps=s.grad_accum,
        lr_scheduler_type=s.scheduler,
        warmup_steps=s.warmup_ratio,  # float in [0, 1) = ratio of total steps
        max_length=s.max_length,
        bf16=cfg.model.dtype == "bfloat16",
        gradient_checkpointing=s.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=s.logging_steps,
        save_steps=s.save_steps,
        save_total_limit=2,
        seed=s.seed,
        report_to=["tensorboard"],
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
