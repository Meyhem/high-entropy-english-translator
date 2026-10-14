"""Round-trip evaluation: original → compressor → dense → decompressor → reconstruction."""

from __future__ import annotations

import json
import logging
from collections import Counter
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
    all_rows = read_jsonl(path)
    rows = [r for r in all_rows if r["split"] == split]
    if not rows:
        counts = Counter(r["split"] for r in all_rows)
        found = ", ".join(f"{k}={counts.get(k, 0)}" for k in ("train", "val", "test"))
        raise FileNotFoundError(
            f"no '{split}' pairs in {path} (found {found}); run `heet gen-data` with a larger "
            "--limit or raise corpus.val_frac / corpus.test_frac"
        )
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
