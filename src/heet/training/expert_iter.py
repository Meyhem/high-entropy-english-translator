"""Stage 3: expert iteration — best-of-K rewrites scored by the round trip, SFT on the winners."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from heet.config import Config
from heet.eval.evaluate import _mean
from heet.jsonl import write_jsonl
from heet.models.loading import generate, generate_samples, use_adapter
from heet.rewards.combined import RewardBreakdown


@dataclass
class Candidate:
    text: str
    greedy: bool = False
    teacher: bool = False
    recon: str = ""
    breakdown: RewardBreakdown | None = None


@dataclass
class Selection:
    winners: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def build_candidates(samples: list[str], greedy: str, teacher: str) -> list[Candidate]:
    by_text: dict[str, Candidate] = {}

    def add(text: str, is_greedy: bool = False, is_teacher: bool = False) -> None:
        text = text.strip()
        if not text and not is_greedy:
            return
        c = by_text.setdefault(text, Candidate(text))
        c.greedy |= is_greedy
        c.teacher |= is_teacher

    add(greedy, is_greedy=True)
    add(teacher, is_teacher=True)
    for s in samples:
        add(s)
    return list(by_text.values())


def round_subset(pairs: list[dict], n: int, seed: int) -> list[dict]:
    return random.Random(seed).sample(pairs, min(n, len(pairs)))


def select_winners(pairs: list[dict], cand_sets: list[list[Candidate]], min_sim: float, min_gain: float) -> Selection:
    winners: list[dict] = []
    greedy_totals: list[float] = []
    n_cands = 0
    for p, cands in zip(pairs, cand_sets, strict=True):
        n_cands += len(cands)
        greedy_total = next(c for c in cands if c.greedy).breakdown.total
        greedy_totals.append(greedy_total)
        eligible = [c for c in cands if not c.greedy and c.breakdown.sim >= min_sim]
        if not eligible:
            continue
        best = max(eligible, key=lambda c: c.breakdown.total)
        gain = best.breakdown.total - greedy_total
        if gain < min_gain:
            continue
        winners.append({
            "id": p["id"], "original": p["original"], "dense": best.text, "teacher": best.teacher,
            "gain": gain, "sim": best.breakdown.sim, "char_ratio": best.breakdown.char_ratio,
            "total": best.breakdown.total,
        })
    n = len(pairs)
    stats = {
        "n_originals": n,
        "n_candidates": n_cands,
        "n_winners": len(winners),
        "win_rate": len(winners) / n if n else 0.0,
        "winner_from_teacher": _mean(w["teacher"] for w in winners),
        "mean_gain": _mean(w["gain"] for w in winners),
        "greedy_reward": _mean(greedy_totals),
        "winner_char_ratio": _mean(w["char_ratio"] for w in winners),
        "winner_sim": _mean(w["sim"] for w in winners),
    }
    return Selection(winners, stats)


def sample_candidates(model, tokenizer, pairs: list[dict], cfg: Config) -> list[list[Candidate]]:
    originals = [p["original"] for p in pairs]
    e = cfg.ei
    with use_adapter(model, "compressor"):
        greedy = generate(model, tokenizer, "compress", originals, cfg.generation)
        samples = generate_samples(model, tokenizer, "compress", originals, cfg.generation,
                                   num_samples=e.k, temperature=e.temperature, top_p=e.top_p)
    return [build_candidates(s, g, p["dense"]) for p, g, s in zip(pairs, greedy, samples, strict=True)]


def score_candidates(model, tokenizer, pairs: list[dict], cand_sets: list[list[Candidate]], scorer, cfg: Config) -> None:
    flat = [(p["original"], c) for p, cands in zip(pairs, cand_sets, strict=True) for c in cands]
    denses = [c.text for _, c in flat]
    with use_adapter(model, "decompressor"):
        recons = generate(model, tokenizer, "decompress", denses, cfg.generation)
    breakdowns = scorer.score([o for o, _ in flat], denses, recons, compute_all=False)
    for (_, c), r, b in zip(flat, recons, breakdowns, strict=True):
        c.recon, c.breakdown = r, b


def write_round_samples(run_dir: Path, round_idx: int, pairs: list[dict], cand_sets: list[list[Candidate]],
                        winners: list[dict], n: int) -> Path:
    won = {w["id"]: w["dense"] for w in winners}
    records = [
        {"id": p["id"], "original": p["original"], "dense": c.text, "recon": c.recon,
         "greedy": c.greedy, "teacher": c.teacher, "winner": won.get(p["id"]) == c.text,
         **c.breakdown.to_dict()}
        for p, cands in zip(pairs[:n], cand_sets[:n]) for c in cands
    ]
    path = Path(run_dir) / "samples" / f"ei_round_{round_idx}.jsonl"
    write_jsonl(path, records)
    return path


def run_round(model, tokenizer, cfg: Config, pairs: list[dict], scorer, round_idx: int, run_dir: Path,
              logger, train_fn: Callable[[list[dict]], None]) -> Selection:
    t0 = time.perf_counter()
    cand_sets = sample_candidates(model, tokenizer, pairs, cfg)
    t1 = time.perf_counter()
    score_candidates(model, tokenizer, pairs, cand_sets, scorer, cfg)
    t2 = time.perf_counter()
    sel = select_winners(pairs, cand_sets, cfg.ei.min_sim, cfg.ei.min_gain)
    write_round_samples(run_dir, round_idx, pairs, cand_sets, sel.winners, cfg.eval.n_samples)
    if sel.winners:
        train_fn(sel.winners)
    t3 = time.perf_counter()
    logger.log(round_idx, {**sel.stats, "sample_s": t1 - t0, "score_s": t2 - t1, "train_s": t3 - t2},
               prefix="ei/", split="train", source="ei")
    return sel
