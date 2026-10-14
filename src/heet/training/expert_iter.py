"""Stage 3: expert iteration — best-of-K rewrites scored by the round trip, SFT on the winners."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from heet.eval.evaluate import _mean
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
