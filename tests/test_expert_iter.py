import contextlib

import pytest

from heet.config import Config
from heet.eval.logger import MetricsLogger
from heet.jsonl import read_jsonl
from heet.rewards.combined import RewardBreakdown
from heet.training.expert_iter import Candidate, build_candidates, round_subset, select_winners


def scored(text, total, sim=1.0, greedy=False, teacher=False, char_ratio=0.5):
    return Candidate(text, greedy=greedy, teacher=teacher,
                     breakdown=RewardBreakdown(total=total, sim=sim, char_ratio=char_ratio))


P = {"id": "a", "original": "The cat sat on the mat."}


def test_build_candidates_composition_and_dedup():
    cs = build_candidates(["cat sat", " cat sat ", "", "mat cat"], greedy="cat on mat", teacher="cat sat")
    assert [c.text for c in cs] == ["cat on mat", "cat sat", "mat cat"]
    assert cs[0].greedy and not cs[0].teacher
    assert cs[1].teacher and not cs[1].greedy


def test_build_candidates_greedy_equals_teacher_and_empty_samples():
    cs = build_candidates(["", "  "], greedy="cat sat", teacher="cat sat")
    assert len(cs) == 1 and cs[0].greedy and cs[0].teacher


def test_build_candidates_keeps_empty_greedy():
    cs = build_candidates(["cat"], greedy="", teacher="cat sat")
    assert cs[0].text == "" and cs[0].greedy


def test_round_subset_caps_at_available():
    pairs = [{"id": str(i)} for i in range(3)]
    assert len(round_subset(pairs, 1000, seed=1)) == 3
    assert round_subset(pairs, 2, seed=1) == round_subset(pairs, 2, seed=1)


def test_select_picks_best_eligible_with_gain():
    cands = [scored("g", 0.30, greedy=True), scored("t", 0.40, teacher=True),
             scored("s1", 0.90, sim=0.50), scored("s2", 0.50, char_ratio=0.4)]
    sel = select_winners([P], [cands], min_sim=0.9, min_gain=0.02)
    [w] = sel.winners
    assert w["dense"] == "s2" and w["original"] == P["original"] and w["id"] == "a"
    assert w["gain"] == pytest.approx(0.20) and w["teacher"] is False
    assert sel.stats["n_winners"] == 1 and sel.stats["win_rate"] == 1.0
    assert sel.stats["n_candidates"] == 4 and sel.stats["greedy_reward"] == pytest.approx(0.30)
    assert sel.stats["winner_char_ratio"] == pytest.approx(0.4)


def test_select_rejects_small_gain_and_greedy_best():
    small = [scored("g", 0.30, greedy=True), scored("s", 0.31)]
    greedy_best = [scored("g", 0.90, greedy=True), scored("s", 0.20)]
    sel = select_winners([P, {**P, "id": "b"}], [small, greedy_best], min_sim=0.9, min_gain=0.02)
    assert sel.winners == [] and sel.stats["win_rate"] == 0.0 and sel.stats["mean_gain"] == 0.0


def test_select_greedy_teacher_collapsed_never_wins():
    only = [scored("same", 0.10, greedy=True, teacher=True)]
    assert select_winners([P], [only], min_sim=0.0, min_gain=-10).winners == []


def test_select_teacher_can_win():
    cands = [scored("g", 0.1, greedy=True), scored("t", 0.5, teacher=True)]
    sel = select_winners([P], [cands], min_sim=0.9, min_gain=0.02)
    assert sel.winners[0]["teacher"] is True and sel.stats["winner_from_teacher"] == 1.0


class ShortIsBetterScorer:
    """total = 1 - len(dense)/100; sim fixed per instance."""

    def __init__(self, sim=1.0):
        self.sim = sim

    def score(self, originals, denses, recons, compute_all=True):
        return [RewardBreakdown(total=1 - len(d) / 100, sim=self.sim, char_ratio=len(d) / len(o))
                for o, d in zip(originals, denses)]


@pytest.fixture
def fake_gen(monkeypatch):
    import heet.training.expert_iter as ei

    monkeypatch.setattr(ei, "use_adapter", lambda model, name: contextlib.nullcontext(model))
    monkeypatch.setattr(ei, "generate", lambda model, tok, task, texts, gen:
                        [t.lower() for t in texts] if task == "compress" else list(texts))
    monkeypatch.setattr(ei, "generate_samples", lambda model, tok, task, texts, gen, num_samples, temperature, top_p:
                        [["sat.", ""][:num_samples] for _ in texts])


PAIRS = [{"id": "a", "original": "The cat sat on the mat.", "dense": "cat sat mat"},
         {"id": "b", "original": "Dogs bark loudly at night.", "dense": "dogs bark @ night"}]


def test_run_round_trains_on_winners_and_logs(tmp_path, fake_gen):
    from heet.training.expert_iter import run_round

    cfg = Config()
    cfg.ei.k = 2
    trained = []
    sel = run_round(None, None, cfg, PAIRS, ShortIsBetterScorer(), 1, tmp_path, MetricsLogger(tmp_path), trained.append)
    assert [w["dense"] for w in sel.winners] == ["sat.", "sat."]
    assert trained == [sel.winners]
    row = read_jsonl(tmp_path / "metrics.jsonl")[0]
    assert row["step"] == 1 and row["source"] == "ei" and row["ei/n_winners"] == 2 and "ei/sample_s" in row
    recs = read_jsonl(tmp_path / "samples" / "ei_round_1.jsonl")
    assert {r["dense"] for r in recs if r["id"] == "a"} == {"the cat sat on the mat.", "cat sat mat", "sat."}
    assert sum(r["winner"] for r in recs) == 2 and all("total" in r for r in recs)


def test_run_round_without_winners_does_not_train(tmp_path, fake_gen):
    from heet.training.expert_iter import run_round

    trained = []
    sel = run_round(None, None, Config(), PAIRS, ShortIsBetterScorer(sim=0.5), 1, tmp_path,
                    MetricsLogger(tmp_path), trained.append)
    assert sel.winners == [] and trained == []


def _fake_init(tmp_path):
    for name in ("compressor", "decompressor"):
        d = tmp_path / "init" / "adapters" / name
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}")


def test_run_ei_missing_init_fails_fast(tmp_path):
    from heet.training.expert_iter import run_ei

    cfg = Config()
    cfg.paths.runs_dir = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="heet sft"):
        run_ei(cfg, run="r2", init="nope")


def test_run_ei_existing_run_needs_overwrite(tmp_path):
    from heet.training.expert_iter import run_ei

    _fake_init(tmp_path)
    (tmp_path / "r2").mkdir()
    (tmp_path / "r2" / "metrics.jsonl").write_text("")
    cfg = Config()
    cfg.paths.runs_dir = str(tmp_path)
    with pytest.raises(FileExistsError, match="--overwrite"):
        run_ei(cfg, run="r2", init="init")


@pytest.mark.parametrize("run", ["init/", "./init", "init/sub"])
def test_run_ei_refuses_run_dir_overlapping_init(tmp_path, run):
    from heet.training.expert_iter import run_ei

    _fake_init(tmp_path)
    cfg = Config()
    cfg.paths.runs_dir = str(tmp_path)
    with pytest.raises(ValueError, match="init"):
        run_ei(cfg, run=run, init="init", overwrite=True)
    assert (tmp_path / "init" / "adapters" / "compressor" / "adapter_config.json").exists()
