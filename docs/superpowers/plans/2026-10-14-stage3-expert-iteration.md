# Stage 3 Expert Iteration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `heet ei`, which improves the compressor LoRA by repeated best-of-K sampling, round-trip scoring with a frozen decompressor, and SFT on the winning rewrites.

**Architecture:** A new `training/expert_iter.py` holds small units: candidate building and winner selection (pure), candidate sampling/scoring (model calls through `heet.models.loading`), one round (`run_round`, trainer injected), and the orchestrator `run_ei`. It reuses stage 2's SFT trainer (extracted into `train_adapter`), the `Scorer`, and the compress-task `EvalCallback.evaluate`. `generate` gains a sampling sibling `generate_samples`.

**Tech Stack:** Python 3, PyTorch, transformers, peft, TRL `SFTTrainer`, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-10-14-stage3-expert-iteration-design.md`

## Global Constraints

- Only the `compressor` adapter trains; the `decompressor` is loaded with `is_trainable=False` and never updated.
- `--init` run is never modified; `--run` must differ from `--init`.
- `reward.margin` default becomes `1.7` (was `math.log(2)`); the eval `pass` gate uses the same value.
- `ei` defaults: `rounds=3, n_per_round=1000, k=8, temperature=0.8, top_p=0.95, min_sim=0.90, min_gain=0.02, lr=5e-5, epochs=1.0, seed=0`.
- Existing public signatures keep working: `generate(model, tokenizer, task, texts, gen) -> list[str]`, `run_sft(cfg, task, run, limit) -> Path`, `EvalCallback(cfg, task, tokenizer, pairs, scorer, logger, run_dir, has_decompressor)`.
- Heavy imports (torch-model loading, trl, datasets) stay lazy inside functions, as in `cli.py`/`sft.py`.
- Run tests with `uv run pytest` (fast) and `uv run pytest -m slow` (tiny-model tests).
- Commits: conventional messages ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Push to `origin main` at the end (CLAUDE.md).

## Review Focus

1. Train split smaller than `ei.n_per_round` → round uses all train pairs, no `ValueError` from `random.sample` (test in Task 4).
2. Greedy output identical to the teacher's dense → one collapsed candidate flagged both greedy and teacher; it can never be a winner (test in Task 4).
3. Every sampled output empty → only greedy (+ teacher) candidates remain; no crash, normal selection (test in Task 4).
4. Re-running `heet ei` with an existing `--run` directory → fails fast before loading models unless `--overwrite` (test in Task 6).
5. `ei.k=1` → `generate_samples` still returns one list per input with one string each (test in Task 2).

---

### Task 1: Config — `EIConfig` and relaxed readability margin

**Files:**
- Modify: `src/heet/config.py` (RewardConfig.margin; new EIConfig; Config.ei)
- Modify: `configs/default.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `heet.config.EIConfig` with fields `rounds:int, n_per_round:int, k:int, temperature:float, top_p:float, min_sim:float, min_gain:float, lr:float, epochs:float, seed:int`; `Config.ei: EIConfig`; `RewardConfig.margin == 1.7`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_config.py`, replace the line `assert math.isclose(c.reward.margin, math.log(2))` with `assert c.reward.margin == 1.7`, and append:

```python
def test_ei_defaults_and_override(tmp_path: Path):
    c = Config()
    e = c.ei
    assert (e.rounds, e.n_per_round, e.k) == (3, 1000, 8)
    assert (e.temperature, e.top_p, e.min_sim, e.min_gain) == (0.8, 0.95, 0.90, 0.02)
    assert (e.lr, e.epochs, e.seed) == (5e-5, 1.0, 0)
    c2 = load_config("configs/default.yaml", ["ei.k=4", "ei.min_gain=-10"])
    assert c2.ei.k == 4 and c2.ei.min_gain == -10.0 and isinstance(c2.ei.min_gain, float)
    assert c2.reward.margin == 1.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL (`AttributeError: 'Config' object has no attribute 'ei'`, margin assertion).

- [ ] **Step 3: Implement**

In `src/heet/config.py`, change `RewardConfig.margin`:

```python
    # readability penalty fires when log(ppl_dense / ppl_orig) exceeds this;
    # 1.7 ≈ the teacher's own level on WikiText (ln 2 penalized every rewrite)
    margin: float = 1.7
```

Add after `EvalConfig`:

```python
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
```

Add to `Config` after `eval`:

```python
    ei: EIConfig = field(default_factory=EIConfig)
```

If `math` is no longer used in `config.py` after this change, remove its import.

In `configs/default.yaml`, change `  margin: 0.6931471805599453` to `  margin: 1.7` and append:

```yaml
ei:
  rounds: 3
  n_per_round: 1000
  k: 8
  temperature: 0.8
  top_p: 0.95
  min_sim: 0.90
  min_gain: 0.02
  lr: 0.00005
  epochs: 1.0
  seed: 0
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/heet/config.py configs/default.yaml tests/test_config.py
git commit -m "feat: ei config section and readability margin 1.7"
```

---

### Task 2: Sampling generation — `generate_samples`

**Files:**
- Modify: `src/heet/models/loading.py:83-113` (`generate`)
- Test: `tests/test_models.py` (slow, tiny model)

**Interfaces:**
- Consumes: `GenerationConfig` (`batch_size`, `max_new_tokens_factor`, `max_new_tokens_extra`).
- Produces: `generate_samples(model, tokenizer, task: str, texts: list[str], gen: GenerationConfig, num_samples: int, temperature: float, top_p: float) -> list[list[str]]` — `len(result) == len(texts)`, each inner list has `num_samples` cleaned strings. `generate(...)` unchanged in signature and behavior.

- [ ] **Step 1: Write the failing test** (append to `tests/test_models.py`)

```python
@pytest.mark.parametrize("k", [1, 3])
def test_generate_samples_groups_per_input(tok, k):
    from heet.models.loading import generate_samples

    base = load_base(ModelConfig(name=TINY, dtype="float32", device="cpu"))
    texts = ["Hello there.", "A second, longer sentence.", "Third."]
    gen = GenerationConfig(batch_size=4, max_new_tokens_factor=0.0, max_new_tokens_extra=4)
    torch.manual_seed(0)
    out = generate_samples(base, tok, "compress", texts, gen, num_samples=k, temperature=0.8, top_p=0.95)
    assert len(out) == len(texts)
    assert all(len(group) == k and all(isinstance(s, str) for s in group) for group in out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -m slow tests/test_models.py -k samples -v`
Expected: FAIL with `ImportError: cannot import name 'generate_samples'`.

- [ ] **Step 3: Implement** — replace the whole `generate` function in `src/heet/models/loading.py` with:

```python
def _batched_generate(model, tokenizer, task: str, texts: list[str], gen: GenerationConfig,
                      prompts_per_batch: int, **gen_kwargs) -> list[str]:
    """Flat list of cleaned outputs; with num_return_sequences=k the k outputs of each input are adjacent."""
    was_training = model.training
    model.eval()
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    outputs: list[str] = []
    try:
        for start in range(0, len(texts), prompts_per_batch):
            batch = texts[start : start + prompts_per_batch]
            prompts = [render_prompt(tokenizer, task, t) for t in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            longest_input = max(len(ids) for ids in tokenizer(batch, add_special_tokens=False)["input_ids"])
            max_new = int(gen.max_new_tokens_factor * longest_input) + gen.max_new_tokens_extra
            out = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tokenizer.pad_token_id, **gen_kwargs)
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            for seq in tokenizer.batch_decode(new_tokens, skip_special_tokens=True):
                outputs.append(clean_output(seq))
    finally:
        tokenizer.padding_side = old_side
        if was_training:
            model.train()
    return outputs


@torch.no_grad()
def generate(model, tokenizer, task: str, texts: list[str], gen: GenerationConfig) -> list[str]:
    return _batched_generate(model, tokenizer, task, texts, gen, gen.batch_size, do_sample=False)


@torch.no_grad()
def generate_samples(model, tokenizer, task: str, texts: list[str], gen: GenerationConfig,
                     num_samples: int, temperature: float, top_p: float) -> list[list[str]]:
    # fewer prompts per batch so batch_size bounds the number of sequences in flight
    flat = _batched_generate(
        model, tokenizer, task, texts, gen, max(1, gen.batch_size // num_samples),
        do_sample=True, temperature=temperature, top_p=top_p, top_k=0,  # top_k=0 overrides Qwen's default 20
        num_return_sequences=num_samples,
    )
    return [flat[i * num_samples : (i + 1) * num_samples] for i in range(len(texts))]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -m slow tests/test_models.py -v && uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/heet/models/loading.py tests/test_models.py
git commit -m "feat: generate_samples for k sampled outputs per input"
```

---

### Task 3: Reusable SFT step and configurable eval callback

**Files:**
- Modify: `src/heet/training/sft.py` (`EvalCallback.__init__`/`evaluate`; extract `train_adapter` from `run_sft`)
- Test: `tests/test_sft.py`

**Interfaces:**
- Produces:
  - `train_adapter(model, tokenizer, cfg: Config, pairs: list[dict], task: str, run_dir: Path, *, lr: float, epochs: float, tag: str, callbacks: list | None = None) -> None` — trains the currently active adapter of `model` on `pairs` (dicts with `original`, `dense`); checkpoints to `run_dir/checkpoints_{tag}`, trainer TensorBoard to `run_dir/tb/trainer_{tag}`.
  - `EvalCallback(..., has_decompressor, source: str = "callback", sample_tag: str | None = None)` — logs with `source=source`; sample files named `{sample_tag}step_XXXXXX.jsonl` (default tag `f"{task}_"`).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_sft.py`)

```python
def test_callback_custom_source_and_tag(tmp_path, monkeypatch):
    import heet.training.sft as sft

    monkeypatch.setattr(sft, "generate", lambda model, tok, task, texts, gen: [t.lower()[:10] for t in texts])
    cfg = Config()
    cb = EvalCallback(cfg, "compress", WsTok(), PAIRS, Scorer(cfg.reward, StubEmbedder(), StubRef()),
                      MetricsLogger(tmp_path), tmp_path, has_decompressor=True, source="ei", sample_tag="ei_val_")
    cb.evaluate(EchoModel(), step=2)
    assert read_jsonl(tmp_path / "metrics.jsonl")[0]["source"] == "ei"
    assert (tmp_path / "samples" / "ei_val_step_000002.jsonl").exists()


def test_train_adapter_uses_given_lr_epochs_and_tag(tmp_path, monkeypatch):
    import trl

    from heet.training.sft import train_adapter

    seen = {}

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, processing_class, callbacks):
            seen.update(args=args, n=len(train_dataset), callbacks=callbacks)

        def train(self):
            seen["trained"] = True

    monkeypatch.setattr(trl, "SFTTrainer", FakeTrainer)
    cfg = Config()
    cfg.model.dtype = "float32"
    train_adapter(object(), FakeTok(), cfg, PAIRS * 3, "compress", tmp_path, lr=5e-5, epochs=1.0, tag="ei_round_1")
    a = seen["args"]
    assert seen["trained"] and seen["n"] == 3 and seen["callbacks"] == []
    assert a.learning_rate == 5e-5 and a.num_train_epochs == 1.0
    assert a.output_dir == str(tmp_path / "checkpoints_ei_round_1")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sft.py -v`
Expected: FAIL (`unexpected keyword argument 'source'`, `cannot import name 'train_adapter'`).

- [ ] **Step 3: Implement**

`EvalCallback.__init__` becomes:

```python
    def __init__(self, cfg: Config, task: str, tokenizer, pairs: list[dict], scorer: Scorer,
                 logger: MetricsLogger, run_dir: Path, has_decompressor: bool,
                 source: str = "callback", sample_tag: str | None = None):
        self.cfg, self.task, self.tokenizer, self.pairs = cfg, task, tokenizer, pairs
        self.scorer, self.logger, self.run_dir = scorer, logger, Path(run_dir)
        self.has_decompressor = has_decompressor
        self.source = source
        self.sample_tag = sample_tag if sample_tag is not None else f"{task}_"
        self._warned = False
```

In `evaluate`, replace the last two statements before `return metrics` with:

```python
        self.logger.log(step, metrics, prefix=f"{self.task}/", split="val", source=self.source)
        write_samples(self.run_dir, step, records, self.cfg.eval.n_samples, self.cfg.eval.worst_k, tag=self.sample_tag)
```

Add `train_adapter` above `run_sft`:

```python
def train_adapter(model, tokenizer, cfg: Config, pairs: list[dict], task: str, run_dir: Path, *,
                  lr: float, epochs: float, tag: str, callbacks: list | None = None) -> None:
    """Train the model's active adapter on `pairs` for the given task."""
    from datasets import Dataset
    from trl import SFTConfig as TrlSFTConfig
    from trl import SFTTrainer

    run_dir = Path(run_dir)
    # transformers 5 reads the TensorBoard dir for trainer loss curves from this env var
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(run_dir / "tb" / f"trainer_{tag}")
    s = cfg.sft
    args = TrlSFTConfig(
        output_dir=str(run_dir / f"checkpoints_{tag}"),
        learning_rate=lr,
        num_train_epochs=epochs,
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
        train_dataset=Dataset.from_list(build_examples(pairs, task, tokenizer)),
        processing_class=tokenizer,
        callbacks=callbacks or [],
    )
    trainer.train()
```

In `run_sft`: delete the three local trl/datasets imports, the `os.environ[...]` line, `s = cfg.sft`, the `args = TrlSFTConfig(...)` block and the `trainer = SFTTrainer(...)` / `trainer.train()` lines, and in their place call:

```python
    train_adapter(model, tokenizer, cfg, train_pairs, task, run_dir,
                  lr=cfg.sft.lr, epochs=cfg.sft.epochs, tag=task, callbacks=callbacks)
```

(Output dirs stay `checkpoints_{task}` and `tb/trainer_{task}` because `tag=task`. `FakeTok` used by the new test is already imported in `tests/test_sft.py`.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q && uv run pytest -m slow tests/test_smoke.py -v`
Expected: all pass (smoke test proves `run_sft` still works end to end).

- [ ] **Step 5: Commit**

```bash
git add src/heet/training/sft.py tests/test_sft.py
git commit -m "refactor: extract train_adapter; EvalCallback source and sample tag"
```

---

### Task 4: Candidates and winner selection (pure logic)

**Files:**
- Create: `src/heet/training/expert_iter.py`
- Test: `tests/test_expert_iter.py`

**Interfaces:**
- Consumes: `heet.rewards.combined.RewardBreakdown`; `heet.eval.evaluate._mean`.
- Produces:
  - `Candidate` dataclass: `text: str, greedy: bool = False, teacher: bool = False, recon: str = "", breakdown: RewardBreakdown | None = None`.
  - `build_candidates(samples: list[str], greedy: str, teacher: str) -> list[Candidate]` — strips text; drops empty non-greedy texts; collapses exact duplicates, OR-ing the `greedy`/`teacher` flags; greedy candidate always present (even if empty) and first.
  - `round_subset(pairs: list[dict], n: int, seed: int) -> list[dict]` — `random.Random(seed).sample(pairs, min(n, len(pairs)))`.
  - `Selection` dataclass: `winners: list[dict]`, `stats: dict`. Winner dict keys: `id, original, dense, teacher, gain, sim, char_ratio, total`.
  - `select_winners(pairs: list[dict], cand_sets: list[list[Candidate]], min_sim: float, min_gain: float) -> Selection` — per original: among non-greedy candidates with `sim >= min_sim`, take the max-`total` one; keep it if `total - greedy_total >= min_gain`. Stats keys: `n_originals, n_candidates, n_winners, win_rate, winner_from_teacher, mean_gain, greedy_reward, winner_char_ratio, winner_sim`.

(Interpretation of spec §4.4: the best candidate is chosen among those that already pass `min_sim` and are not greedy, so a high-reward but meaning-losing candidate cannot hide a valid second-best one.)

- [ ] **Step 1: Write the failing tests** — create `tests/test_expert_iter.py`:

```python
import pytest

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_expert_iter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'heet.training.expert_iter'`.

- [ ] **Step 3: Implement** — create `src/heet/training/expert_iter.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_expert_iter.py -v && uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/heet/training/expert_iter.py tests/test_expert_iter.py
git commit -m "feat: expert iteration candidate building and winner selection"
```

---

### Task 5: Sampling, scoring and one round (`run_round`)

**Files:**
- Modify: `src/heet/training/expert_iter.py`
- Test: `tests/test_expert_iter.py`

**Interfaces:**
- Consumes: `generate`, `generate_samples`, `use_adapter` from `heet.models.loading` (Task 2) — imported at module level in `expert_iter.py` so tests can monkeypatch `heet.training.expert_iter.<name>`; `Scorer.score(originals, denses, recons, compute_all=False)`; `MetricsLogger.log`; `heet.jsonl.write_jsonl`; Task 4 types.
- Produces:
  - `sample_candidates(model, tokenizer, pairs, cfg) -> list[list[Candidate]]`
  - `score_candidates(model, tokenizer, pairs, cand_sets, scorer, cfg) -> None` (fills `recon`, `breakdown` in place)
  - `write_round_samples(run_dir, round_idx, pairs, cand_sets, winners, n) -> Path` → `run_dir/samples/ei_round_{round_idx}.jsonl`
  - `run_round(model, tokenizer, cfg, pairs, scorer, round_idx: int, run_dir: Path, logger, train_fn: Callable[[list[dict]], None]) -> Selection` — logs `prefix="ei/"`, `split="train"`, `source="ei"`, `step=round_idx`, stats plus `sample_s, score_s, train_s`; calls `train_fn(winners)` only if there are winners.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_expert_iter.py`)

```python
import contextlib

from heet.config import Config
from heet.eval.logger import MetricsLogger
from heet.jsonl import read_jsonl


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_expert_iter.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_round'`.

- [ ] **Step 3: Implement** — in `src/heet/training/expert_iter.py` add to the imports:

```python
import time
from collections.abc import Callable
from pathlib import Path

from heet.config import Config
from heet.jsonl import write_jsonl
from heet.models.loading import generate, generate_samples, use_adapter
```

and append:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_expert_iter.py -v && uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/heet/training/expert_iter.py tests/test_expert_iter.py
git commit -m "feat: expert iteration sampling, scoring and round loop"
```

---

### Task 6: Orchestrator `run_ei` and `heet ei` CLI

**Files:**
- Modify: `src/heet/training/expert_iter.py` (add `run_ei`, `save_compressor`)
- Modify: `src/heet/cli.py`
- Test: `tests/test_expert_iter.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `adapter_dir`, `check_adapter_dir`, `load_base`, `load_tokenizer` (loading.py); `build_scorer`, `load_split_pairs` (evaluate.py); `EvalCallback`, `select_val_pairs`, `train_adapter` (sft.py, Task 3); `dump_config`; Task 4/5 functions.
- Produces: `run_ei(cfg: Config, run: str, init: str, rounds: int | None = None, overwrite: bool = False) -> Path` (returns `runs/<run>/adapters/compressor`); CLI `heet ei --run R --init I [--rounds N] [--overwrite]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_expert_iter.py`:

```python
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
```

Append to `tests/test_cli.py`:

```python
def test_parser_ei():
    p = build_parser()
    a = p.parse_args(["ei", "--run", "r2", "--init", "r1", "--rounds", "2"])
    assert (a.command, a.run, a.init, a.rounds, a.overwrite) == ("ei", "r2", "r1", 2, False)
    assert p.parse_args(["ei", "--run", "r2", "--init", "r1"]).rounds is None


def test_ei_same_run_and_init_rejected():
    from heet.cli import main

    with pytest.raises(SystemExit):
        main(["ei", "--run", "r1", "--init", "r1"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_expert_iter.py tests/test_cli.py -v`
Expected: FAIL (`cannot import name 'run_ei'`; `invalid choice: 'ei'`).

- [ ] **Step 3: Implement**

Append to `src/heet/training/expert_iter.py` (add `import logging`, `import shutil` to the imports and `log = logging.getLogger(__name__)` below them):

```python
def save_compressor(model, tokenizer, root: Path) -> None:
    model.save_pretrained(str(root), selected_adapters=["compressor"])
    tokenizer.save_pretrained(str(Path(root) / "compressor"))


def run_ei(cfg: Config, run: str, init: str, rounds: int | None = None, overwrite: bool = False) -> Path:
    import torch
    from peft import PeftModel

    from heet.config import dump_config
    from heet.eval.evaluate import build_scorer, load_split_pairs
    from heet.eval.logger import MetricsLogger
    from heet.models.loading import adapter_dir, check_adapter_dir, load_base, load_tokenizer
    from heet.training.sft import EvalCallback, select_val_pairs, train_adapter

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    comp_dir, decomp_dir = adapter_dir(cfg, init, "compressor"), adapter_dir(cfg, init, "decompressor")
    check_adapter_dir("compressor", comp_dir)
    check_adapter_dir("decompressor", decomp_dir)
    run_dir = Path(cfg.paths.runs_dir) / run
    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{run_dir} already exists; pick another --run or pass --overwrite")
        shutil.rmtree(run_dir)
    n_rounds = rounds if rounds is not None else cfg.ei.rounds
    dump_config(cfg, run_dir / "config_ei.yaml")

    train_pairs = load_split_pairs(cfg, "train", None)
    val_pairs = select_val_pairs(cfg)
    tokenizer = load_tokenizer(cfg.model.name)
    model = PeftModel.from_pretrained(load_base(cfg.model), str(comp_dir), adapter_name="compressor",
                                      is_trainable=True)
    model.load_adapter(str(decomp_dir), adapter_name="decompressor", is_trainable=False)
    model.set_adapter("compressor")
    shutil.copytree(decomp_dir, run_dir / "adapters" / "decompressor")

    scorer = build_scorer(model, tokenizer, cfg)
    logger = MetricsLogger(run_dir)
    evaluator = EvalCallback(cfg, "compress", tokenizer, val_pairs, scorer, logger, run_dir,
                             has_decompressor=True, source="ei", sample_tag="ei_val_")
    if val_pairs:
        evaluator.evaluate(model, 0)
    save_compressor(model, tokenizer, run_dir / "adapters")  # round 0 = init weights until a round trains

    for r in range(1, n_rounds + 1):
        torch.manual_seed(cfg.ei.seed + r)
        subset = round_subset(train_pairs, cfg.ei.n_per_round, cfg.ei.seed + r)
        sel = run_round(
            model, tokenizer, cfg, subset, scorer, r, run_dir, logger,
            train_fn=lambda winners, r=r: train_adapter(
                model, tokenizer, cfg, winners, "compress", run_dir,
                lr=cfg.ei.lr, epochs=cfg.ei.epochs, tag=f"ei_round_{r}"),
        )
        log.info("round %d: %d/%d winners, mean gain %.3f", r, sel.stats["n_winners"],
                 sel.stats["n_originals"], sel.stats["mean_gain"])
        if not sel.winners:
            log.warning("round %d produced no winners; stopping early", r)
            break
        if val_pairs:
            evaluator.evaluate(model, r)
        save_compressor(model, tokenizer, run_dir / "rounds" / str(r))
        save_compressor(model, tokenizer, run_dir / "adapters")
    logger.close()
    return run_dir / "adapters" / "compressor"
```

In `src/heet/cli.py`, add the subparser before `return parser`:

```python
    x = sub.add_parser("ei", help="stage 3: expert iteration on the compressor")
    _common(x)
    x.add_argument("--run", required=True, help="new run dir for stage 3 outputs")
    x.add_argument("--init", required=True, help="stage 2 run whose adapters to start from")
    x.add_argument("--rounds", type=int, default=None, help="overrides ei.rounds")
    x.add_argument("--overwrite", action="store_true", help="delete an existing --run dir first")
```

Change the start of `main` to:

```python
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ei" and args.run == args.init:
        parser.error("--run must differ from --init (the init run is never modified)")
    cfg = load_config(args.config, args.overrides)
```

and add the branch before `return 0`:

```python
    elif args.command == "ei":
        from heet.training.expert_iter import run_ei

        run_ei(cfg, run=args.run, init=args.init, rounds=args.rounds, overwrite=args.overwrite)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/heet/training/expert_iter.py src/heet/cli.py tests/test_expert_iter.py tests/test_cli.py
git commit -m "feat: heet ei command orchestrating expert iteration rounds"
```

---

### Task 7: End-to-end slow test, docs, push

**Files:**
- Modify: `tests/test_smoke.py`
- Modify: `README.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: `run_sft`, `run_ei`, `run_eval`.

- [ ] **Step 1: Write the slow test** — in `tests/test_smoke.py`, move the data/config setup of `test_sft_then_eval_end_to_end` into a helper and add the ei test:

```python
def _tiny_setup(tmp_path):
    rows = [{"id": f"{s}{i}", "split": s, "original": ORIG, "dense": DENSE, "teacher_recon": ORIG,
             "similarity": 1.0, "ratio": 0.5} for s in ("train", "val", "test") for i in range(4)]
    write_jsonl(tmp_path / "data" / "pairs.jsonl", rows)
    cfg = load_config(None, [
        f"paths.data_dir={tmp_path / 'data'}", f"paths.runs_dir={tmp_path / 'runs'}",
        f"model.name={TINY}", "model.dtype=float32", "model.device=cpu",
        "sft.epochs=1", "sft.per_device_batch_size=2", "sft.grad_accum=1", "sft.eval_steps=1",
        "sft.gradient_checkpointing=false", "sft.max_length=128",
        "generation.max_new_tokens_factor=0.0", "generation.max_new_tokens_extra=4",
        "eval.callback_n=2", "eval.n_samples=2", "eval.worst_k=1",
        "reward.embedder=sentence-transformers/all-MiniLM-L6-v2",
    ])
    cfg.lora.target_modules = ["q_proj", "v_proj"]
    return cfg
```

`test_sft_then_eval_end_to_end` starts with `cfg = _tiny_setup(tmp_path)` instead of the inlined setup (rest unchanged). Add:

```python
def test_ei_one_round_end_to_end(tmp_path):
    from heet.training.expert_iter import run_ei

    cfg = _tiny_setup(tmp_path)
    run_sft(cfg, task="decompress", run="r", limit=None)
    run_sft(cfg, task="compress", run="r", limit=None)
    # random tiny model: accept any non-greedy candidate so a round always trains
    cfg.ei.n_per_round, cfg.ei.k, cfg.ei.min_sim, cfg.ei.min_gain = 4, 2, -1.0, -10.0
    run_ei(cfg, run="r2", init="r", rounds=1)

    d = tmp_path / "runs" / "r2"
    assert (d / "rounds" / "1" / "compressor" / "adapter_config.json").exists()
    assert (d / "adapters" / "compressor" / "adapter_config.json").exists()
    assert (d / "adapters" / "decompressor" / "adapter_config.json").exists()
    assert (d / "samples" / "ei_round_1.jsonl").exists()
    rows = read_jsonl(d / "metrics.jsonl")
    assert {r["step"] for r in rows if r["source"] == "ei"} == {0, 1}
    assert any("ei/n_winners" in r for r in rows)
    assert run_eval(cfg, run="r2", split="test", limit=4)["student"]["n"] == 4
```

Add `from heet.jsonl import read_jsonl, write_jsonl` (replacing the `write_jsonl`-only import).

- [ ] **Step 2: Run it**

Run: `uv run pytest -m slow tests/test_smoke.py -v`
Expected: PASS. If `PeftModel.from_pretrained(..., is_trainable=True)` fails under gradient checkpointing in the real config, add `model.enable_input_require_grads()` right after loading in `run_ei` (the test disables checkpointing, so verify again with the real run in Step 5).

- [ ] **Step 3: Docs**

README usage block — add after the `heet eval` line:

```
uv run heet ei --run r2 --init r1          # stage 3: expert iteration on the compressor (runs/r2/)
```

CLAUDE.md — change the stage-order line to:

```
- Stage order: `gen-data` → `sft --task decompress` → `sft --task compress` → `eval` → `ei --run r2 --init r1` → `eval --run r2`.
```

- [ ] **Step 4: Full test run, commit**

```bash
uv run pytest -q && uv run pytest -m slow -q
git add tests/test_smoke.py README.md CLAUDE.md
git commit -m "test: stage 3 end-to-end smoke test; docs for heet ei"
```

- [ ] **Step 5: Real-run sanity check and push**

Run one short real round on the GPU to confirm the stack works at scale and to measure timing:

```bash
uv run heet ei --run ei_check --init r1 --rounds 1 --set ei.n_per_round=64
```

Expected: completes; `runs/ei_check/metrics.jsonl` has `source: "ei"` rows for steps 0 and 1 with `ei/sample_s`, `ei/score_s`, `ei/train_s`. Then:

```bash
git push origin main
```
