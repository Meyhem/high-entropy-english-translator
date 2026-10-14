# HEET Stage 3 — Expert Iteration: Design Spec

Date: 2026-10-14
Status: Draft for review
Parent spec: `2026-10-14-dense-english-autoencoder-design.md` (§1 goals, §7 rewards, §8 eval)

## 1. Goal

Stage 2 trains the compressor to imitate the teacher, so it cannot beat the teacher.
Stage 3 improves the **compressor** directly against the round-trip reward (§7 of the
parent spec) using **expert iteration**: sample several rewrites per sentence, score them,
keep the best, fine-tune on the winners, repeat.

On the r1 baseline (test split, 41 pairs) student and teacher both sit at
char_ratio ≈ 0.61, sim ≈ 0.93–0.95, ppl_log_ratio ≈ 1.7. Stage 3 succeeds if, versus r1:

- mean round-trip `reward` on val/test increases, and
- `char_ratio` decreases while mean `sim` stays ≥ 0.90.

### Decisions

| Decision | Choice | Reason |
|---|---|---|
| What trains | Compressor only; decompressor frozen | Frozen decompressor is a fixed judge; prevents a private code that only a co-trained decompressor understands. |
| Method | Best-of-K selection + SFT on winners | Reuses stage 2 trainer; every training target is a real, verified-good rewrite. GRPO deferred. |
| Readability margin | `reward.margin` default `ln 2` → **1.7** | At `ln 2` every rewrite (teacher included) is penalized; at 1.7 the penalty fires only when a rewrite is less readable than the current teacher/student level. The eval gate uses the same value. |
| Starting point | Adapters of an existing stage 2 run (`--init`) | Expert iteration needs a reasonable policy to sample from. |

### Out of scope

- Training the decompressor, GRPO / policy-gradient methods, explicit KL penalty.

## 2. CLI

```bash
uv run heet ei --run r2 --init r1 [--rounds 3] [--set ei.k=8 ...]
```

- `--init`: run whose `adapters/compressor` and `adapters/decompressor` are the starting
  point. Missing adapters → fail fast with the `heet sft` hint (existing `check_adapter_dir`).
- `--run` must differ from `--init`; the init run is never modified.
- `--rounds` overrides `ei.rounds`.
- `--overwrite` allows reusing a `--run` directory that already has `rounds/` (see §6).

## 3. Configuration (`ei` section, new `EIConfig` dataclass)

| Key | Default | Meaning |
|---|---|---|
| `rounds` | 3 | expert-iteration rounds |
| `n_per_round` | 1000 | train originals sampled per round (fresh random subset each round, seeded) |
| `k` | 8 | sampled rewrites per original |
| `temperature` | 0.8 | sampling temperature |
| `top_p` | 0.95 | nucleus sampling |
| `min_sim` | 0.90 | a winner must have reconstruction similarity ≥ this |
| `min_gain` | 0.02 | a winner's reward must exceed the greedy candidate's by ≥ this |
| `lr` | 5e-5 | SFT learning rate per round |
| `epochs` | 1.0 | SFT epochs over the round's winners |
| `seed` | 0 | subset sampling and generation seed (offset by round index) |

All other SFT settings (batch size, grad accumulation, max length, scheduler, gradient
checkpointing) come from the existing `sft` section.

Also changed: `reward.margin` default `math.log(2)` → `1.7` in `config.py` and
`configs/default.yaml`; `tests/test_config.py` updated accordingly.

## 4. Round algorithm

For round `r = 1..rounds`, with the model holding the current `compressor` (trainable)
and the frozen `decompressor`:

1. **Subset**: sample `n_per_round` pairs from the train split with
   `random.Random(seed + r)`.
2. **Candidates** (per original), via `sample_candidates`:
   - `k` sampled compressor outputs (`temperature`, `top_p`),
   - 1 greedy compressor output (the *baseline* candidate),
   - the teacher's `dense` from `pairs.jsonl`.
   Empty strings are dropped; exact duplicates within one original's candidate set are
   collapsed (scored once), keeping the flag that one of them is the greedy baseline.
3. **Score**: decompress every unique candidate with the frozen decompressor (greedy) and
   score `(original, candidate, recon)` with the existing `Scorer` (`compute_all=False`,
   so only terms with non-zero weight are computed).
4. **Select** (`select_winners`, pure function): for each original take the
   highest-`total` candidate; keep it as a training pair `(original → candidate)` only if
   - `sim ≥ min_sim`, and
   - `total ≥ greedy_total + min_gain`, and
   - it is not the greedy candidate itself.
   Returns winners plus selection stats.
5. **Fine-tune**: if there are no winners, log a warning and stop early. Otherwise run
   SFT on the winners (compress task, `ei.lr`, `ei.epochs`) continuing from the current
   compressor weights.
6. **Evaluate**: run the existing compress-task round-trip evaluation (`EvalCallback.evaluate`
   logic) on the fixed val subset; log with `source="ei"`, `step=r`, prefix `compress/`.
7. **Save**: compressor adapter to `runs/<run>/rounds/<r>/compressor/` and to
   `runs/<run>/adapters/compressor/` (latest). The decompressor from `--init` is copied once
   to `runs/<run>/adapters/decompressor/` so `heet eval` / `heet compress` work on the new run.

Round 0 (before any training) is also evaluated and logged as the baseline.

### Selection stats logged per round (`ei/` prefix, `source="ei"`)

`n_originals`, `n_candidates` (unique), `n_winners`, `win_rate`,
`winner_from_teacher` (fraction of winners that are the teacher's dense),
`mean_gain` (winner total − greedy total, over winners), `greedy_reward` (mean greedy total),
`winner_char_ratio`, `winner_sim`, and wall-clock seconds for sampling, scoring and training.

### Samples

`samples/ei_round_<r>.jsonl`: for the first `eval.n_samples` originals of the round, every
candidate with its breakdown, a `greedy` flag, a `teacher` flag and a `winner` flag.

## 5. Code changes

| File | Change |
|---|---|
| `models/loading.py` | `generate(..., sample: bool = False, num_samples: int = 1, temperature: float = 1.0, top_p: float = 1.0)`. With `sample=True` returns `num_samples` outputs per input, grouped per input (`list[list[str]]`) via a new `generate_samples` wrapper so the existing `generate` return type stays `list[str]`. Prompts per batch = `max(1, batch_size // num_samples)` to keep memory constant. |
| `training/sft.py` | Extract the trainer construction/run from `run_sft` into `train_adapter(model, tokenizer, cfg, pairs, task, run_dir, lr, epochs, tag, callbacks)` so stage 3 can continue training an already-loaded adapter on an in-memory pair list. `run_sft` keeps its signature and behavior. |
| `training/expert_iter.py` (new) | `Candidate` dataclass; `sample_candidates`; `select_winners` (pure); `run_round`; `run_ei(cfg, run, init, rounds)`. |
| `config.py`, `configs/default.yaml` | `EIConfig`; `reward.margin` default 1.7. |
| `cli.py` | `ei` subcommand. |
| `README.md`, `CLAUDE.md` | usage line; stage order. |

Model setup for `run_ei`: load base, `PeftModel.from_pretrained(init compressor, is_trainable=True)`,
`load_adapter(init decompressor, is_trainable=False)`, active adapter `compressor`.
Generation switches adapters through the existing `use_adapter`; training always happens
with `compressor` active.

## 6. Error handling

- Missing init adapters → `FileNotFoundError` with `heet sft` hint, before loading models.
- `--run == --init` → argparse-level error.
- Round with zero winners → warning, stop early, outputs from the last good round remain.
- `runs/<run>/` already containing `rounds/` → fail fast unless `--overwrite` is given (no
  silent mixing of two experiments' metrics).

## 7. Testing

Fast (default `pytest`), using stub embedder/ref LM from `tests/stubs.py` and a fake
`generate`:

- `select_winners`: picks max-total; rejects below `min_sim`; rejects insufficient gain;
  rejects greedy-as-winner; teacher can win; duplicate collapsing keeps the greedy flag;
  stats arithmetic.
- `sample_candidates`: candidate set composition (k samples + greedy + teacher), empty
  outputs dropped, duplicates collapsed.
- `run_round` with fake generate + stub scorer + fake trainer: winners passed to trainer,
  no training when there are no winners, stats logged.
- CLI parser: `ei` args, `--rounds` override, `--run == --init` rejected.
- Config: `ei` defaults, new `reward.margin` default.

Slow (`-m slow`): tiny random Qwen3 on CPU — create two tiny adapters, run `heet ei` for
1 round with `n_per_round=4, k=2`, assert `rounds/1/compressor/adapter_config.json`,
`adapters/decompressor/`, and `ei` lines in `metrics.jsonl` exist.

## 8. Expected cost

Per round with defaults: ~1,000 × (8 sampled + 1 greedy) compressor generations,
≤ 10,000 decompressor generations, scoring (embedder + ref LM) of ≤ 10,000 triples, and
one SFT epoch over ≤ 1,000 winners. Estimated 15–25 min/round on the RTX 5080
(Qwen3-0.6B); first run will report measured per-phase timings.

## 9. Workflow

```bash
uv run heet ei --run r2 --init r1          # 3 rounds
uv run heet eval --run r2                  # compare against r1
uv run heet compress --run r2 --mode roundtrip "Some text."
```
