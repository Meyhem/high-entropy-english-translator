# HEET — Dense-English Text Autoencoder: Design Spec

Date: 2026-10-14
Status: Draft for review

## 1. Goal

A PyTorch research prototype that learns to rewrite plain English into shorter,
higher-entropy but still human-readable English ("dense English"), and to
reconstruct the original from it.

- **Compressor**: plain → dense (abbreviations, dropped function words, telegraphic phrasing).
- **Decompressor**: dense → plain.

### Success criteria (evaluated on the test split)

| Metric | Target |
|---|---|
| Character compression ratio (dense chars / original chars) | ≤ 0.55 |
| Reconstruction cosine similarity (mean) | ≥ 0.90 |
| Readability: `log(ppl_dense / ppl_orig)` under frozen reference LM | ≤ margin (default `log 2`) |

### Constraints

- Runs on a single 16 GB GPU (dev machine: RTX 5080, Blackwell sm_120 → PyTorch CUDA 12.8+ build).
- Python project managed with **uv**; CLI command is **`heet`**.
- Local teacher LLM via **Ollama** only.
- Each stage runnable and testable independently.

### Out of scope (for now)

- Stage 3 (RL / GRPO / expert iteration). The reward module is built so a later
  stage 3 can reuse it unchanged, but no RL code is written now.
- Hosted LLM APIs (Anthropic, OpenAI).
- Weights & Biases or other hosted tracking.

## 2. Key decisions

| Decision | Choice | Reason |
|---|---|---|
| Base model | `Qwen/Qwen3-0.6B` default; `Qwen/Qwen3-1.7B` via config | Fast iteration; 1.7B still fits 16 GB with LoRA. Thinking disabled (`enable_thinking=False`). |
| Adapter layout | One frozen base, two named LoRA adapters: `compressor`, `decompressor` | Independent training of each direction; base with adapters disabled doubles as the frozen reference LM. |
| Teacher | Ollama `/api/chat`, default model `qwen3:14b`, `think: false` | Local, free; ~9 GB at Q4. Runs in a separate stage, never alongside training. |
| Corpus | WikiText-103 (HF `Salesforce/wikitext`, `wikitext-103-raw-v1`); any plain-text file also accepted | Clean prose, easy download. |
| Unit of compression | Chunks of 1–3 sentences, 40–300 characters | Short sequences keep training/eval cheap; strict reconstruction checks feasible. |
| Readability penalty | Reference-LM perplexity ratio (not KL) | KL needs a policy distribution, only meaningful under RL. |
| Logging | JSONL + TensorBoard | Local, no account. |

## 3. Repository layout

```
pyproject.toml                # uv project; torch from PyTorch cu128 index; script entry `heet`
configs/default.yaml          # every knob (models, prompts, thresholds, LoRA, reward weights)
src/heet/
  __init__.py
  config.py                   # typed config (dataclasses) from YAML + `--set a.b=c` overrides
  prompts.py                  # compress/decompress chat messages + teacher prompts
  cli.py                      # `heet gen-data | sft | eval`
  data/
    corpus.py                 # corpus → chunks.jsonl
    llm_client.py             # async Ollama client
    generate.py               # stage 1 pipeline + filter
  models/
    loading.py                # base model, adapters, batched generation
  training/
    sft.py                    # stage 2 + eval callback
  rewards/
    fidelity.py               # embedding similarity, exact match, chrF
    ref_lm.py                 # frozen reference LM log-probs
    length.py                 # char ratio, bits ratio
    readability.py            # perplexity-ratio penalty
    combined.py               # weighted reward + breakdown
  eval/
    evaluate.py               # round-trip evaluation + teacher baseline
    logger.py                 # JSONL + TensorBoard writer
    samples.py                # sample-triple dumps
tests/
runs/                         # (gitignored) outputs per run
data/                         # (gitignored) chunks/pairs
```

Module dependency direction: `cli → {data, training, eval} → {models, rewards, prompts, config}`.
`rewards` depends only on `models` (for the reference LM) and third-party libs; it
never imports training code.

## 4. Configuration

Single YAML file (`configs/default.yaml`) parsed into nested dataclasses. Any field
can be overridden on the command line: `heet sft --task compress --set sft.lr=1e-4`.
Each run writes its resolved config to `runs/<run>/config.yaml`.

Top-level sections: `paths`, `corpus`, `teacher`, `filter`, `model`, `lora`, `sft`,
`generation`, `reward`, `eval`.

## 5. Stage 1 — data generation (`heet gen-data`)

### 5.1 Chunking (`data/corpus.py`)

1. Load WikiText-103 (or a local `.txt` file via `corpus.path`).
2. Clean: drop headings (`= ... =` lines), empty lines; normalize wikitext artifacts
   (` @-@ ` → `-`, ` @,@ ` → `,`, ` @.@ ` → `.` including surrounding spaces, so `1 @,@ 000` → `1,000`; remove spaces before punctuation).
3. Split paragraphs into sentences (regex splitter with abbreviation guard; no NLTK
   download required).
4. Greedily group 1–3 consecutive sentences into chunks of 40–300 characters; drop
   chunks outside the range.
5. `id = sha1(text)[:16]`; split assigned deterministically from the id hash
   (default 98% / 1% / 1% train/val/test).
6. Write `data/chunks.jsonl`: `{id, text, split}`. `corpus.max_chunks` caps the output.

### 5.2 Teacher generation (`data/llm_client.py`, `data/generate.py`)

For each chunk not already present in `pairs.jsonl` or `rejected.jsonl`:

1. **Rewrite**: teacher receives a system prompt with the dense-English rules plus
   few-shot examples, and the chunk. Rules: abbreviate common words, drop
   articles/copulas/auxiliaries where meaning survives, telegraphic phrasing, use
   symbols (`&`, `w/`, `→`, `~`, `b/c`), keep every proper name and number exactly,
   output only the rewrite.
2. **Blind reconstruction**: a fresh teacher call (no access to the original) expands
   the dense text back to full English.
3. **Filter** — keep the pair only if all hold:
   - embedding cosine similarity(original, reconstruction) ≥ `filter.min_similarity` (0.90);
   - `filter.min_ratio` (0.30) ≤ dense chars / original chars ≤ `filter.max_ratio` (0.80);
   - every number token in the original appears verbatim in the dense text;
   - dense is not a near-copy (normalized char-level similarity to original < 0.9).
4. Accepted → `data/pairs.jsonl`: `{id, split, original, dense, teacher_recon, similarity, ratio}`.
   Rejected → `data/rejected.jsonl` with the same fields plus `reason`.

Client details: `httpx.AsyncClient`, bounded concurrency (`teacher.concurrency`,
default 4; set `OLLAMA_NUM_PARALLEL` accordingly), exponential-backoff retries,
temperature 0.3 for rewrites and 0.0 for reconstruction, `think: false`.
Writes are appended and flushed per item, so the stage can be interrupted and resumed.
`--limit N` processes at most N new chunks.

The embedder used by the filter is the same `rewards/fidelity.py` embedder used at eval.

## 6. Stage 2 — supervised fine-tuning (`heet sft --task {decompress,compress}`)

Recommended order: **decompress first**, then compress (the compressor's eval callback
needs a decompressor).

- Base model loaded in bf16; LoRA via peft: r=16, alpha=32, dropout 0.05, targets
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
- TRL `SFTTrainer` with prompt/completion conversational examples; loss on completion
  only. Max length 256 tokens; gradient checkpointing on; defaults lr 2e-4, cosine
  schedule, 2 epochs, effective batch 32.
- Training data from `pairs.jsonl` train split:
  - `decompress`: user = dense, assistant = original.
  - `compress`: user = original, assistant = dense.
- Prompts built by `prompts.py` with the Qwen3 chat template, `enable_thinking=False`.
- Output: `runs/<run>/adapters/{compressor|decompressor}/` (adapter weights + tokenizer).

### 6.1 Eval callback

Runs every `sft.eval_steps` on a fixed val subset (`eval.callback_n`, default 64),
model in eval mode under `torch.no_grad()`:

- **compress task**: generate dense (greedy) → load the configured decompressor
  adapter as a second, frozen adapter on the same PeftModel → switch to it, reconstruct
  → switch back to `compressor`. Computes the full metric set (§8).
- **decompress task**: reconstruct from teacher dense; reports similarity, exact match, chrF.

Metrics go to the logger (§8); sample triples are written to `samples/`.

## 7. Rewards (`rewards/`)

Pure scoring, no training code. Main entry point:

```python
score(originals: list[str], denses: list[str], recons: list[str]) -> list[RewardBreakdown]
```

`RewardBreakdown` holds every component plus the total.

| Component | Definition |
|---|---|
| `sim` | cosine similarity of sentence embeddings (`sentence-transformers/all-mpnet-base-v2`, configurable) between original and reconstruction |
| `exact` | 1 if normalized (lowercase, collapse whitespace, strip punctuation at ends) original == reconstruction else 0 |
| `chrf` | sacrebleu chrF / 100 (optional) |
| `char_ratio` | `len(dense) / len(original)` |
| `bits_ratio` | ref-LM bits(dense) / ref-LM bits(original) |
| `read_pen` | `max(0, log(ppl_dense / ppl_orig) − margin)` |

Reference LM (`ref_lm.py`): the base Qwen3 with adapters disabled (or a separately
loaded base if no PeftModel is present). Text is scored unconditionally: tokens are
prefixed with a fixed `<|endoftext|>` context token and summed token NLL gives
`bits = Σ −log2 p`; `ppl = exp(mean NLL)`.

Combined reward:

```
R = w_sim·sim + w_exact·exact + w_chrf·chrf − w_char·char_ratio − w_bits·bits_ratio − w_read·read_pen
```

Defaults: `w_sim=1.0, w_exact=0.2, w_chrf=0.0, w_char=0.5, w_bits=0.25, w_read=0.5, margin=ln 2`.
Components with weight 0 are skipped when computed (e.g. no ref-LM pass if both `w_bits` and `w_read` are 0
— except at eval, where all metrics are always computed for logging).

## 8. Evaluation (`heet eval --run <run>`)

Round trip on the test split (or `--split val`, `--limit N`):
original → compressor (greedy) → dense → decompressor (greedy) → reconstruction.

Logged metrics (each as mean; similarity also p10):

- `char_ratio`, `token_ratio` (dense tokens / original tokens)
- `bpc_dense` = ref-LM bits(dense) / chars(original); `bpc_orig` = bits(original) / chars(original)
- `bits_ratio`
- `sim`, `sim_p10`, `exact_rate`, `chrf`
- `ppl_dense`, `ppl_orig`, `ppl_log_ratio`
- `reward` (combined)
- `pass` — boolean against the success criteria in §1

**Teacher baseline**: same metrics with dense = teacher's dense from `pairs.jsonl`,
decompressed by our decompressor. Logged under the `teacher/` prefix.

Outputs under `runs/<run>/`:

- `metrics.jsonl` — one line per eval: `{step, split, source, ...metrics}`
- `tb/` — TensorBoard scalars
- `samples/step_XXXXXX.jsonl` — `{id, original, dense, recon, ...breakdown}` for a
  fixed set of sample ids (`eval.n_samples`, default 32) plus the 10 lowest-reward items.

## 9. Error handling

- Ollama unreachable / model missing → fail fast with a message naming `teacher.base_url`
  and the `ollama pull <model>` command.
- Malformed teacher output (empty, contains the prompt, contains `<think>`) → rejected with
  reason, not retried.
- HTTP / timeout errors → retried with backoff (`teacher.max_retries`, default 3), then rejected
  with reason `teacher_error`.
- Compress-task eval without a decompressor adapter path → callback logs only the
  compressor-side metrics and warns once.
- Generation runs with `max_new_tokens` = 2 × input tokens + 16 to bound runaway outputs.

## 10. Testing (pytest)

- `test_corpus.py` — cleaning, sentence splitting, chunk length bounds, deterministic split.
- `test_generate.py` — fake LLM client: each filter reason, number-preservation check,
  resume skips processed ids.
- `test_rewards.py` — combined-reward arithmetic with stub embedder and stub ref LM;
  readability margin; exact-match normalization.
- `test_prompts.py` — chat template rendering, thinking disabled, completion-only labels.
- `test_smoke.py` (`@pytest.mark.slow`) — tiny random Qwen3 model on CPU: 2 SFT steps per
  task, 4-example eval, checks output files exist.

Default `uv run pytest` skips `slow`; `uv run pytest -m slow` runs them.

## 11. Typical workflow

```bash
ollama pull qwen3:14b
uv sync
uv run heet gen-data --limit 20000
uv run heet sft --task decompress --run r1
uv run heet sft --task compress --run r1
uv run heet eval --run r1
uv run tensorboard --logdir runs/
```

## 12. Future stage 3 hook

A later RL stage (GRPO or expert iteration) would call `rewards.combined.score` on
compressor samples and periodically re-run decompressor SFT on the compressor's
outputs mixed with stage-1 pairs. Nothing in stages 1–2 needs to change for that.
