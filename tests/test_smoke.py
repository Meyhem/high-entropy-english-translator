import pytest

from heet.config import load_config
from heet.eval.evaluate import run_eval
from heet.jsonl import write_jsonl
from heet.training.sft import run_sft

pytestmark = pytest.mark.slow

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
ORIG = "The government announced that the new bridge would open in March 2021."
DENSE = "Gov: new bridge opens March 2021."


def test_sft_then_eval_end_to_end(tmp_path):
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
    run_sft(cfg, task="decompress", run="r", limit=None)
    run_sft(cfg, task="compress", run="r", limit=None)
    out = run_eval(cfg, run="r", split="test", limit=4)

    run_dir = tmp_path / "runs" / "r"
    assert (run_dir / "adapters" / "compressor" / "adapter_config.json").exists()
    assert (run_dir / "adapters" / "decompressor" / "adapter_config.json").exists()
    assert (run_dir / "metrics.jsonl").exists() and any((run_dir / "tb").iterdir())
    assert list((run_dir / "samples").glob("compress_step_*.jsonl"))
    assert list((run_dir / "samples").glob("test_student_step_*.jsonl"))
    assert {"char_ratio", "sim", "bpc_dense", "ppl_log_ratio", "pass"} <= out["student"].keys()
