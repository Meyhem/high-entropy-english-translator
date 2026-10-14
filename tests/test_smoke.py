import pytest

from heet.config import load_config
from heet.eval.evaluate import run_eval
from heet.jsonl import read_jsonl, write_jsonl
from heet.training.sft import run_sft

pytestmark = pytest.mark.slow

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
ORIG = "The government announced that the new bridge would open in March 2021."
DENSE = "Gov: new bridge opens March 2021."


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


def test_sft_then_eval_end_to_end(tmp_path):
    cfg = _tiny_setup(tmp_path)
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
