from heet.eval.logger import MetricsLogger
from heet.eval.samples import write_samples
from heet.jsonl import read_jsonl


def test_metrics_logger_jsonl_and_tb(tmp_path):
    lg = MetricsLogger(tmp_path)
    lg.log(10, {"sim": 0.9, "pass": True}, prefix="student/", split="val")
    lg.log(20, {"sim": 0.95}, prefix="teacher/", split="val")
    lg.close()
    rows = read_jsonl(tmp_path / "metrics.jsonl")
    assert rows[0] == {"step": 10, "split": "val", "student/sim": 0.9, "student/pass": True}
    assert rows[1]["teacher/sim"] == 0.95
    assert any((tmp_path / "tb").iterdir())


def test_write_samples_fixed_plus_worst(tmp_path):
    recs = [{"id": str(i), "total": float(i)} for i in range(20)]
    recs[15]["total"] = -5.0
    p = write_samples(tmp_path, 7, recs, n_samples=5, worst_k=2, tag="val_")
    assert p.name == "val_step_000007.jsonl"
    rows = read_jsonl(p)
    assert [r["id"] for r in rows[:5]] == ["0", "1", "2", "3", "4"]
    # "0" (total 0.0) is lower than "5" but already in the fixed set
    worst = [r["id"] for r in rows if r["worst"]]
    assert worst == ["15", "5"]
