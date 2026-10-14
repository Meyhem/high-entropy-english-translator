"""Metrics → JSONL + TensorBoard."""

from __future__ import annotations

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from heet.jsonl import append_jsonl


class MetricsLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))

    def log(self, step: int, metrics: dict, prefix: str = "", **meta) -> None:
        row = {"step": step, **meta, **{f"{prefix}{k}": v for k, v in metrics.items()}}
        append_jsonl(self.run_dir / "metrics.jsonl", row)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._tb.add_scalar(f"{prefix}{k}", float(v), step)
        self._tb.flush()

    def close(self) -> None:
        self._tb.close()
