"""`heet` command-line entry point."""

from __future__ import annotations

import argparse
import sys

from heet.config import load_config


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heet", description="Dense-English autoencoder")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gen-data", help="stage 1: chunk corpus and generate filtered pairs")
    _common(g)
    g.add_argument("--limit", type=int, default=None, help="max new chunks to send to the teacher")
    g.add_argument("--rechunk", action="store_true", help="rebuild chunks.jsonl even if present")

    s = sub.add_parser("sft", help="stage 2: train one LoRA adapter")
    _common(s)
    s.add_argument("--task", choices=["compress", "decompress"], required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--limit", type=int, default=None, help="max training pairs")

    e = sub.add_parser("eval", help="round-trip evaluation")
    _common(e)
    e.add_argument("--run", required=True)
    e.add_argument("--split", choices=["val", "test"], default="test")
    e.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if args.command == "gen-data":
        from heet.data.generate import run_gen_data

        run_gen_data(cfg, limit=args.limit, rechunk=args.rechunk)
    elif args.command == "sft":
        from heet.training.sft import run_sft

        run_sft(cfg, task=args.task, run=args.run, limit=args.limit)
    elif args.command == "eval":
        from heet.eval.evaluate import run_eval

        run_eval(cfg, run=args.run, split=args.split, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
