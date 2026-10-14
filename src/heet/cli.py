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

    c = sub.add_parser("compress", help="compress / decompress / round-trip your own text")
    _common(c)
    c.add_argument("--run", required=True)
    c.add_argument("--mode", choices=["compress", "decompress", "roundtrip"], default="compress")
    c.add_argument("text", nargs="*", help="input texts; reads one per line from stdin if omitted")

    x = sub.add_parser("ei", help="stage 3: expert iteration on the compressor")
    _common(x)
    x.add_argument("--run", required=True, help="new run dir for stage 3 outputs")
    x.add_argument("--init", required=True, help="stage 2 run whose adapters to start from")
    x.add_argument("--rounds", type=int, default=None, help="overrides ei.rounds")
    x.add_argument("--overwrite", action="store_true", help="delete an existing --run dir first")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ei" and args.run == args.init:
        parser.error("--run must differ from --init (the init run is never modified)")
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
    elif args.command == "compress":
        from heet.infer import run_compress

        run_compress(cfg, run=args.run, texts=args.text or sys.stdin.read().splitlines(), mode=args.mode)
    elif args.command == "ei":
        from heet.training.expert_iter import run_ei

        run_ei(cfg, run=args.run, init=args.init, rounds=args.rounds, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
