import pytest

from heet.cli import build_parser


def test_parser_subcommands():
    p = build_parser()
    a = p.parse_args(["sft", "--task", "compress", "--run", "r1", "--set", "sft.lr=1e-4"])
    assert (a.command, a.task, a.run, a.overrides) == ("sft", "compress", "r1", ["sft.lr=1e-4"])
    assert p.parse_args(["eval", "--run", "r1"]).split == "test"
    with pytest.raises(SystemExit):
        p.parse_args(["sft", "--task", "bogus", "--run", "r1"])
