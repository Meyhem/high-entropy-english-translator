import pytest

from heet.cli import build_parser


def test_parser_subcommands():
    p = build_parser()
    a = p.parse_args(["sft", "--task", "compress", "--run", "r1", "--set", "sft.lr=1e-4"])
    assert (a.command, a.task, a.run, a.overrides) == ("sft", "compress", "r1", ["sft.lr=1e-4"])
    assert p.parse_args(["eval", "--run", "r1"]).split == "test"
    with pytest.raises(SystemExit):
        p.parse_args(["sft", "--task", "bogus", "--run", "r1"])


def test_parser_compress():
    p = build_parser()
    a = p.parse_args(["compress", "--run", "r1", "hello there", "second"])
    assert (a.command, a.run, a.mode, a.text) == ("compress", "r1", "compress", ["hello there", "second"])
    assert p.parse_args(["compress", "--run", "r1", "--mode", "roundtrip"]).text == []
    with pytest.raises(SystemExit):
        p.parse_args(["compress", "--run", "r1", "--mode", "bogus"])


def test_parser_ei():
    p = build_parser()
    a = p.parse_args(["ei", "--run", "r2", "--init", "r1", "--rounds", "2"])
    assert (a.command, a.run, a.init, a.rounds, a.overwrite) == ("ei", "r2", "r1", 2, False)
    assert p.parse_args(["ei", "--run", "r2", "--init", "r1"]).rounds is None


def test_ei_same_run_and_init_rejected():
    from heet.cli import main

    with pytest.raises(SystemExit):
        main(["ei", "--run", "r1", "--init", "r1"])
