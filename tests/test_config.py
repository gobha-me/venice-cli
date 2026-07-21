"""Tests for `venice config` (userconfig I/O + the config subcommands).

Points config.CONFIG_DIR/CONFIG_FILE at a tmpdir so nothing touches the real
~/.config/venice. The API key is never written here.
"""
import argparse
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import venice.config as cfg
import venice.userconfig as uc
from venice import cli
from venice.commands import config as cfgcmd
from venice.commands import (
    image, image_edit, index, music, sfx, tts, upscale, video,
)


def _add_args(name="srv", **ov):
    base = dict(
        name=name, server_command=None, arg=[], env=[],
        url=None, server_type="http", header=[], force=False,
    )
    base.update(ov)
    return argparse.Namespace(**base)


def _capture(fn, *args):
    """Run fn(*args), swallow stdout/stderr, return (rc, out, err)."""
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(sys, "stdout", out), mock.patch.object(sys, "stderr", err):
        rc = fn(*args)
    return rc, out.getvalue(), err.getvalue()


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        home = Path(self.tmp.name)
        self.cfg_dir = home / ".config" / "venice"
        self.cfg_file = self.cfg_dir / "config.json"
        for name, val in (("CONFIG_DIR", self.cfg_dir), ("CONFIG_FILE", self.cfg_file)):
            p = mock.patch.object(cfg, name, val)
            p.start()
            self.addCleanup(p.stop)


# --------------------------------------------------------------------------- #
# userconfig I/O
# --------------------------------------------------------------------------- #
class TestUserConfigIO(_Base):
    def test_load_missing_returns_default(self):
        doc = uc.load_config()
        self.assertEqual(doc, {"version": 1, "mcpServers": {}, "defaults": {}})

    def test_save_is_atomic_0600_and_parses(self):
        uc.save_config({"version": 1, "mcpServers": {"a": {"command": "x"}}, "defaults": {}})
        self.assertTrue(self.cfg_file.exists())
        mode = stat.S_IMODE(self.cfg_file.stat().st_mode)
        self.assertEqual(mode, 0o600)
        self.assertEqual(list(self.cfg_dir.glob("*.tmp")), [])  # no leftover temp
        self.assertEqual(json.loads(self.cfg_file.read_text())["mcpServers"]["a"]["command"], "x")

    def test_load_malformed_is_tolerant(self):
        self.cfg_dir.mkdir(parents=True)
        self.cfg_file.write_text("{not json")
        _, _, err = _capture(uc.load_config)  # warns but does not raise
        self.assertIn(str(self.cfg_file), err)
        self.assertEqual(uc.load_config()["defaults"], {})

    def test_load_for_write_refuses_malformed(self):
        self.cfg_dir.mkdir(parents=True)
        self.cfg_file.write_text("[]")  # valid JSON, wrong type
        with self.assertRaises(uc.ConfigError):
            uc.load_config_for_write()

    def test_dotted_get_set_unset(self):
        doc = uc._default_doc()
        uc.set_value(doc, "defaults.chat.model", "m1")
        self.assertEqual(uc.get_value(doc, "defaults.chat.model"), "m1")
        self.assertTrue(uc.unset_value(doc, "defaults.chat.model"))
        self.assertFalse(uc.unset_value(doc, "defaults.chat.model"))
        with self.assertRaises(KeyError):
            uc.get_value(doc, "defaults.chat.model")

    def test_set_through_non_table_raises(self):
        doc = {"defaults": {"chat": "oops"}}
        with self.assertRaises(uc.ConfigError):
            uc.set_value(doc, "defaults.chat.model", "m1")

    def test_unknown_keys_survive_round_trip(self):
        self.cfg_dir.mkdir(parents=True)
        self.cfg_file.write_text(json.dumps({"version": 1, "future_thing": {"k": 1}}))
        doc = uc.load_config_for_write()
        uc.set_value(doc, "defaults.chat.model", "m1")
        uc.save_config(doc)
        reloaded = json.loads(self.cfg_file.read_text())
        self.assertEqual(reloaded["future_thing"], {"k": 1})  # not dropped


# --------------------------------------------------------------------------- #
# resolve_default / apply_defaults (#17)
# --------------------------------------------------------------------------- #
class TestApplyDefaults(_Base):
    def test_resolve_default_command_beats_global(self):
        doc = {"defaults": {"max_spend": 1.0, "chat": {"max_spend": 0.2}}}
        self.assertEqual(uc.resolve_default("chat", "max_spend", doc), 0.2)
        self.assertEqual(uc.resolve_default("sfx", "max_spend", doc), 1.0)
        self.assertIsNone(uc.resolve_default("chat", "missing", doc))

    def test_apply_fills_none_only(self):
        doc = {"defaults": {"chat": {"model": "cfg-model"}}}
        args = argparse.Namespace(model=None, system=None, temperature=None,
                                  max_tokens=None, web_search=None, character=None)
        uc.apply_defaults(args, "chat", doc)
        self.assertEqual(args.model, "cfg-model")
        # explicit value is never overwritten
        args2 = argparse.Namespace(model="explicit", system=None, temperature=None,
                                   max_tokens=None, web_search=None, character=None)
        uc.apply_defaults(args2, "chat", doc)
        self.assertEqual(args2.model, "explicit")

    def test_apply_global_output_dir_expands_user(self):
        doc = {"defaults": {"output_dir": "~/venice-out", "max_spend": 0.5, "yes": True}}
        args = argparse.Namespace(output=None, max_spend=None, yes=None)
        uc.apply_defaults(args, "sfx", doc)
        self.assertEqual(args.output, Path("~/venice-out").expanduser())
        self.assertEqual(args.max_spend, 0.5)
        self.assertIs(args.yes, True)

    def test_apply_skips_flag_command_does_not_have(self):
        # chat has no --output; a global output_dir must not invent the attr
        doc = {"defaults": {"output_dir": "~/x"}}
        args = argparse.Namespace(model=None)
        uc.apply_defaults(args, "chat", doc)
        self.assertFalse(hasattr(args, "output"))

    def test_apply_bad_value_is_skipped_not_fatal(self):
        doc = {"defaults": {"chat": {"temperature": "not-a-number"}}}
        args = argparse.Namespace(temperature=None, model=None, system=None,
                                  max_tokens=None, web_search=None, character=None)
        _, _, err = _capture(uc.apply_defaults, args, "chat", doc)
        self.assertIsNone(args.temperature)  # unchanged
        self.assertIn("temperature", err)


# --------------------------------------------------------------------------- #
# #57 config parity -- Class A: flags that already default None become
# config-backable by a pure `_COMMAND_MAP` addition (no argparse change). Each
# case parses the command's REAL parser (so a wrong dest name would be caught),
# fills from a `defaults.<cmd>.<key>` doc, and confirms an explicit CLI wins.
# --------------------------------------------------------------------------- #
def _build_parser(mod):
    parser = argparse.ArgumentParser(prog="venice")
    sub = parser.add_subparsers(dest="command")
    mod.register(sub)
    return parser


_CLASS_A_CASES = [
    dict(
        mod=image, argv=["image"], key="image",
        config={
            "width": 512, "height": 768, "aspect_ratio": "16:9",
            "resolution": "2K", "style_prefix": "oil painting of",
            "preset": "myp", "preset_file": "~/p.json",
            "negative_prompt": "blurry", "cfg_scale": "7.5", "steps": 30,
            "style_preset": "anime",
        },
        expected={
            "width": 512, "height": 768, "aspect_ratio": "16:9",
            "resolution": "2K", "style_prefix": "oil painting of",
            "preset": "myp", "preset_file": Path("~/p.json").expanduser(),
            "negative_prompt": "blurry", "cfg_scale": 7.5, "steps": 30,
            "style_preset": "anime",
        },
        explicit=["image", "--steps", "10"], edest="steps", eval=10,
    ),
    dict(
        mod=image_edit, argv=["image-edit"], key="image_edit",
        config={"model": "edit-m", "aspect_ratio": "1:1",
                "resolution": "1K", "output_format": "webp"},
        expected={"model": "edit-m", "aspect_ratio": "1:1",
                  "resolution": "1K", "output_format": "webp"},
        explicit=["image-edit", "--model", "cli-m"], edest="model", eval="cli-m",
    ),
    dict(
        mod=tts, argv=["tts"], key="tts",
        config={"voice": "af_sky", "speed": "1.25", "play": "false"},
        expected={"voice": "af_sky", "speed": 1.25, "play": False},
        explicit=["tts", "--voice", "cli-v"], edest="voice", eval="cli-v",
    ),
    dict(
        mod=sfx, argv=["sfx"], key="sfx",
        config={"play": True},
        expected={"play": True},
        explicit=["sfx", "--no-play"], edest="play", eval=False,
    ),
    dict(
        mod=music, argv=["music"], key="music",
        config={"duration": 30, "speed": "0.9", "play": "no"},
        expected={"duration": 30, "speed": 0.9, "play": False},
        explicit=["music", "--speed", "2.0"], edest="speed", eval=2.0,
    ),
    dict(
        mod=video, argv=["video"], key="video",
        config={"model": "vid-1", "resolution": "720p",
                "aspect_ratio": "16:9", "negative_prompt": "text"},
        expected={"model": "vid-1", "resolution": "720p",
                  "aspect_ratio": "16:9", "negative_prompt": "text"},
        explicit=["video", "--model", "cli-vid"], edest="model", eval="cli-vid",
    ),
    dict(
        mod=upscale, argv=["upscale", "in.png"], key="upscale",
        config={"enhance_creativity": "0.5", "enhance_prompt": "gold",
                "replication": "0.3"},
        expected={"enhance_creativity": 0.5, "enhance_prompt": "gold",
                  "replication": 0.3},
        explicit=["upscale", "in.png", "--replication", "0.9"],
        edest="replication", eval=0.9,
    ),
    dict(
        mod=index, argv=["index"], key="index",
        config={"exclude": ["*.min.js", "vendor/"]},
        expected={"exclude": ["*.min.js", "vendor/"]},
        explicit=["index", "--exclude", "cli-pat"], edest="exclude",
        eval=["cli-pat"],
    ),
]


class TestClassAParity(unittest.TestCase):
    def test_config_fills_none_dests(self):
        for case in _CLASS_A_CASES:
            with self.subTest(cmd=case["key"]):
                parser = _build_parser(case["mod"])
                args = parser.parse_args(case["argv"])
                doc = {"defaults": {case["key"]: case["config"]}}
                uc.apply_defaults(args, case["key"], doc)
                for dest, want in case["expected"].items():
                    self.assertEqual(getattr(args, dest), want,
                                     msg=f"{case['key']}.{dest}")

    def test_explicit_cli_beats_config(self):
        for case in _CLASS_A_CASES:
            with self.subTest(cmd=case["key"]):
                parser = _build_parser(case["mod"])
                args = parser.parse_args(case["explicit"])
                doc = {"defaults": {case["key"]: case["config"]}}
                uc.apply_defaults(args, case["key"], doc)
                self.assertEqual(getattr(args, case["edest"]), case["eval"])

    def test_index_exclude_scalar_becomes_list(self):
        parser = _build_parser(index)
        args = parser.parse_args(["index"])
        doc = {"defaults": {"index": {"exclude": "solo-pat"}}}
        uc.apply_defaults(args, "index", doc)
        self.assertEqual(args.exclude, ["solo-pat"])


# --------------------------------------------------------------------------- #
# config subcommands
# --------------------------------------------------------------------------- #
class TestConfigCommand(_Base):
    def test_add_stdio_roundtrip(self):
        rc, _, _ = _capture(cfgcmd._run_add,
                            _add_args("venice", server_command="venice", arg=["mcp-serve"]))
        self.assertEqual(rc, 0)
        entry = uc.mcp_get(uc.load_config(), "venice")
        self.assertEqual(entry, {"command": "venice", "args": ["mcp-serve"]})
        # list shows it
        rc, out, _ = _capture(cfgcmd._run_list, argparse.Namespace(json=False))
        self.assertEqual(rc, 0)
        self.assertIn("venice", out)
        self.assertIn("mcp-serve", out)
        # show one entry as JSON
        rc, out, _ = _capture(cfgcmd._run_show, argparse.Namespace(name="venice", json=False))
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out)["command"], "venice")
        # remove it
        rc, _, _ = _capture(cfgcmd._run_remove, argparse.Namespace(name="venice"))
        self.assertEqual(rc, 0)
        self.assertEqual(uc.mcp_map(uc.load_config()), {})

    def test_add_http_with_header(self):
        rc, _, _ = _capture(cfgcmd._run_add, _add_args(
            "remote", url="https://x/mcp", header=["Authorization: Bearer T"]))
        self.assertEqual(rc, 0)
        entry = uc.mcp_get(uc.load_config(), "remote")
        self.assertEqual(entry, {"type": "http", "url": "https://x/mcp",
                                 "headers": {"Authorization": "Bearer T"}})

    def test_add_requires_exactly_one_transport(self):
        rc, _, err = _capture(cfgcmd._run_add, _add_args("bad"))  # neither
        self.assertEqual(rc, 2)
        rc2, _, _ = _capture(cfgcmd._run_add,
                             _add_args("bad", server_command="x", url="http://y"))
        self.assertEqual(rc2, 2)

    def test_add_bad_env_pair(self):
        rc, _, err = _capture(cfgcmd._run_add,
                             _add_args("e", server_command="x", env=["NOEQUALS"]))
        self.assertEqual(rc, 2)
        self.assertIn("--env", err)

    def test_add_dup_needs_force(self):
        _capture(cfgcmd._run_add, _add_args("a", server_command="x"))
        rc, _, _ = _capture(cfgcmd._run_add, _add_args("a", server_command="y"))
        self.assertEqual(rc, 2)
        rc2, _, _ = _capture(cfgcmd._run_add, _add_args("a", server_command="y", force=True))
        self.assertEqual(rc2, 0)
        self.assertEqual(uc.mcp_get(uc.load_config(), "a")["command"], "y")

    def test_remove_unknown_lists_available(self):
        _capture(cfgcmd._run_add, _add_args("a", server_command="x"))
        rc, _, err = _capture(cfgcmd._run_remove, argparse.Namespace(name="nope"))
        self.assertEqual(rc, 2)
        self.assertIn("a", err)

    def test_set_get_unset_typed(self):
        rc, _, _ = _capture(cfgcmd._run_set,
                            argparse.Namespace(key="defaults.max_spend", value="0.5"))
        self.assertEqual(rc, 0)
        # stored as a JSON number, not a string
        self.assertEqual(uc.get_value(uc.load_config(), "defaults.max_spend"), 0.5)
        rc, out, _ = _capture(cfgcmd._run_get,
                             argparse.Namespace(key="defaults.max_spend"))
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), "0.5")
        # a bareword stays a string
        _capture(cfgcmd._run_set,
                 argparse.Namespace(key="defaults.chat.model", value="llama-3.3-70b"))
        self.assertEqual(uc.get_value(uc.load_config(), "defaults.chat.model"), "llama-3.3-70b")
        rc, _, _ = _capture(cfgcmd._run_unset,
                            argparse.Namespace(key="defaults.chat.model"))
        self.assertEqual(rc, 0)

    def test_get_missing_exit_2(self):
        rc, _, _ = _capture(cfgcmd._run_get, argparse.Namespace(key="defaults.nope"))
        self.assertEqual(rc, 2)

    def test_set_on_corrupt_file_refuses(self):
        self.cfg_dir.mkdir(parents=True)
        self.cfg_file.write_text("{bad")
        rc, _, err = _capture(cfgcmd._run_set,
                             argparse.Namespace(key="defaults.x", value="1"))
        self.assertEqual(rc, 2)
        self.assertEqual(self.cfg_file.read_text(), "{bad")  # not clobbered


# --------------------------------------------------------------------------- #
# dispatch through cli.main (nested subparser wiring)
# --------------------------------------------------------------------------- #
class TestDispatch(_Base):
    def test_bare_config_prints_help_exit_2(self):
        rc, _, err = _capture(cli.main, ["config"])
        self.assertEqual(rc, 2)
        self.assertIn("ACTION", err)

    def test_list_reaches_handler(self):
        rc, out, _ = _capture(cli.main, ["config", "list"])
        self.assertEqual(rc, 0)
        self.assertIn("no MCP servers", out)

    def test_add_via_main_persists(self):
        rc, _, _ = _capture(cli.main,
                            ["config", "add", "venice", "--command", "venice", "--arg", "mcp-serve"])
        self.assertEqual(rc, 0)
        self.assertEqual(uc.mcp_get(uc.load_config(), "venice"),
                         {"command": "venice", "args": ["mcp-serve"]})


if __name__ == "__main__":
    unittest.main()
