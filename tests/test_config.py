"""Tests for `venice config` (userconfig I/O + the config subcommands).

Points config.CONFIG_DIR/CONFIG_FILE at a tmpdir so nothing touches the real
~/.config/venice. The API key is never written here.
"""
import argparse
import contextlib
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
from venice import audio_post, cli, image_montage
from venice.commands import config as cfgcmd
from venice.commands import (
    balance, bg_remove, chat, code, contact_sheet, image, image_edit, index,
    master, music, sfx, tts, upscale, video,
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

    def test_apply_fills_chat_persona(self):
        # #68: defaults.chat.persona is a plain-string key like system/character.
        doc = {"defaults": {"chat": {"persona": "pirate"}}}
        args = argparse.Namespace(persona=None, model=None, system=None)
        uc.apply_defaults(args, "chat", doc)
        self.assertEqual(args.persona, "pirate")
        args2 = argparse.Namespace(persona="cli", model=None, system=None)
        uc.apply_defaults(args2, "chat", doc)
        self.assertEqual(args2.persona, "cli")  # explicit wins

    def test_chat_parser_has_persona_dest_config_fills_it(self):
        # Guards the config key against the real argparser's dest (#68): a wrong
        # dest name would silently no-op.
        parser = _build_parser(chat)
        args = parser.parse_args(["chat"])
        self.assertTrue(hasattr(args, "persona"))
        doc = {"defaults": {"chat": {"persona": "pirate"}}}
        uc.apply_defaults(args, "chat", doc)
        self.assertEqual(args.persona, "pirate")

    def test_apply_fills_code_spawn_max_spend(self):
        # #52: defaults.code.spawn_max_spend backs --spawn-max-spend (float, None-only).
        doc = {"defaults": {"code": {"spawn_max_spend": 1.25}}}
        args = argparse.Namespace(spawn_max_spend=None, model=None, system=None)
        uc.apply_defaults(args, "code", doc)
        self.assertEqual(args.spawn_max_spend, 1.25)
        args2 = argparse.Namespace(spawn_max_spend=0.5, model=None, system=None)
        uc.apply_defaults(args2, "code", doc)
        self.assertEqual(args2.spawn_max_spend, 0.5)  # explicit wins

    def test_code_parser_has_spawn_max_spend_dest_config_fills_it(self):
        # Guards the config key against the real argparser's dest: a wrong dest name
        # (or a missing flag) would silently no-op the config backing.
        parser = _build_parser(code)
        args = parser.parse_args(["code", "do x"])
        self.assertTrue(hasattr(args, "spawn_max_spend"))
        self.assertIsNone(args.spawn_max_spend)
        doc = {"defaults": {"code": {"spawn_max_spend": 3.0}}}
        uc.apply_defaults(args, "code", doc)
        self.assertEqual(args.spawn_max_spend, 3.0)

    def test_apply_fills_code_subagent_max_tokens(self):
        # #52: defaults.code.subagent_max_tokens backs --subagent-max-tokens (int, None-only).
        doc = {"defaults": {"code": {"subagent_max_tokens": "5000"}}}  # int-coerced
        args = argparse.Namespace(subagent_max_tokens=None, model=None, system=None)
        uc.apply_defaults(args, "code", doc)
        self.assertEqual(args.subagent_max_tokens, 5000)
        args2 = argparse.Namespace(subagent_max_tokens=200, model=None, system=None)
        uc.apply_defaults(args2, "code", doc)
        self.assertEqual(args2.subagent_max_tokens, 200)  # explicit wins

    def test_code_parser_has_subagent_max_tokens_dest_config_fills_it(self):
        parser = _build_parser(code)
        args = parser.parse_args(["code", "do x"])
        self.assertTrue(hasattr(args, "subagent_max_tokens"))
        self.assertIsNone(args.subagent_max_tokens)        # default off
        doc = {"defaults": {"code": {"subagent_max_tokens": 8000}}}
        uc.apply_defaults(args, "code", doc)
        self.assertEqual(args.subagent_max_tokens, 8000)

    def test_apply_fills_code_parallel(self):
        # #52: defaults.code.parallel (_as_bool) backs --parallel; explicit wins.
        doc = {"defaults": {"code": {"parallel": True}}}
        args = argparse.Namespace(parallel=None, model=None, system=None)
        uc.apply_defaults(args, "code", doc)
        self.assertIs(args.parallel, True)
        args2 = argparse.Namespace(parallel=False, model=None, system=None)
        uc.apply_defaults(args2, "code", doc)
        self.assertIs(args2.parallel, False)  # explicit wins

    def test_code_parser_has_parallel_dest_config_fills_it(self):
        parser = _build_parser(code)
        args = parser.parse_args(["code", "do x"])
        self.assertTrue(hasattr(args, "parallel"))
        self.assertIsNone(args.parallel)
        doc = {"defaults": {"code": {"parallel": True}}}
        uc.apply_defaults(args, "code", doc)
        self.assertIs(args.parallel, True)

    def test_apply_fills_code_web_search(self):
        # #77: defaults.code.web_search (_as_bool) backs --web-search; explicit wins.
        doc = {"defaults": {"code": {"web_search": True}}}
        args = argparse.Namespace(web_search=None, model=None, system=None)
        uc.apply_defaults(args, "code", doc)
        self.assertIs(args.web_search, True)
        args2 = argparse.Namespace(web_search=False, model=None, system=None)
        uc.apply_defaults(args2, "code", doc)
        self.assertIs(args2.web_search, False)  # explicit wins

    def test_code_parser_has_web_search_dests_config_fills_them(self):
        # #77: guard both config keys against the real parser's dests.
        parser = _build_parser(code)
        args = parser.parse_args(["code", "do x"])
        self.assertTrue(hasattr(args, "web_search"))
        self.assertIsNone(args.web_search)
        self.assertTrue(hasattr(args, "web_search_model"))
        self.assertIsNone(args.web_search_model)
        doc = {"defaults": {"code": {"web_search": True,
                                     "web_search_model": "web-model-x"}}}
        uc.apply_defaults(args, "code", doc)
        self.assertIs(args.web_search, True)
        self.assertEqual(args.web_search_model, "web-model-x")

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

    def test_apply_compact_defaults_chat_and_code(self):
        doc = {"defaults": {
            "chat": {"auto_compact": True, "compact_threshold": 80000},
            "code": {"auto_compact": "yes", "compact_keep_turns": 6},
        }}
        chat_args = argparse.Namespace(auto_compact=None, compact_threshold=None,
                                       compact_keep_turns=None)
        uc.apply_defaults(chat_args, "chat", doc)
        self.assertIs(chat_args.auto_compact, True)
        self.assertEqual(chat_args.compact_threshold, 80000)
        self.assertIsNone(chat_args.compact_keep_turns)  # unset for chat

        code_args = argparse.Namespace(auto_compact=None, compact_threshold=None,
                                       compact_keep_turns=None)
        uc.apply_defaults(code_args, "code", doc)
        self.assertIs(code_args.auto_compact, True)      # coerced from "yes"
        self.assertEqual(code_args.compact_keep_turns, 6)
        self.assertIsNone(code_args.compact_threshold)

    def test_apply_session_max_spend_chat_and_code(self):
        doc = {"defaults": {
            "chat": {"session_max_spend": 1.5},
            "code": {"session_max_spend": "2.25"},
        }}
        chat_args = argparse.Namespace(session_max_spend=None)
        uc.apply_defaults(chat_args, "chat", doc)
        self.assertEqual(chat_args.session_max_spend, 1.5)
        code_args = argparse.Namespace(session_max_spend=None)
        uc.apply_defaults(code_args, "code", doc)
        self.assertEqual(code_args.session_max_spend, 2.25)  # coerced from str
        # explicit CLI value always wins
        cli = argparse.Namespace(session_max_spend=9.0)
        uc.apply_defaults(cli, "chat", doc)
        self.assertEqual(cli.session_max_spend, 9.0)


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


def _build_status_parser(mod):
    """The `-status` half of a command. `_build_parser` registers only the
    generate parser, but a `-status` parser shares its parent's config section,
    so parity has to be asserted on both."""
    parser = argparse.ArgumentParser(prog="venice")
    mod.register_status(parser.add_subparsers(dest="command"))
    return parser


# The real resolver, not a copy of it: a hand-rebuilt mapping would silently
# exempt any layer added later -- which is how the mastering chain escaped the
# pre-C2 `_COMMAND_MAP`-only sweep in the first place. (#57 C2)
_rows_for = uc.rows_for


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

    def test_balance_min_is_config_backable(self):
        """#57 Class D: `venice balance` had no `apply_defaults` call at all, so
        even a pure-Class-A dest like `--min` (already default None) was
        unreachable. `--json`/`--verbose` stay CLI-only on purpose."""
        args = _build_parser(balance).parse_args(["balance"])
        uc.apply_defaults(args, "balance", {"defaults": {"balance": {"min": "5"}}})
        self.assertEqual(args.min, 5.0)
        args = _build_parser(balance).parse_args(["balance", "--min", "10"])
        uc.apply_defaults(args, "balance", {"defaults": {"balance": {"min": "5"}}})
        self.assertEqual(args.min, 10.0)

    def test_balance_output_modes_stay_cli_only(self):
        dests = {d for d, _ in uc._COMMAND_MAP["balance"].values()}
        self.assertEqual(dests, {"min"})

    def test_index_exclude_scalar_becomes_list(self):
        parser = _build_parser(index)
        args = parser.parse_args(["index"])
        doc = {"defaults": {"index": {"exclude": "solo-pat"}}}
        uc.apply_defaults(args, "index", doc)
        self.assertEqual(args.exclude, ["solo-pat"])


# --------------------------------------------------------------------------- #
# #57 config parity -- Class B: `store_true` flags tri-stated to default=None so
# `apply_defaults` (which only fills a dest still None) can reach them. Each case
# parses the command's REAL parser, so a wrong dest -- or an option string that
# collides with an existing flag -- fails here rather than at runtime.
#
# `on`/`off` are the two spellings that must produce True/False. Positive-sense
# flags use BooleanOptionalAction (--enhance/--no-enhance); negative-sense dests
# keep their name and gain a positive counterpart (--no-audio/--with-audio), so
# the config key stays honest and no tool-argument name changes.
# --------------------------------------------------------------------------- #
_CLASS_B_CASES = [
    dict(key="image_edit", mod=image_edit, argv=["image-edit", "p"],
         dest="safe_mode", on="--safe-mode", off="--no-safe-mode",
         cfg=False, want=False, explicit="--safe-mode", eval=True),
    dict(key="upscale", mod=upscale, argv=["upscale", "in.png"],
         dest="enhance", on="--enhance", off="--no-enhance",
         cfg="yes", want=True, explicit="--no-enhance", eval=False),
    dict(key="video", mod=video, argv=["video", "p"],
         dest="no_audio", on="--no-audio", off="--with-audio",
         cfg=True, want=True, explicit="--with-audio", eval=False),
    dict(key="video", mod=video, argv=["video", "p"],
         dest="no_cleanup", on="--no-cleanup", off="--cleanup",
         cfg=True, want=True, explicit="--cleanup", eval=False),
    dict(key="music", mod=music, argv=["music", "p"],
         dest="instrumental", on="--instrumental", off="--no-instrumental",
         cfg="true", want=True, explicit="--no-instrumental", eval=False),
    dict(key="music", mod=music, argv=["music", "p"],
         dest="master", on="--master", off="--no-master",
         cfg=True, want=True, explicit="--no-master", eval=False),
    dict(key="music", mod=music, argv=["music", "p"],
         dest="loop", on="--loop", off="--no-loop",
         cfg=True, want=True, explicit="--no-loop", eval=False),
    dict(key="music", mod=music, argv=["music", "p"],
         dest="no_cleanup", on="--no-cleanup", off="--cleanup",
         cfg=True, want=True, explicit="--cleanup", eval=False),
    dict(key="sfx", mod=sfx, argv=["sfx", "p"],
         dest="master", on="--master", off="--no-master",
         cfg=True, want=True, explicit="--no-master", eval=False),
    dict(key="sfx", mod=sfx, argv=["sfx", "p"],
         dest="loop", on="--loop", off="--no-loop",
         cfg="on", want=True, explicit="--no-loop", eval=False),
    dict(key="sfx", mod=sfx, argv=["sfx", "p"],
         dest="no_cleanup", on="--no-cleanup", off="--cleanup",
         cfg=True, want=True, explicit="--cleanup", eval=False),
    dict(key="master", mod=master, argv=["master", "in.wav"],
         dest="loop", on="--loop", off="--no-loop",
         cfg=True, want=True, explicit="--no-loop", eval=False),
    # #57 Class D: contact-sheet's --label was a bare store_true.
    dict(key="contact_sheet", mod=contact_sheet, argv=["contact-sheet", "."],
         dest="label", on="--label", off="--no-label",
         cfg=True, want=True, explicit="--no-label", eval=False),
]

# Every command carrying the tri-stated `--no-balance`/`--balance` pair. It lives
# in `_GLOBAL_MAP`, not eight sections, so it gets its own cases below.
_BALANCE_MODS = [
    ("sfx", sfx, ["sfx", "p"]), ("music", music, ["music", "p"]),
    ("video", video, ["video", "p"]), ("tts", tts, ["tts", "hello"]),
    ("image", image, ["image", "p"]), ("image_edit", image_edit, ["image-edit", "p"]),
    ("upscale", upscale, ["upscale", "in.png"]),
    ("bg_remove", bg_remove, ["bg-remove", "in.png"]),
]


class TestClassBParity(unittest.TestCase):
    def test_flag_is_tristate(self):
        """Unset -> None (so config can fill it), on -> True, off -> False."""
        for case in _CLASS_B_CASES:
            with self.subTest(cmd=case["key"], dest=case["dest"]):
                parser = _build_parser(case["mod"])
                got = tuple(
                    getattr(parser.parse_args(case["argv"] + extra), case["dest"])
                    for extra in ([], [case["on"]], [case["off"]])
                )
                self.assertEqual(got, (None, True, False))

    def test_config_fills_none_dest(self):
        for case in _CLASS_B_CASES:
            with self.subTest(cmd=case["key"], dest=case["dest"]):
                parser = _build_parser(case["mod"])
                args = parser.parse_args(case["argv"])
                doc = {"defaults": {case["key"]: {case["dest"]: case["cfg"]}}}
                uc.apply_defaults(args, case["key"], doc)
                self.assertIs(getattr(args, case["dest"]), case["want"])

    def test_explicit_cli_beats_config(self):
        for case in _CLASS_B_CASES:
            with self.subTest(cmd=case["key"], dest=case["dest"]):
                parser = _build_parser(case["mod"])
                args = parser.parse_args(case["argv"] + [case["explicit"]])
                doc = {"defaults": {case["key"]: {case["dest"]: case["cfg"]}}}
                uc.apply_defaults(args, case["key"], doc)
                self.assertIs(getattr(args, case["dest"]), case["eval"])

    # -- the `--no-balance` global ------------------------------------------ #
    def test_no_balance_tristate_on_every_spend_command(self):
        for key, mod, argv in _BALANCE_MODS:
            with self.subTest(cmd=key):
                parser = _build_parser(mod)
                got = tuple(
                    parser.parse_args(argv + extra).no_balance
                    for extra in ([], ["--no-balance"], ["--show-balance"])
                )
                self.assertEqual(got, (None, True, False))

    def test_no_balance_global_fills_every_spend_command(self):
        doc = {"defaults": {"no_balance": True}}
        for key, mod, argv in _BALANCE_MODS:
            with self.subTest(cmd=key):
                args = _build_parser(mod).parse_args(argv)
                uc.apply_defaults(args, key, doc)
                self.assertIs(args.no_balance, True)

    def test_explicit_balance_beats_the_global(self):
        doc = {"defaults": {"no_balance": True}}
        args = _build_parser(image).parse_args(["image", "p", "--show-balance"])
        uc.apply_defaults(args, "image", doc)
        self.assertIs(args.no_balance, False)

    def test_per_command_no_balance_overrides_the_global(self):
        doc = {"defaults": {"no_balance": True, "image": {"no_balance": False}}}
        args = _build_parser(image).parse_args(["image", "p"])
        uc.apply_defaults(args, "image", doc)
        self.assertIs(args.no_balance, False)

    def test_global_is_skipped_for_a_command_without_the_flag(self):
        """The `hasattr` guard is what makes a _GLOBAL_MAP row safe -- `chat` has
        no --no-balance, and must not sprout the attribute."""
        args = argparse.Namespace(model=None)
        uc.apply_defaults(args, "chat", {"defaults": {"no_balance": True}})
        self.assertFalse(hasattr(args, "no_balance"))

    def test_bg_remove_gets_globals_without_a_section(self):
        """bg_remove calls apply_defaults but has no _COMMAND_MAP section; the
        globals must still reach it."""
        self.assertNotIn("bg_remove", uc._COMMAND_MAP)
        args = _build_parser(bg_remove).parse_args(["bg-remove", "in.png"])
        uc.apply_defaults(args, "bg_remove", {"defaults": {"no_balance": True}})
        self.assertIs(args.no_balance, True)

    def test_master_section_has_no_master_key(self):
        """`venice master` registers the mastering flags with include_toggle=False,
        so its namespace has no `master` attr -- a key would be dead config."""
        self.assertEqual(set(uc._COMMAND_MAP["master"]), {"loop"})
        args = _build_parser(master).parse_args(["master", "in.wav"])
        self.assertFalse(hasattr(args, "master"))

    def test_defaults_master_section_does_not_leak_into_sfx_master(self):
        """`defaults.master` is a command SECTION while `defaults.sfx.master` is a
        KEY. resolve_default must keep the global lookup from mistaking the
        section for a scalar -- in BOTH shapes."""
        doc = {"defaults": {"master": {"loop": True}}}
        args = _build_parser(sfx).parse_args(["sfx", "p"])
        uc.apply_defaults(args, "sfx", doc)
        self.assertIsNone(args.master)  # not True, not a dict
        self.assertIsNone(args.loop)    # defaults.master.loop is master's, not sfx's

    def test_scalar_defaults_master_is_not_a_global_for_the_master_flag(self):
        """The dict shape is guarded by an isinstance check, but a SCALAR
        `defaults.master = true` used to fall through as a global and silently
        enable mastering on every sfx/music render (while making
        defaults.master.loop unreachable). A key naming a command section is
        never a global scalar."""
        doc = {"defaults": {"master": True}}
        for mod, argv, key in ((sfx, ["sfx", "p"], "sfx"),
                               (music, ["music", "p"], "music")):
            with self.subTest(cmd=key):
                args = _build_parser(mod).parse_args(argv)
                uc.apply_defaults(args, key, doc)
                self.assertIsNone(args.master)

    def test_ba_abbreviation_still_resolves_to_background(self):
        """`--show-balance`, not `--balance`: a bare `--balance` would make the
        abbreviation `--ba` ambiguous against the pre-existing `--background`."""
        for mod, argv in ((sfx, ["sfx", "p"]), (music, ["music", "p"]),
                          (video, ["video", "p"])):
            with self.subTest(cmd=argv[0]):
                args = _build_parser(mod).parse_args(argv + ["--ba"])
                self.assertTrue(args.background)

    def test_status_play_is_tristate_and_config_backable(self):
        """The parent section must reach BOTH halves of a command, not just the
        generate parser -- `--play` was left a bare store_true at first."""
        for mod, argv, key in ((sfx, ["sfx-status", "j1"], "sfx"),
                               (music, ["music-status", "j1"], "music")):
            with self.subTest(cmd=argv[0]):
                got = tuple(
                    _build_status_parser(mod).parse_args(argv + extra).play
                    for extra in ([], ["--play"], ["--no-play"])
                )
                self.assertEqual(got, (None, True, False))
                args = _build_status_parser(mod).parse_args(argv)
                uc.apply_defaults(args, key, {"defaults": {key: {"play": True}}})
                self.assertIs(args.play, True)


# --------------------------------------------------------------------------- #
# #57 config parity -- Class C: flags whose argparse default was a hardcoded
# VALUE (`--model venice-sd35`, `--variants 1`, `--scale 2.0`). Same blocker as
# Class B: `apply_defaults` only fills a dest that is still None, so the literal
# had to move off the parser. It now lives in a `userconfig.apply_literals` call
# in each handler, which runs AFTER `apply_defaults` -- giving the full ladder:
#
#     explicit CLI flag > defaults.<cmd>.<key> > built-in literal
#
# `literal` is the value the command falls back to when nothing is set; it is
# asserted against the module constant, not a copy of the string, so moving a
# default value doesn't quietly desync the test from the code.
# --------------------------------------------------------------------------- #
_CLASS_C_CASES = [
    dict(key="image", mod=image, argv=["image", "p"], dest="model",
         literal=image.DEFAULT_IMAGE_MODEL, cfg="hidream", want="hidream",
         explicit=["--model", "cli-m"], eval="cli-m"),
    dict(key="image", mod=image, argv=["image", "p"], dest="format",
         literal=image.DEFAULT_FORMAT, cfg="webp", want="webp",
         explicit=["--format", "jpeg"], eval="jpeg"),
    dict(key="image", mod=image, argv=["image", "p"], dest="variants",
         literal=image.DEFAULT_VARIANTS, cfg="3", want=3,
         explicit=["--variants", "2"], eval=2),
    # cfg/explicit must be DISTINCT legal values, and both distinct from the
    # literal -- otherwise test_explicit_cli_beats_config asserts nothing.
    dict(key="tts", mod=tts, argv=["tts", "hi"], dest="model",
         literal=tts.DEFAULT_TTS_MODEL, cfg="tts-orpheus", want="tts-orpheus",
         explicit=["--model", "tts-xai-v1"], eval="tts-xai-v1"),
    dict(key="tts", mod=tts, argv=["tts", "hi"], dest="format",
         literal=tts.DEFAULT_FORMAT, cfg="wav", want="wav",
         explicit=["--format", "flac"], eval="flac"),
    dict(key="sfx", mod=sfx, argv=["sfx", "p"], dest="model",
         literal=sfx.DEFAULT_SFX_MODEL, cfg="mmaudio-v2-text-to-audio",
         want="mmaudio-v2-text-to-audio",
         explicit=["--model", "elevenlabs-sound-effects-v2"],
         eval="elevenlabs-sound-effects-v2"),
    dict(key="sfx", mod=sfx, argv=["sfx", "p"], dest="duration",
         literal=sfx.DEFAULT_DURATION, cfg="12", want=12,
         explicit=["--duration", "7"], eval=7),
    dict(key="music", mod=music, argv=["music", "p"], dest="model",
         literal=music.DEFAULT_MUSIC_MODEL, cfg="other-music", want="other-music",
         explicit=["--model", "cli-mu"], eval="cli-mu"),
    dict(key="video", mod=video, argv=["video", "p"], dest="duration",
         literal=video.DEFAULT_VIDEO_DURATION, cfg="10s", want="10s",
         explicit=["--duration", "8s"], eval="8s"),
    dict(key="upscale", mod=upscale, argv=["upscale", "in.png"], dest="scale",
         literal=upscale.DEFAULT_SCALE, cfg="3", want=3.0,
         explicit=["--scale", "4"], eval=4.0),

    # #57 Class C2 -- the shared mastering chain. These five live in
    # `_GLOBAL_MAP`, not a section, because `audio_post.add_master_flags` puts
    # one chain on three commands. The rows below exercise the SECTION-override
    # half (`defaults.sfx.lufs`); `TestMasterChainGlobals` covers the bare
    # global and proves the two layers compose.
    dict(key="master", mod=master, argv=["master", "in.wav"], dest="lufs",
         literal=audio_post.DEFAULT_LUFS, cfg="-14", want=-14.0,
         explicit=["--lufs", "-9"], eval=-9.0),
    dict(key="sfx", mod=sfx, argv=["sfx", "p"], dest="true_peak",
         literal=audio_post.DEFAULT_TRUE_PEAK, cfg="-1.5", want=-1.5,
         explicit=["--true-peak", "-2"], eval=-2.0),
    dict(key="music", mod=music, argv=["music", "p"], dest="sample_rate",
         literal=audio_post.DEFAULT_SAMPLE_RATE, cfg="44100", want=44100,
         explicit=["--sample-rate", "96000"], eval=96000),
    dict(key="master", mod=master, argv=["master", "in.wav"], dest="bit_depth",
         literal=audio_post.DEFAULT_BIT_DEPTH, cfg="16", want=16,
         explicit=["--bit-depth", "32"], eval=32),
    dict(key="sfx", mod=sfx, argv=["sfx", "p"], dest="loop_crossfade",
         literal=audio_post.DEFAULT_LOOP_CROSSFADE, cfg="3", want=3.0,
         explicit=["--loop-crossfade", "4"], eval=4.0),

    # #57 Class C2 -- poll cadence. Sections, not globals: video deliberately
    # polls slower, so the literals differ per command.
    dict(key="sfx", mod=sfx, argv=["sfx", "p"], dest="poll_interval",
         literal=cfg.SFX_POLL_INTERVAL_SEC, cfg="0.5", want=0.5,
         explicit=["--poll-interval", "1.5"], eval=1.5),
    dict(key="sfx", mod=sfx, argv=["sfx", "p"], dest="max_wait",
         literal=cfg.SFX_POLL_MAX_WAIT_SEC, cfg="60", want=60.0,
         explicit=["--max-wait", "120"], eval=120.0),
    dict(key="music", mod=music, argv=["music", "p"], dest="poll_interval",
         literal=cfg.MUSIC_POLL_INTERVAL_SEC, cfg="0.25", want=0.25,
         explicit=["--poll-interval", "3.5"], eval=3.5),
    dict(key="music", mod=music, argv=["music", "p"], dest="max_wait",
         literal=cfg.MUSIC_POLL_MAX_WAIT_SEC, cfg="45", want=45.0,
         explicit=["--max-wait", "90"], eval=90.0),
    dict(key="video", mod=video, argv=["video", "p"], dest="poll_interval",
         literal=cfg.VIDEO_POLL_INTERVAL_SEC, cfg="2.5", want=2.5,
         explicit=["--poll-interval", "7.5"], eval=7.5),
    dict(key="video", mod=video, argv=["video", "p"], dest="max_wait",
         literal=cfg.VIDEO_POLL_MAX_WAIT_SEC, cfg="120", want=120.0,
         explicit=["--max-wait", "600"], eval=600.0),

    # #57 Class D -- contact-sheet's grid knobs.
    dict(key="contact_sheet", mod=contact_sheet, argv=["contact-sheet", "."],
         dest="cols", literal=image_montage.DEFAULT_COLS, cfg="6", want=6,
         explicit=["--cols", "3"], eval=3),
    dict(key="contact_sheet", mod=contact_sheet, argv=["contact-sheet", "."],
         dest="cell", literal=image_montage.DEFAULT_CELL, cfg="512x512",
         want="512x512", explicit=["--cell", "128x128"], eval="128x128"),
    dict(key="contact_sheet", mod=contact_sheet, argv=["contact-sheet", "."],
         dest="background", literal=image_montage.DEFAULT_BACKGROUND,
         cfg="black", want="black",
         explicit=["--background", "#202020"], eval="#202020"),
    dict(key="contact_sheet", mod=contact_sheet, argv=["contact-sheet", "."],
         dest="padding", literal=image_montage.DEFAULT_PADDING, cfg="8", want=8,
         explicit=["--padding", "2"], eval=2),
    dict(key="contact_sheet", mod=contact_sheet, argv=["contact-sheet", "."],
         dest="engine", literal=image_montage.DEFAULT_ENGINE, cfg="montage",
         want="montage", explicit=["--engine", "ffmpeg"], eval="ffmpeg"),
]


class TestClassCParity(unittest.TestCase):

    def test_parser_default_is_none(self):
        """The load-bearing one: this IS what "the literal moved off the parser"
        means. A concrete default here makes the dest unreachable from config,
        because `apply_defaults` only fills a dest that is still None."""
        for case in _CLASS_C_CASES:
            with self.subTest(cmd=case["key"], dest=case["dest"]):
                args = _build_parser(case["mod"]).parse_args(case["argv"])
                self.assertIsNone(getattr(args, case["dest"]),
                                  msg=f"{case['key']}.{case['dest']}")

    def test_config_fills_none_dest(self):
        for case in _CLASS_C_CASES:
            with self.subTest(cmd=case["key"], dest=case["dest"]):
                args = _build_parser(case["mod"]).parse_args(case["argv"])
                doc = {"defaults": {case["key"]: {case["dest"]: case["cfg"]}}}
                uc.apply_defaults(args, case["key"], doc)
                self.assertEqual(getattr(args, case["dest"]), case["want"],
                                 msg=f"{case['key']}.{case['dest']}")

    def test_explicit_cli_beats_config(self):
        for case in _CLASS_C_CASES:
            with self.subTest(cmd=case["key"], dest=case["dest"]):
                args = _build_parser(case["mod"]).parse_args(
                    case["argv"] + case["explicit"])
                doc = {"defaults": {case["key"]: {case["dest"]: case["cfg"]}}}
                uc.apply_defaults(args, case["key"], doc)
                self.assertEqual(getattr(args, case["dest"]), case["eval"],
                                 msg=f"{case['key']}.{case['dest']}")

    def test_literal_applies_when_nothing_is_set(self):
        """No flag, no config -> the built-in literal. This is the contract that
        keeps the UX unchanged for everyone who never opens config.json."""
        for case in _CLASS_C_CASES:
            with self.subTest(cmd=case["key"], dest=case["dest"]):
                args = _build_parser(case["mod"]).parse_args(case["argv"])
                uc.apply_defaults(args, case["key"], {})
                uc.apply_literals(args, **{case["dest"]: case["literal"]})
                self.assertEqual(getattr(args, case["dest"]), case["literal"],
                                 msg=f"{case['key']}.{case['dest']}")

    def test_every_class_c_dest_has_a_command_map_row(self):
        """Relocating a default without adding the config row leaves the flag
        unreachable from BOTH layers -- it silently stops being configurable and
        never becomes so. Pins the half-done state."""
        for case in _CLASS_C_CASES:
            with self.subTest(cmd=case["key"], dest=case["dest"]):
                dests = {d for d, _ in _rows_for(case["key"]).values()}
                self.assertIn(case["dest"], dests,
                              msg=f"no config row fills {case['key']}."
                                  f"{case['dest']}")

    def test_class_c_rows_are_not_vacuous(self):
        """Each parity assertion must be able to FAIL. v0.72's review found rows
        seeded with the same value on both sides of a precedence check, so the
        test held no matter which layer won; this mechanizes the lesson.

        Two constraints, one per assertion:
          - `want != literal`, or `test_config_fills_none_dest` passes even if
            config is ignored and the literal is what lands.
          - `eval != want`, or `test_explicit_cli_beats_config` passes even if
            config beats the command line.

        `eval == literal` is deliberately allowed: a flag with only two legal
        choices (`sfx --model`) cannot offer a third value, and the assertion
        still discriminates, because a broken precedence would yield `want`.
        """
        for case in _CLASS_C_CASES:
            with self.subTest(cmd=case["key"], dest=case["dest"]):
                self.assertNotEqual(
                    case["want"], case["literal"],
                    msg=f"{case['key']}.{case['dest']}: cfg value equals the "
                        "built-in literal -- test_config_fills_none_dest cannot fail")
                self.assertNotEqual(
                    case["eval"], case["want"],
                    msg=f"{case['key']}.{case['dest']}: explicit value equals the "
                        "cfg value -- test_explicit_cli_beats_config cannot fail")

    def test_status_model_is_not_config_backable(self):
        """`sfx-status`/`music-status` --model is job IDENTITY, not a preference:
        it goes in the /audio/retrieve and /audio/complete bodies. A config value
        reaching it would retarget an already-queued, already-charged job. The
        concrete argparse default is what makes `apply_defaults` skip the dest,
        so "tidying" it to None would silently reintroduce that bug."""
        for mod, argv, key, literal in (
            (sfx, ["sfx-status", "j1"], "sfx", sfx.DEFAULT_SFX_MODEL),
            (music, ["music-status", "j1"], "music", music.DEFAULT_MUSIC_MODEL),
        ):
            with self.subTest(cmd=key):
                args = _build_status_parser(mod).parse_args(argv)
                self.assertEqual(args.model, literal)  # NOT None
                # ...while the two flags registered right beside it DID go
                # None in Class C2, because cadence IS a preference. Asserting
                # both here turns the "do not make these consistent" comment in
                # `register_status` into something CI enforces.
                self.assertIsNone(args.poll_interval)
                self.assertIsNone(args.max_wait)
                uc.apply_defaults(
                    args, key, {"defaults": {key: {"model": "retargeted"}}})
                self.assertEqual(args.model, literal,
                                 msg="config retargeted a queued job")

    def test_invalid_choice_in_config_is_warned_and_skipped(self):
        """A config key whose flag has `choices=` must not be able to do what the
        command line can't. Without `_one_of`, `defaults.sfx.model = "bogus"`
        reaches SFX_MODELS[model] as a raw KeyError traceback."""
        cases = [
            ("image", image, ["image", "p"], "format", "gif"),
            ("tts", tts, ["tts", "hi"], "model", "tts-nope"),
            ("tts", tts, ["tts", "hi"], "format", "ogg"),
            ("sfx", sfx, ["sfx", "p"], "model", "bogus"),
            ("video", video, ["video", "p"], "duration", "99s"),
            # #57 C2/D: the first int choice set, and the first choices row
            # reached through _GLOBAL_MAP rather than a section.
            ("master", master, ["master", "in.wav"], "bit_depth", 8),
            ("contact_sheet", contact_sheet, ["contact-sheet", "."],
             "engine", "imagemagick"),
        ]
        for key, mod, argv, dest, bad in cases:
            with self.subTest(cmd=key, dest=dest):
                args = _build_parser(mod).parse_args(argv)
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    uc.apply_defaults(
                        args, key, {"defaults": {key: {dest: bad}}})
                self.assertIsNone(getattr(args, dest),
                                  msg="a bad choice was accepted")
                self.assertIn(f"ignoring invalid config default {dest}={bad!r}",
                              err.getvalue())

    def test_valid_choice_in_config_is_accepted(self):
        """The other half of `_one_of`: a legal value must still pass through."""
        args = _build_parser(sfx).parse_args(["sfx", "p"])
        uc.apply_defaults(
            args, "sfx", {"defaults": {"sfx": {"model": "mmaudio-v2-text-to-audio"}}})
        self.assertEqual(args.model, "mmaudio-v2-text-to-audio")

    def test_every_one_of_row_resolves_to_a_real_collection(self):
        """`_one_of` names its allowed set by (module, attr) STRINGS, because a
        module-scope import would cycle. Strings are invisible to a rename, and a
        stale one raises ImportError/AttributeError -- which `apply_defaults` and
        `config_defaults_for` do NOT catch (only TypeError/ValueError), breaking
        userconfig's "degrade, never crash" contract. Resolving every row here
        turns that runtime traceback into a CI failure."""
        sections = [("_GLOBAL_MAP", uc._GLOBAL_MAP)] + list(uc._COMMAND_MAP.items())
        for section, rows in sections:
            for key, (_dest, coerce) in rows.items():
                if getattr(coerce, "__name__", "") != "coerce":
                    continue  # not a _one_of closure
                with self.subTest(section=section, key=key):
                    # A value no real choice set contains: resolving the
                    # collection is the point, ValueError is the proof it did.
                    with self.assertRaises(ValueError):
                        coerce("\x00 definitely not a valid choice")

    def test_every_choices_flag_has_a_validating_coercer(self):
        """The invariant `_one_of` exists for: config must never be able to set a
        value the command line would reject. Any `_COMMAND_MAP` row whose flag
        carries argparse `choices=` must validate against that set, or config
        silently ships a value argparse would have exited 2 on."""
        mods = {"image": image, "image_edit": image_edit, "tts": tts, "sfx": sfx,
                "music": music, "video": video, "upscale": upscale, "chat": chat,
                # #57 C2/D. `master` and `contact_sheet` matter most here: their
                # choices-carrying rows (`bit_depth`, `engine`) are the first
                # ever routed through `_GLOBAL_MAP` and the first with non-string
                # choices, so a `_COMMAND_MAP`-only sweep would exempt them.
                "master": master, "contact_sheet": contact_sheet,
                "balance": balance}
        for section, mod in mods.items():
            parser = _build_parser(mod)
            sub = list(parser._subparsers._group_actions[0].choices.values())[0]
            by_dest = {a.dest: a for a in sub._actions}
            for key, (dest, coerce) in _rows_for(section).items():
                action = by_dest.get(dest)
                if action is None or not action.choices:
                    continue
                with self.subTest(section=section, key=key):
                    self.assertEqual(
                        getattr(coerce, "__name__", ""), "coerce",
                        msg=f"defaults.{section}.{key} maps to a flag with "
                            f"choices={list(action.choices)!r} but is not "
                            "validated -- wrap it in _one_of")


# --------------------------------------------------------------------------- #
# #57 Class C2 -- the shared mastering chain as _GLOBAL_MAP rows.
#
# `audio_post.add_master_flags` registers one chain on three commands, so the
# five valued knobs are globals rather than three duplicated sections. That
# choice is only correct if two things hold, and both are asserted here: a bare
# `defaults.lufs` must reach all three commands, and `defaults.<cmd>.lufs` must
# still override it -- which `resolve_default` gives for free by checking the
# command section first.
# --------------------------------------------------------------------------- #
_MASTER_CHAIN = ("lufs", "true_peak", "sample_rate", "bit_depth", "loop_crossfade")
_MASTER_CMDS = ((master, ["master", "in.wav"], "master"),
                (sfx, ["sfx", "p"], "sfx"),
                (music, ["music", "p"], "music"))


class TestMasterChainGlobals(unittest.TestCase):
    def test_bare_global_reaches_all_three_commands(self):
        doc = {"defaults": {"lufs": -14.0, "bit_depth": 16, "sample_rate": 44100}}
        for mod, argv, key in _MASTER_CMDS:
            with self.subTest(cmd=key):
                args = _build_parser(mod).parse_args(argv)
                uc.apply_defaults(args, key, doc)
                self.assertEqual(args.lufs, -14.0)
                self.assertEqual(args.bit_depth, 16)
                self.assertEqual(args.sample_rate, 44100)

    def test_section_overrides_the_global(self):
        """The whole justification for the globals decision: per-command
        override with no `_COMMAND_MAP` row, straight out of `resolve_default`."""
        doc = {"defaults": {"lufs": -14.0, "sfx": {"lufs": -9.0}}}
        args = _build_parser(sfx).parse_args(["sfx", "p"])
        uc.apply_defaults(args, "sfx", doc)
        self.assertEqual(args.lufs, -9.0)
        args = _build_parser(music).parse_args(["music", "p"])
        uc.apply_defaults(args, "music", doc)
        self.assertEqual(args.lufs, -14.0)  # untouched by the sfx override

    def test_literals_restore_the_pre_config_defaults(self):
        """The UX contract for everyone who never opens config.json: with no
        config at all, the ladder must land on exactly the old argparse values."""
        for mod, argv, key in _MASTER_CMDS:
            with self.subTest(cmd=key):
                args = _build_parser(mod).parse_args(argv)
                uc.apply_defaults(args, key, {})
                audio_post.apply_master_literals(args)
                self.assertEqual(
                    audio_post.master_kwargs(args),
                    dict(sample_rate=48000, bit_depth=24, lufs=-16.0,
                         true_peak=-1.0, loop=False, loop_crossfade=2.0))

    def test_globals_are_skipped_on_commands_without_the_flags(self):
        """`apply_defaults`' hasattr guard is what keeps these globals from
        minting dead attributes on every other command."""
        args = _build_parser(image).parse_args(["image", "p"])
        uc.apply_defaults(args, "image", {"defaults": {"lufs": -14.0}})
        self.assertFalse(hasattr(args, "lufs"))

    def test_the_chain_is_declared_only_by_the_audio_commands(self):
        """A global reaches ANY command declaring the dest. If some future
        command grows a `--sample-rate`, it would silently inherit an audio
        mastering preference -- so pin the current blast radius."""
        parser = cli.build_parser()
        subs = parser._subparsers._group_actions[0].choices
        for dest in _MASTER_CHAIN:
            with self.subTest(dest=dest):
                have = {name for name, sp in subs.items()
                        if dest in {a.dest for a in sp._actions}}
                self.assertEqual(have, {"master", "sfx", "music"})

    def test_only_global_map_keys_act_as_top_level_scalars(self):
        """Globals are an explicit ALLOW-LIST (#57 C2). `resolve_default` used to
        fall through to a top-level scalar for any key that didn't NAME a
        section, so a bare `defaults.max_wait = 60` reached sfx, music AND video
        -- silently replacing video's deliberately-longer 900s built-in, which
        queues and CHARGES for a render and then abandons it. A bare
        `defaults.model` retargeted every command with a `model` row the same
        way. The per-command section form is unaffected."""
        for key in ("max_wait", "poll_interval", "model", "duration", "cols"):
            with self.subTest(key=key):
                self.assertIsNone(
                    uc.resolve_default("video", key, {"defaults": {key: 60}}),
                    msg=f"bare defaults.{key} still acts as a global")
        # ...while every real global still does.
        self.assertEqual(
            uc.resolve_default("video", "max_spend", {"defaults": {"max_spend": 2}}), 2)
        self.assertEqual(
            uc.resolve_default("sfx", "lufs", {"defaults": {"lufs": -14}}), -14)
        # ...and the section form is untouched.
        self.assertEqual(
            uc.resolve_default("video", "max_wait",
                               {"defaults": {"video": {"max_wait": 60}}}), 60)

    def test_video_keeps_its_longer_builtin_wait(self):
        """The concrete regression: the bare global must not shorten a video
        render's deadline, because the job is queued and paid before the first
        poll and the timeout abandons it."""
        args = _build_parser(video).parse_args(["video", "p"])
        uc.apply_defaults(args, "video", {"defaults": {"max_wait": 60}})
        self.assertIsNone(args.max_wait)  # untouched -> the 900s literal applies

    def test_bit_depth_coercer_accepts_ints(self):
        """`_one_of` was string-only: `str(24)` never matches the int member 24,
        so every legal value was rejected AND the error message itself raised
        TypeError on `', '.join`. `--bit-depth` is the tree's only int choice
        set, so nothing caught it until this row existed."""
        coerce = dict(uc._GLOBAL_MAP)["bit_depth"][1]
        self.assertEqual(coerce(16), 16)      # JSON int
        self.assertEqual(coerce("24"), 24)    # string form
        self.assertIsInstance(coerce("32"), int)
        for bad in (8, "nope", "24.5", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    coerce(bad)

    def test_int_rows_reject_a_lossy_value(self):
        """Config must never set a value the CLI would refuse. Plain `int`
        truncates -- `int(24.9) == 24` -- so `defaults.bit_depth = 24.9` would
        have mastered at 24-bit while `--bit-depth 24.9` exits 2. An integral
        float (`24.0`, which a JSON writer may emit) is still fine."""
        rows = dict(uc._GLOBAL_MAP)
        for key, good, integral_float, lossy in (
            ("bit_depth", "16", 24.0, 24.9),
            ("sample_rate", "48000", 44100.0, 44100.9),
        ):
            coerce = rows[key][1]
            with self.subTest(key=key):
                self.assertEqual(coerce(good), int(good))
                self.assertEqual(coerce(integral_float), int(integral_float))
                with self.assertRaises(ValueError):
                    coerce(lossy)
                with self.assertRaises(ValueError):
                    coerce(True)  # bool is an int subclass; never a count

    def test_bit_depths_match_the_codec_table(self):
        """The choices are derived from `_CODECS`, so a depth can never be
        offered that `master()` has no encoder for."""
        self.assertEqual(set(audio_post.BIT_DEPTHS), set(audio_post._CODECS))


# --------------------------------------------------------------------------- #
# #57 Class C2 -- poll cadence, including the `-status` half and the deliberate
# CLI-only/tool-visible split between the two flags.
# --------------------------------------------------------------------------- #
class TestPollCadenceParity(unittest.TestCase):
    def test_status_parsers_are_tristate_and_config_backable(self):
        """A `-status` parser shares its parent's section, so the cadence
        preference must reach it too -- and it needs its own literal call, or a
        config-free run leaves None in the poll loop."""
        for mod, argv, key in ((sfx, ["sfx-status", "j1"], "sfx"),
                               (music, ["music-status", "j1"], "music"),
                               (video, ["video-status", "j1"], "video")):
            with self.subTest(cmd=argv[0]):
                args = _build_status_parser(mod).parse_args(argv)
                self.assertIsNone(args.poll_interval)
                self.assertIsNone(args.max_wait)
                uc.apply_defaults(args, key, {"defaults": {
                    key: {"poll_interval": 0.5, "max_wait": 60}}})
                self.assertEqual(args.poll_interval, 0.5)
                self.assertEqual(args.max_wait, 60.0)

    def test_max_wait_reaches_the_tool_path_but_poll_interval_does_not(self):
        """The documented asymmetry: `config_defaults_for` injects only keys the
        impl accepts, and the tool impls fix their own cadence. Asserted here
        rather than in test_mcp_serve, which is skipped without the mcp extra."""
        from venice.commands import _mcp
        doc = {"defaults": {s: {"poll_interval": 9.0, "max_wait": 42}
                            for s in ("sfx", "music", "video")}}
        for section, impl in (("sfx", _mcp.sfx_tool), ("music", _mcp.music_tool),
                              ("video", _mcp.video_tool)):
            with self.subTest(section=section):
                got = uc.config_defaults_for(section, impl, doc)
                self.assertEqual(got["max_wait"], 42.0)
                self.assertNotIn("poll_interval", got)

    def test_job_result_max_wait_is_not_config_reachable(self):
        """`job_result`'s max_wait=0.0 means "one non-blocking probe" -- a
        meaning, not a stand-in default. A sweep that tri-stated every max_wait
        would turn it into a blocking call."""
        import inspect

        from venice.commands import _mcp
        self.assertNotIn("job_result", uc._COMMAND_MAP)
        self.assertEqual(
            inspect.signature(_mcp.job_result_tool).parameters["max_wait"].default,
            0.0)

    def test_negative_cadence_from_config_is_rejected_at_the_coercer(self):
        """The guard has to live in the COERCER, not the handler: these rows also
        feed the agent/MCP tool impls through `config_defaults_for`, where a
        negative max_wait abandons a job that `_queue_media` has already queued
        and CHARGED. A handler-side clamp protects only the CLI."""
        from venice.commands import _mcp
        doc = {"defaults": {"sfx": {"poll_interval": -1, "max_wait": -5}}}

        # CLI surface: warn-and-skip leaves the dests None, so the literal wins.
        args = _build_parser(sfx).parse_args(["sfx", "p"])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            uc.apply_defaults(args, "sfx", doc)
        self.assertIsNone(args.poll_interval)
        self.assertIsNone(args.max_wait)
        self.assertIn("must be >= 0", err.getvalue())

        # Tool surface: the key is dropped entirely, so the impl default wins.
        with contextlib.redirect_stderr(io.StringIO()):
            got = uc.config_defaults_for("sfx", _mcp.sfx_tool, doc)
        self.assertNotIn("max_wait", got)

    def test_negative_cadence_typed_on_the_cli_falls_back(self):
        """A value the user TYPED never passes through a coercer, so the handler
        still clamps it -- `time.sleep(-1)` is a ValueError traceback."""
        from venice.commands import _shared
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            interval, max_wait = _shared.resolve_poll(
                -1.0, -5.0, label="sfx", interval=2.0, max_wait_default=300)
        self.assertEqual((interval, max_wait), (2.0, 300))
        self.assertIn("--poll-interval must be >= 0", err.getvalue())
        self.assertIn("--max-wait must be >= 0", err.getvalue())

    def test_zero_cadence_survives(self):
        """0 is meaningful on both -- a tight loop and a single probe -- so it
        must NOT be rewritten to the literal. This is the `is not None`, never
        `or`, contract."""
        from venice.commands import _shared
        args = _build_parser(sfx).parse_args(["sfx", "p"])
        uc.apply_defaults(args, "sfx", {"defaults": {
            "sfx": {"poll_interval": 0, "max_wait": 0}}})
        self.assertEqual(args.poll_interval, 0.0)
        self.assertEqual(args.max_wait, 0.0)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            got = _shared.resolve_poll(args.poll_interval, args.max_wait,
                                       label="sfx", interval=2.0, max_wait_default=300)
        self.assertEqual(got, (0.0, 0.0))
        self.assertEqual(err.getvalue(), "")


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

    def test_add_http_header_secret_ref_stored_verbatim(self):
        # #70: a @secret:<name> token rides the existing --header flag and is
        # stored literally (the second ':' stays with the value; nothing is
        # resolved at write time -- resolution happens at attach).
        rc, _, _ = _capture(cfgcmd._run_add, _add_args(
            "remote", url="https://x/mcp",
            header=["Authorization: Bearer @secret:cluster"]))
        self.assertEqual(rc, 0)
        entry = uc.mcp_get(uc.load_config(), "remote")
        self.assertEqual(entry["headers"],
                         {"Authorization": "Bearer @secret:cluster"})

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


class TestShellPolicy(_Base):
    """The top-level `shell` allow/deny reader (#33), mirroring mcp_map."""

    def test_missing_section_is_empty(self):
        self.assertEqual(uc.shell_policy({}), {"allow": [], "deny": []})

    def test_malformed_section_is_empty(self):
        self.assertEqual(uc.shell_policy({"shell": "nope"}), {"allow": [], "deny": []})

    def test_reads_lists(self):
        doc = {"shell": {"allow": ["git", "ls"], "deny": ["rm *", "sudo *"]}}
        self.assertEqual(
            uc.shell_policy(doc),
            {"allow": ["git", "ls"], "deny": ["rm *", "sudo *"]},
        )

    def test_scalar_string_coerced_to_list(self):
        # Mirrors _as_list: a bare string becomes a single-element list.
        self.assertEqual(
            uc.shell_policy({"shell": {"allow": "git", "deny": "rm"}}),
            {"allow": ["git"], "deny": ["rm"]},
        )

    def test_dotted_key_set_roundtrips_through_generic_store(self):
        # `venice config set shell.deny '["rm *"]'` works with no bespoke plumbing.
        doc = uc.load_config()
        uc.set_value(doc, "shell.deny", ["rm *"])
        self.assertEqual(uc.shell_policy(doc), {"allow": [], "deny": ["rm *"]})


class TestBrowserPolicy(_Base):
    """The top-level `browser` URL allow/deny reader (#71), mirroring shell_policy."""

    def test_missing_and_malformed_are_empty(self):
        self.assertEqual(uc.browser_policy({}), {"allow": [], "deny": []})
        self.assertEqual(uc.browser_policy({"browser": "nope"}), {"allow": [], "deny": []})

    def test_reads_lists_and_coerces_scalar(self):
        doc = {"browser": {"allow": ["example.com"], "deny": "*.internal"}}
        self.assertEqual(
            uc.browser_policy(doc),
            {"allow": ["example.com"], "deny": ["*.internal"]},
        )

    def test_dotted_key_set_roundtrips(self):
        doc = uc.load_config()
        uc.set_value(doc, "browser.deny", ["*.internal"])
        self.assertEqual(uc.browser_policy(doc), {"allow": [], "deny": ["*.internal"]})


class TestRootsPolicy(_Base):
    """The top-level `roots` writable/read-only reader (#76), mirroring shell_policy."""

    def test_missing_and_malformed_are_empty(self):
        self.assertEqual(uc.roots_policy({}), {"allow": [], "deny": []})
        self.assertEqual(uc.roots_policy({"roots": "nope"}), {"allow": [], "deny": []})

    def test_reads_lists_and_coerces_scalar(self):
        doc = {"roots": {"allow": ["/a", "/b"], "deny": "/a/vendor"}}
        self.assertEqual(
            uc.roots_policy(doc),
            {"allow": ["/a", "/b"], "deny": ["/a/vendor"]},
        )

    def test_dotted_key_set_roundtrips(self):
        doc = uc.load_config()
        uc.set_value(doc, "roots.deny", ["*/vendor"])
        self.assertEqual(uc.roots_policy(doc), {"allow": [], "deny": ["*/vendor"]})


class TestConfigDefaultsFor(unittest.TestCase):
    """#58: the shared tool-path resolver -- allow-listed, coerced, signature-gated.

    Uses `commands._mcp` (the pure impl module, import-safe without the [mcp] extra)
    as the introspection target, exactly as mcp-serve/chat/code do at runtime."""

    def test_introspects_coerces_and_allowlists(self):
        from venice.commands import _mcp
        doc = {"defaults": {"image": {
            "hide_watermark": "true", "safe_mode": False, "steps": "12", "preset": "x",
        }}}
        out = uc.config_defaults_for("image", _mcp.image_tool, doc)
        self.assertIs(out["hide_watermark"], True)   # _as_bool("true")
        self.assertIs(out["safe_mode"], False)
        self.assertEqual(out["steps"], 12)           # int("12")
        self.assertNotIn("preset", out)              # not an image_tool param

    def test_none_doc_and_unknown_section_are_empty(self):
        from venice.commands import _mcp
        self.assertEqual(
            uc.config_defaults_for("image", _mcp.image_tool, None), {}
        )
        self.assertEqual(
            uc.config_defaults_for(
                "bg_remove", _mcp.bg_remove_tool, {"defaults": {"bg_remove": {"x": 1}}}
            ),
            {},
        )

    def test_bad_value_is_skipped_not_raised(self):
        from venice.commands import _mcp
        doc = {"defaults": {"image": {"steps": "not-an-int", "safe_mode": False}}}
        out = uc.config_defaults_for("image", _mcp.image_tool, doc)
        self.assertNotIn("steps", out)               # int("not-an-int") -> skipped
        self.assertIs(out["safe_mode"], False)       # the good key still lands

    def test_browser_section_gates_by_signature(self):
        # #71: web_fetch/browser_capture share the `browser` section; each impl gets only
        # the keys its signature accepts (capture: wait_ms/timeout; fetch: max_bytes/timeout).
        from venice.commands import _mcp
        doc = {"defaults": {"browser": {"wait_ms": "2000", "timeout": 10, "max_bytes": 5}}}
        cap = uc.config_defaults_for("browser", _mcp.browser_capture_tool, doc)
        self.assertEqual(cap, {"wait_ms": 2000, "timeout": 10})
        self.assertNotIn("max_bytes", cap)           # browser_capture takes no max_bytes
        fetch = uc.config_defaults_for("browser", _mcp.web_fetch_tool, doc)
        self.assertEqual(fetch, {"timeout": 10, "max_bytes": 5})
        self.assertNotIn("wait_ms", fetch)           # web_fetch takes no wait_ms


if __name__ == "__main__":
    unittest.main()
