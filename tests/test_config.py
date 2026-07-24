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
    chat, code, image, image_edit, index, music, sfx, tts, upscale, video,
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
