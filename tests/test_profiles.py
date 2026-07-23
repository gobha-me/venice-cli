"""Round-trip tests for the AgentProfile abstraction (#51).

`venice chat` and `venice code` are two profiles over one shared agent core. These
tests pin every seeded value each profile carries, and that the profile's builder
callables are the commands' own helpers (so there is no prompt/kwarg drift between the
profile and the command path). Reuses the arg-builders from the chat/code suites.
"""

import unittest

from venice.commands import _agent, _code, _session, chat, code
from venice.commands.code import CODING_SYSTEM_PROMPT

from tests.test_chat import _args as _chat_args
from tests.test_code_command import _code_args


class TestProfileScalars(unittest.TestCase):
    def test_chat_profile_seeds(self):
        p = chat.PROFILE
        self.assertIsInstance(p, _agent.AgentProfile)
        self.assertEqual(p.name, "chat")
        self.assertEqual(p.label, "venice chat")
        self.assertEqual(p.default_max_tool_calls, 8)
        self.assertFalse(p.plan_mode)
        self.assertTrue(p.degrade_to_chat)  # non-FC model -> plain chat
        self.assertFalse(p.system_reseed)
        self.assertFalse(p.injects_tools_session)  # REPL derives tools from args

    def test_code_profile_seeds(self):
        p = code.PROFILE
        self.assertIsInstance(p, _agent.AgentProfile)
        self.assertEqual(p.name, "code")
        self.assertEqual(p.label, "venice code")
        # Seeded from the live module constant (a test reads it directly), value 25.
        self.assertEqual(p.default_max_tool_calls, code._DEFAULT_MAX_TOOL_CALLS)
        self.assertEqual(p.default_max_tool_calls, 25)
        self.assertTrue(p.plan_mode)
        self.assertFalse(p.degrade_to_chat)  # non-FC model -> exit 2
        self.assertTrue(p.system_reseed)  # rebind root-aware prompt on resume
        self.assertTrue(p.injects_tools_session)

    def test_label_maps_to_session_command(self):
        # command_from_label drives the persisted session `command` field (#47).
        self.assertEqual(_session.command_from_label(chat.PROFILE.label), "chat")
        self.assertEqual(_session.command_from_label(code.PROFILE.label), "code")


class TestProfileBuilders(unittest.TestCase):
    def test_builders_are_the_command_helpers(self):
        # Same function objects -> the profile can never drift from the command path.
        self.assertIs(chat.PROFILE.build_gen_kwargs, chat._gen_kwargs)
        self.assertIs(chat.PROFILE.build_system, chat._system_for)
        self.assertIs(code.PROFILE.build_gen_kwargs, code._gen_kwargs)
        self.assertIs(code.PROFILE.build_system, code._system_prompt)

    def test_chat_gen_kwargs_carry_venice_parameters(self):
        kw = chat.PROFILE.build_gen_kwargs(_chat_args(web_search=True, temperature=0.5))
        self.assertEqual(kw["temperature"], 0.5)
        self.assertEqual(
            kw["extra_body"], {"venice_parameters": {"enable_web_search": True}}
        )

    def test_chat_gen_kwargs_omit_extra_body_when_no_extensions(self):
        kw = chat.PROFILE.build_gen_kwargs(_chat_args(temperature=0.2))
        self.assertNotIn("extra_body", kw)

    def test_code_gen_kwargs_never_carry_venice_parameters(self):
        kw = code.PROFILE.build_gen_kwargs(_code_args(temperature=0.5, max_tokens=100))
        self.assertEqual(kw, {"temperature": 0.5, "max_tokens": 100})
        self.assertNotIn("extra_body", kw)

    def test_chat_system_is_verbatim_and_ignores_root_tools(self):
        args = _chat_args(system="be terse")
        self.assertEqual(chat.PROFILE.build_system(args), "be terse")
        self.assertEqual(chat.PROFILE.build_system(args, "/repo", []), "be terse")
        self.assertIsNone(chat.PROFILE.build_system(_chat_args(system=None)))

    def test_code_system_is_templated_and_appends_user_system(self):
        base = CODING_SYSTEM_PROMPT.format(root="/repo", tools=_code.tool_names([]))
        out = code.PROFILE.build_system(_code_args(system="use tabs"), "/repo", [])
        self.assertIn("/repo", out)
        self.assertTrue(out.startswith(base))
        self.assertIn("Project-specific instructions:\nuse tabs", out)

    def test_code_system_without_user_system(self):
        base = CODING_SYSTEM_PROMPT.format(root="/repo", tools=_code.tool_names([]))
        out = code.PROFILE.build_system(_code_args(system=None), "/repo", [])
        self.assertEqual(out, base)
        self.assertNotIn("Project-specific instructions", out)


if __name__ == "__main__":
    unittest.main()
