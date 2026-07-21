"""Interactive multi-turn REPL for `venice chat` (issue #22).

`venice chat` is one-shot by default. With `-i`/`--interactive` -- or simply no
message on an interactive terminal -- it drops into a REPL that holds the
conversation in memory across turns: read a line, append it to the history, run
one completion turn, print the reply, repeat. Streaming is on by default; when
tools are enabled (`--tools`, #15) each turn is an agent turn via
:func:`_agent.run_loop` (non-streamed, matching one-shot `--tools`).

This is purely a loop + state layer over `chat`'s existing request plumbing -- it
adds no new API surface. It reuses ``chat._gen_kwargs`` (per-turn generation
kwargs), ``chat._consume_stream`` (streamed reply accumulation), and
``chat._tools_for`` (built-in tool resolution + capability guard), plus
:func:`_agent.run_loop` and :func:`_models.resolve_model`.

Slash-commands (minimal set): ``/system`` ``/model`` ``/reset`` ``/save``
``/exit`` (plus ``/help`` and the ``/quit`` alias). Transcripts round-trip as a
JSON list of messages via ``--resume FILE`` and ``/save``.

Secret hygiene (CLAUDE.md): the REPL prints only model output and message
content; it never echoes the API key. A saved transcript holds only the
``messages`` array (roles + content), never auth material.
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import _agent, _compact, _models


_PROMPT = "you> "

_HELP = """\
Commands:
  /system [text]   show, or set, the system prompt (reseeds the conversation)
  /model [name]    switch model; with no name, show the current one and list the catalog
  /models          list the available models (marks the current and default)
  /auto            auto-accept paid/side-effecting tool calls for following turns
  /manual          confirm each paid/side-effecting tool call (undo /auto)
  /compact [N]     summarize older history to shrink the context (keeps last N turns)
  /reset           clear the conversation (keeps the system prompt)
  /save [file]     write the transcript JSON (default: the --resume file)
  /help            show this help
  /exit, /quit     leave the REPL
Anything else is sent to the model as your next message."""

# Slash-commands, in help order -- the single source of truth for tab-completion
# (#40). Keep in sync with `_dispatch_slash` and `_HELP`.
_COMMANDS = (
    "/system", "/model", "/models", "/auto", "/manual", "/compact", "/reset",
    "/save", "/help", "/exit", "/quit",
)


class _TranscriptError(Exception):
    """A --resume transcript file is missing or malformed. Message is printable."""


# --------------------------------------------------------------------------- #
# Transcript I/O (--resume / /save)
# --------------------------------------------------------------------------- #
def _load_transcript(path: str) -> List[dict]:
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        raise _TranscriptError(f"cannot read transcript {path}: {e}")
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise _TranscriptError(f"invalid transcript JSON in {path}: {e}")
    if not isinstance(data, list) or not all(
        isinstance(m, dict) and "role" in m for m in data
    ):
        raise _TranscriptError(
            f"transcript {path} must be a JSON list of message objects"
        )
    return data


def _save_transcript(path: str, messages: List[dict]) -> None:
    Path(path).write_text(
        json.dumps(messages, indent=2, default=str) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# History state
# --------------------------------------------------------------------------- #
def _seed_messages(args) -> List[dict]:
    """Initial history: a resumed transcript, else an optional system prompt."""
    resume = getattr(args, "resume", None)
    if resume:
        msgs = _load_transcript(resume)
        if args.system and not any(m.get("role") == "system" for m in msgs):
            msgs.insert(0, {"role": "system", "content": args.system})
        return msgs
    if args.system:
        return [{"role": "system", "content": args.system}]
    return []


def _reset_messages(messages: List[dict]) -> None:
    """Clear history in place, keeping only a leading system message if present."""
    keep = messages[0] if messages and messages[0].get("role") == "system" else None
    messages.clear()
    if keep is not None:
        messages.append(keep)


def _set_system(messages: List[dict], text: str) -> None:
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = text
    else:
        messages.insert(0, {"role": "system", "content": text})


def _current_system(messages: List[dict]) -> Optional[str]:
    if messages and messages[0].get("role") == "system":
        return messages[0].get("content")
    return None


# --------------------------------------------------------------------------- #
# Model listing (#39) + tab-completion (#40)
# --------------------------------------------------------------------------- #
def _format_model_list(models, current: Optional[str]) -> str:
    """One id per line for `/models` and bare `/model`.

    Marks the current model with ``*`` and the catalog's default-trait model with
    a trailing ``(default)``; appends ``model_spec.name`` when present. Reuses the
    already-fetched `models` list (no catalog re-fetch). Returns a printable block.
    """
    if not models:
        return "(model catalog unavailable; pass --model or /model <id> to switch)"
    default = _models.default_model(models)
    lines = []
    for m in models:
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        if not mid:
            continue
        spec = m.get("model_spec") if isinstance(m.get("model_spec"), dict) else {}
        name = spec.get("name")
        marker = "*" if mid == current else " "
        tags = " (default)" if mid == default else ""
        line = f"  {marker} {mid}{tags}"
        if name:
            line += f"  -- {name}"
        lines.append(line)
    return "\n".join(lines) if lines else "(no models advertised)"


def _make_completer(models, rl):
    """Build a `readline` completer over the slash-commands and model ids (#40).

    `rl` is the readline module (injected so the closure is unit-testable without
    a real terminal). The returned ``completer(text, state)`` completes the leading
    ``/command`` token, and model ids after ``/model ``; it is a no-op on non-slash
    lines so ordinary prose is never auto-completed.
    """
    model_ids = [
        m.get("id") for m in (models or [])
        if isinstance(m, dict) and m.get("id")
    ]

    def completer(text, state):
        buf = rl.get_line_buffer()
        if not buf.lstrip().startswith("/"):
            return None
        # Empty prefix left of the token => we're completing the command word.
        if not buf[: rl.get_begidx()].strip():
            candidates = _COMMANDS
        elif buf.lstrip().split(maxsplit=1)[0].lower() == "/model":
            candidates = model_ids
        else:
            candidates = ()
        matches = [c for c in candidates if c.startswith(text)]
        return matches[state] if state < len(matches) else None

    return completer


def _install_completer(rl, models, stack) -> None:
    """Register the tab-completer for the session, restoring the previous one on exit.

    Sets whitespace-only delimiters so a leading ``/command`` is one token, installs
    the `_make_completer` closure, and binds Tab (libedit vs GNU readline). The
    `stack` callback restores the prior completer + delims so we never leak into a
    parent readline context (e.g. a REPL launched from another readline program).
    """
    prev_completer = rl.get_completer()
    prev_delims = rl.get_completer_delims()

    def _restore():
        rl.set_completer(prev_completer)
        rl.set_completer_delims(prev_delims)

    stack.callback(_restore)
    rl.set_completer_delims(" \t\n")
    rl.set_completer(_make_completer(models, rl))
    if "libedit" in (getattr(rl, "__doc__", "") or ""):
        rl.parse_and_bind("bind ^I rl_complete")
    else:
        rl.parse_and_bind("tab: complete")


# --------------------------------------------------------------------------- #
# One turn
# --------------------------------------------------------------------------- #
def _stream_turn(oai, chat, model: str, messages: List[dict], gen_kwargs: dict):
    """One streamed turn; returns (reply_text, usage) for history + budget (#48)."""
    kwargs = dict(gen_kwargs)
    kwargs["model"] = model
    kwargs["messages"] = messages
    kwargs["stream"] = True
    kwargs["stream_options"] = {"include_usage": True}
    return chat._consume_stream_full(oai.chat.completions.create(**kwargs))


def _do_turn(oai, openai, chat, text, messages, gen_kwargs, state, args) -> None:
    """Run one turn. Any failure/interrupt rolls the turn's messages back so the
    persistent history stays a valid, replayable conversation, and the session
    survives (only `/exit`/EOF ends the REPL).

    Auto-compaction (#48): when the session carries a `budget`, an over-budget
    history is summarized BEFORE the completion call, outside the rollback
    window -- a turn failure then rolls back only that turn, not the compaction.
    Tool turns pass the budget to `run_loop`, which compacts between calls and
    observes `usage` itself; streamed turns compact here and observe usage from
    the stream's final chunk.
    """
    budget = state.get("budget")
    _compact.maybe_compact(
        oai, state["model"], messages, budget, gen_kwargs,
        on_compact=lambda b, a: print(
            f"(auto-compacted history: {b} -> {a} messages)", file=sys.stderr,
        ),
    )
    mark = len(messages)
    messages.append({"role": "user", "content": text})
    try:
        if state["tools_on"]:
            _agent.run_loop(
                oai, state["model"], messages, gen_kwargs, state["tools"],
                max_tool_calls=state["max_tool_calls"],
                yes=state["yes"],
                json_out=False,
                budget=budget,
            )
        else:
            reply, usage = _stream_turn(oai, chat, state["model"], messages, gen_kwargs)
            if budget is not None:
                budget.observe(usage)
            messages.append({"role": "assistant", "content": reply})
    except KeyboardInterrupt:
        # Ctrl-C aborts just this turn -- roll it back and keep the session.
        del messages[mark:]
        print("\n[turn aborted]", file=sys.stderr)
    except openai.OpenAIError as e:
        del messages[mark:]
        chat._openai.status_to_exit(openai, e, "chat")  # prints; session survives


# --------------------------------------------------------------------------- #
# Slash-commands
# --------------------------------------------------------------------------- #
def _dispatch_slash(line, messages, state, args, models, oai=None, gen_kwargs=None) -> str:
    """Handle a ``/command``. Returns ``"exit"`` to leave the REPL, else
    ``"continue"``. `oai`/`gen_kwargs` are needed only by `/compact` (#48)."""
    parts = line[1:].split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    rest = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("exit", "quit"):
        return "exit"
    if cmd == "help":
        print(_HELP, file=sys.stderr)
    elif cmd == "reset":
        _reset_messages(messages)
        print("(conversation cleared)", file=sys.stderr)
    elif cmd == "system":
        if rest:
            _set_system(messages, rest)
            print("(system prompt set)", file=sys.stderr)
        else:
            print(f"system: {_current_system(messages) or '(none)'}", file=sys.stderr)
    elif cmd == "model":
        if rest:
            new, rc = _models.resolve_model(
                rest, models, label="chat", noun="text model"
            )
            if rc is not None:
                pass  # resolve_model printed why; keep the current model
            else:
                state["model"] = new
                if state["tools_on"] and (
                    _agent.supports_function_calling(models, new) is False
                ):
                    state["tools_on"] = False
                    print(
                        f"(model {new} has no function calling; tools disabled)",
                        file=sys.stderr,
                    )
                print(f"(model -> {new})", file=sys.stderr)
        else:
            print(f"model: {state['model']}", file=sys.stderr)
            print(_format_model_list(models, state["model"]), file=sys.stderr)
    elif cmd == "models":
        print(_format_model_list(models, state["model"]), file=sys.stderr)
    elif cmd in ("auto", "manual"):
        if not state["tools_on"]:
            print("(no tools this session; nothing to auto-accept)", file=sys.stderr)
        else:
            state["yes"] = cmd == "auto"
            on = "on" if state["yes"] else "off"
            print(f"(auto-accept {on})", file=sys.stderr)
    elif cmd == "compact":
        # Manual compaction (#48): summarize the older prefix with the session
        # model; `rest` can override how many recent turns stay verbatim.
        keep = _compact.DEFAULT_KEEP_TURNS
        if state.get("budget") is not None:
            keep = state["budget"].keep_turns
        if rest:
            try:
                keep = max(1, int(rest))
            except ValueError:
                print(f"/compact: bad turn count {rest!r}", file=sys.stderr)
                return "continue"
        before = len(messages)
        if _compact.compact_messages(
            oai, state["model"], messages,
            keep_turns=keep, base_kwargs=gen_kwargs,
        ):
            if state.get("budget") is not None:
                state["budget"].last_prompt_tokens = None
            print(
                f"(compacted: {before} -> {len(messages)} messages; "
                f"last {keep} turn(s) verbatim)",
                file=sys.stderr,
            )
        else:
            print("(nothing to compact)", file=sys.stderr)
    elif cmd == "save":
        target = rest or getattr(args, "resume", None)
        if not target:
            print(
                "/save: give a file path (or start with --resume FILE)",
                file=sys.stderr,
            )
        else:
            try:
                _save_transcript(target, messages)
                print(f"(saved {len(messages)} msg(s) -> {target})", file=sys.stderr)
            except OSError as e:
                print(f"/save: {e}", file=sys.stderr)
    else:
        print(f"unknown command /{cmd}; /help for the list", file=sys.stderr)
    return "continue"


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def run(args, oai, openai, client, models, model, initial=None, *,
        tools_session=None, gen_kwargs=None, label="venice chat",
        max_tool_calls=8) -> int:
    """Drive the interactive REPL until `/exit`, `/quit`, or EOF (Ctrl-D).

    `initial` is an already-resolved first message (e.g. `venice chat -i "hi"`);
    when set it runs as the opening turn before the prompt loop starts.

    `tools_session` / `gen_kwargs` / `label` let another command reuse this REPL
    with its own tool set and generation kwargs: `venice code` (#30) injects a
    coding tools session + minimal gen kwargs so the REPL never touches chat's
    Venice-extension flags. Defaults preserve `venice chat`'s behavior exactly.

    `max_tool_calls` is the per-turn budget when no `--max-tool-calls` is given
    (chat=8, `venice code` passes its higher default); `--max-tool-calls 0`/`<=0`
    means unlimited. `/auto` and `/manual` flip per-turn auto-accept live (#55).
    """
    from . import chat  # lazy: chat imports this module at top (avoid a cycle)

    try:
        import readline as _rl  # line editing + in-session history + completion
    except Exception:  # pragma: no cover - platform without readline
        _rl = None

    try:
        messages = _seed_messages(args)
    except _TranscriptError as e:
        print(f"chat: {e}", file=sys.stderr)
        return 2

    if gen_kwargs is None:
        gen_kwargs = chat._gen_kwargs(args)

    # `--tools`/`--mcp` (or an injected `tools_session`) turns the REPL into an
    # agent session. Any MCP servers stay attached for the whole session via the
    # ExitStack, torn down on exit; the capability guard runs inside the session.
    if tools_session is not None:
        tools_on = True
        session_cm = tools_session
    else:
        tools_on = bool(getattr(args, "tools", None)) or bool(
            chat._requested_mcp_servers(args)
        )
        session_cm = (
            chat._tools_session(args, client, models, model) if tools_on else None
        )
    with contextlib.ExitStack() as stack:
        tools = None
        if tools_on:
            tools, rc = stack.enter_context(session_cm)
            if tools is None:
                if rc is not None:
                    return rc      # invalid --tool subset / MCP attach error
                tools_on = False   # model lacks function calling -> plain chat
        cap = getattr(args, "max_tool_calls", None)
        state = {
            "model": model,
            "tools": tools,
            "tools_on": tools_on,
            "yes": bool(getattr(args, "yes", False)),  # /auto and /manual flip this
            "max_tool_calls": cap if cap is not None else max_tool_calls,
        }
        # Auto-compaction (#48) is opt-in: `--auto-compact` or
        # `defaults.<cmd>.auto_compact` (it costs a summarization call).
        state["budget"] = _compact.budget_from_args(args)

        if _rl is not None:
            _install_completer(_rl, models, stack)

        _banner(model, tools_on, getattr(args, "resume", None), messages,
                label=label, auto=state["yes"])

        # An explicit message (`venice chat -i "hello"`) becomes the first turn.
        if initial:
            _do_turn(oai, openai, chat, initial, messages, gen_kwargs, state, args)

        while True:
            try:
                line = input(_PROMPT)
            except EOFError:
                print(file=sys.stderr)  # newline after ^D
                return 0
            except KeyboardInterrupt:
                print(file=sys.stderr)  # ^C at the prompt: discard the line, re-prompt
                continue
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if _dispatch_slash(
                    line, messages, state, args, models, oai=oai, gen_kwargs=gen_kwargs
                ) == "exit":
                    return 0
                continue
            _do_turn(oai, openai, chat, line, messages, gen_kwargs, state, args)


def _banner(model, tools_on, resume, messages, *, label="venice chat",
            auto=False) -> None:
    bits = [f"model {model}"]
    if tools_on:
        bits.append("tools on")
        bits.append("auto-accept on" if auto else "auto-accept off (/auto to enable)")
    if resume:
        bits.append(f"resumed {len(messages)} msg(s) from {resume}")
    print(
        f"{label} -- interactive ({', '.join(bits)}). "
        "/help for commands; /exit or Ctrl-D to quit.",
        file=sys.stderr,
    )
