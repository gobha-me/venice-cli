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

import json
import sys
from pathlib import Path
from typing import List, Optional

from . import _agent, _models


_PROMPT = "you> "

_HELP = """\
Commands:
  /system [text]   show, or set, the system prompt (reseeds the conversation)
  /model [name]    show, or switch, the model for following turns
  /reset           clear the conversation (keeps the system prompt)
  /save [file]     write the transcript JSON (default: the --resume file)
  /help            show this help
  /exit, /quit     leave the REPL
Anything else is sent to the model as your next message."""


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
# One turn
# --------------------------------------------------------------------------- #
def _stream_turn(oai, chat, model: str, messages: List[dict], gen_kwargs: dict) -> None:
    kwargs = dict(gen_kwargs)
    kwargs["model"] = model
    kwargs["messages"] = messages
    kwargs["stream"] = True
    kwargs["stream_options"] = {"include_usage": True}
    text = chat._consume_stream(oai.chat.completions.create(**kwargs))
    messages.append({"role": "assistant", "content": text})


def _do_turn(oai, openai, chat, text, messages, gen_kwargs, state, args) -> None:
    """Run one turn. Any failure/interrupt rolls the turn's messages back so the
    persistent history stays a valid, replayable conversation, and the session
    survives (only `/exit`/EOF ends the REPL)."""
    mark = len(messages)
    messages.append({"role": "user", "content": text})
    try:
        if state["tools_on"]:
            _agent.run_loop(
                oai, state["model"], messages, gen_kwargs, state["tools"],
                max_tool_calls=(args.max_tool_calls or 8),
                yes=bool(args.yes),
                json_out=False,
            )
        else:
            _stream_turn(oai, chat, state["model"], messages, gen_kwargs)
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
def _dispatch_slash(line, messages, state, args, models) -> str:
    """Handle a ``/command``. Returns ``"exit"`` to leave the REPL, else
    ``"continue"``."""
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
def run(args, oai, openai, client, models, model, initial=None) -> int:
    """Drive the interactive REPL until `/exit`, `/quit`, or EOF (Ctrl-D).

    `initial` is an already-resolved first message (e.g. `venice chat -i "hi"`);
    when set it runs as the opening turn before the prompt loop starts.
    """
    from . import chat  # lazy: chat imports this module at top (avoid a cycle)

    try:
        import readline  # noqa: F401  (line editing + in-session history)
    except Exception:  # pragma: no cover - platform without readline
        pass

    try:
        messages = _seed_messages(args)
    except _TranscriptError as e:
        print(f"chat: {e}", file=sys.stderr)
        return 2

    gen_kwargs = chat._gen_kwargs(args)

    tools_on = bool(getattr(args, "tools", None))
    tools = None
    if tools_on:
        tools, rc = chat._tools_for(args, client, models, model)
        if tools is None:
            if rc is not None:
                return rc      # invalid --tool subset
            tools_on = False   # model lacks function calling -> plain chat
    state = {"model": model, "tools": tools, "tools_on": tools_on}

    _banner(model, tools_on, getattr(args, "resume", None), messages)

    # An explicit message (e.g. `venice chat -i "hello"`) becomes the first turn.
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
            if _dispatch_slash(line, messages, state, args, models) == "exit":
                return 0
            continue
        _do_turn(oai, openai, chat, line, messages, gen_kwargs, state, args)


def _banner(model, tools_on, resume, messages) -> None:
    bits = [f"model {model}"]
    if tools_on:
        bits.append("tools on")
    if resume:
        bits.append(f"resumed {len(messages)} msg(s) from {resume}")
    print(
        f"venice chat -- interactive ({', '.join(bits)}). "
        "/help for commands; /exit or Ctrl-D to quit.",
        file=sys.stderr,
    )
