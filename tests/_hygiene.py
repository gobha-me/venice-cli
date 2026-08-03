"""Reject source characters that are invisible to a diff (#109).

Every other guard in this suite asks whether the code is *correct*. This one
asks whether the code a reviewer read is the code the interpreter runs.

The GlassWorm class of supply-chain payload hides executable source inside
codepoints that render as nothing in an editor, in `git diff`, and in the
GitHub review UI: zero-width joiners, the bidi overrides behind "Trojan
Source", Unicode tag characters (a full ASCII alphabet with no glyphs), and
variation selectors. A homoglyph identifier -- Cyrillic `a` in `api_key` --
is the same attack wearing a visible costume: the reviewer sees the right
word and the parser binds a different name.

Reading the diff cannot catch any of this, and reading the diff is how every
PR here is reviewed. That was tolerable while every PR came from `gobha-me`;
#108 was the first one that didn't.

**Visible punctuation is deliberately legal.** The repo already contains 181
em dashes, plus arrows, ellipses and `<=`/`>=` glyphs in prose and CLI output.
A guard that flagged those would be turned off the first week, so the rule is
drawn at *invisibility*, not at non-ASCII: if it has a glyph, it passes.

The identifier pass tokenizes rather than pattern-matches for the same reason
-- it is what keeps a horizontal ellipsis legal inside a docstring while
rejecting one spliced into a variable name.

Scanning `git ls-files` rather than walking the tree is deliberate: the
authoritative set is what is actually committed and shipped, not whatever
happens to be sitting in a working copy. Without git (an unpacked sdist) there
is no such set -- `scan_repo` returns `None` and callers decide, exactly as
`make test` skips the pty suite without pexpect while `make drive` hard-fails.
"""
import io
import subprocess
import sys
import tokenize
import unicodedata
from collections import namedtuple
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

Finding = namedtuple("Finding", "path line col label detail")

# Variation selectors report as Mn (a combining mark), not Cf, so the category
# sweep below genuinely cannot see them -- hence the explicit ranges. Private
# use needs no such range: `unicodedata.category` returns Co across the BMP
# block and both supplementary planes, which the Co branch already covers.
_VARIATION_SELECTORS = ((0xFE00, 0xFE0F), (0xE0100, 0xE01EF))

# The only whitespace that may appear literally. Tabs are load-bearing in the
# Makefile; CR is tolerated so a CRLF checkout is not a security finding.
_ALLOWED_CONTROLS = "\n\r\t"


def classify(ch: str):
    """The suspect class of `ch`, or None if it is legitimate.

    Anything with a visible glyph returns None -- see the module docstring on
    why the line is drawn at invisibility rather than at non-ASCII.
    """
    code = ord(ch)
    category = unicodedata.category(ch)
    if any(lo <= code <= hi for lo, hi in _VARIATION_SELECTORS):
        return "VARIATION_SELECTOR"
    if category == "Cf":
        # Zero-width joiners, U+FEFF, the bidi overrides, and the U+E0000 tag
        # alphabet all land here. This is the main event.
        return "INVISIBLE_FORMAT"
    if category == "Co":
        return "PRIVATE_USE"
    if category == "Cc" and ch not in _ALLOWED_CONTROLS:
        return "CONTROL"
    if category == "Zs" and ch != " ":
        # Non-breaking and en/em spaces: indistinguishable from a space on
        # screen, distinct to the parser.
        return "UNUSUAL_SPACE"
    if category in ("Zl", "Zp"):
        return "LINE_SEPARATOR"
    return None


def scan_text(path, text):
    """Findings for one file's decoded text."""
    findings = []
    for line_no, line in enumerate(text.splitlines(), 1):
        for col, ch in enumerate(line, 1):
            if ch.isascii():
                continue
            label = classify(ch)
            if label:
                try:
                    name = unicodedata.name(ch)
                except ValueError:
                    name = "<unnamed>"
                findings.append(
                    Finding(path, line_no, col, label, f"U+{ord(ch):04X} {name}")
                )
    return findings


def scan_identifiers(path, data):
    """Findings for non-ASCII NAME tokens -- the homoglyph half.

    A file that will not tokenize is *not* reported here: `make lint` compiles
    every source in the repo and CI does it on five interpreters, so a genuine
    syntax error is already caught, loudly, somewhere that explains itself. The
    character sweep above still runs on the file either way, so nothing is
    hidden by declining to guess here.
    """
    findings = []
    try:
        for token in tokenize.tokenize(io.BytesIO(data).readline):
            if token.type == tokenize.NAME and not token.string.isascii():
                findings.append(
                    Finding(
                        path,
                        token.start[0],
                        token.start[1] + 1,
                        "NON_ASCII_IDENTIFIER",
                        repr(token.string),
                    )
                )
    except (tokenize.TokenError, SyntaxError, IndentationError, UnicodeDecodeError):
        pass
    return findings


def scan_file(path, data):
    """Findings for one file, given its raw bytes."""
    if data.startswith(b"\xef\xbb\xbf"):
        return [Finding(path, 1, 1, "UTF8_BOM", "byte-order mark")]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [Finding(path, 0, 0, "NOT_UTF8", str(exc))]
    findings = scan_text(path, text)
    if path.endswith(".py"):
        findings.extend(scan_identifiers(path, data))
    # The two passes overlap on one input: `tokenize` reports a bare zero-width
    # character as a NAME token, so the sweep and the identifier pass both flag
    # it. They are otherwise complementary -- a Cyrillic `a` is a perfectly
    # visible letter the sweep ignores by design. Collapse by position, keeping
    # the sweep's label, so one character is never two findings.
    seen = set()
    unique = []
    for finding in findings:
        if (finding.line, finding.col) in seen:
            continue
        seen.add((finding.line, finding.col))
        unique.append(finding)
    return unique


def tracked_files(root=REPO_ROOT):
    """Every file git is tracking, or None when this is not a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [name for name in out.decode("utf-8").split("\0") if name]


def scan_repo(root=REPO_ROOT):
    """All findings across the tracked set, or None without git."""
    names = tracked_files(root)
    if names is None:
        return None
    findings = []
    for name in names:
        path = Path(root) / name
        if not path.is_file():
            continue  # tracked but deleted in the working tree
        findings.extend(scan_file(name, path.read_bytes()))
    return findings


def format_findings(findings):
    return "\n".join(
        f"  {f.path}:{f.line}:{f.col}  {f.label}  {f.detail}" for f in findings
    )


def main():
    findings = scan_repo()
    if findings is None:
        print("scan: not a git checkout -- nothing to scan", file=sys.stderr)
        return 1
    if findings:
        print(
            f"scan: {len(findings)} invisible-character finding(s):\n"
            + format_findings(findings),
            file=sys.stderr,
        )
        return 1
    print("scan: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
