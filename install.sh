#!/bin/sh
# install.sh -- idempotent symlink installer for `venice`.
# Re-runnable. Replaces stale symlinks. Refuses to clobber real files.

set -eu

# This bootstrap is intentionally kept in sync with uninstall.sh.  It cannot
# live in a sourced helper: finding that helper safely already requires
# resolving this script through any symlinks used to invoke it.
resolve_script() {
    self="$1"
    hops=0
    while [ -L "$self" ]; do
        hops=$((hops + 1))
        if [ "$hops" -gt 40 ]; then
            echo "unable to resolve script path: too many symlinks" >&2
            return 1
        fi
        self_dir="$(CDPATH= cd -P "$(dirname "$self")" && pwd)" || return 1
        target="$(readlink "$self")" || return 1
        case "$target" in
            /*) self="$target" ;;
            *) self="$self_dir/$target" ;;
        esac
    done
    self_dir="$(CDPATH= cd -P "$(dirname "$self")" && pwd)" || return 1
    printf '%s/%s\n' "$self_dir" "$(basename "$self")"
}

SCRIPT="$(resolve_script "$0")"
REPO="$(CDPATH= cd -P "$(dirname "$SCRIPT")" && pwd)"

BIN_SRC="$REPO/bin/venice"
PKG_SRC="$REPO/src/venice"
BIN_DST="$HOME/.local/bin/venice"
LIB_DST="$HOME/.local/lib/venice"
CFG_DIR="$HOME/.config/venice"

[ -f "$BIN_SRC" ] || { echo "missing: $BIN_SRC" >&2; exit 1; }
[ -d "$PKG_SRC" ] || { echo "missing: $PKG_SRC" >&2; exit 1; }
[ -x "$BIN_SRC" ] || chmod +x "$BIN_SRC"

mkdir -p "$HOME/.local/bin"
mkdir -p "$HOME/.local/lib"

if [ ! -d "$CFG_DIR" ]; then
    mkdir -p "$CFG_DIR"
    chmod 700 "$CFG_DIR"
    echo "created  $CFG_DIR (mode 0700)"
else
    chmod 700 "$CFG_DIR"
fi

link() {
    src="$1"; dst="$2"
    if [ -L "$dst" ]; then
        current="$(readlink "$dst")"
        if [ "$current" = "$src" ]; then
            echo "ok       $dst -> $src"
            return 0
        fi
        rm "$dst"
        ln -s "$src" "$dst"
        echo "updated  $dst -> $src  (was: $current)"
    elif [ -e "$dst" ]; then
        echo "REFUSE   $dst exists and is not a symlink -- remove it manually" >&2
        return 1
    else
        ln -s "$src" "$dst"
        echo "linked   $dst -> $src"
    fi
}

link "$BIN_SRC" "$BIN_DST"
link "$PKG_SRC" "$LIB_DST"

# Bash completion (best-effort; source installs only). pip users instead run
# `source <(venice completion bash)`. Never fatal: a missing dir or unwritable
# path just skips it (a partial write on failure is cleaned up).
COMPL_DST="${XDG_DATA_HOME:-$HOME/.local/share}/bash-completion/completions/venice"
COMPL_OWNER="# venice source completion owner: $REPO"
if mkdir -p "$(dirname "$COMPL_DST")" 2>/dev/null \
   && { printf '%s\n' "$COMPL_OWNER"; "$BIN_SRC" completion bash; } \
      > "$COMPL_DST" 2>/dev/null; then
    echo "wrote    $COMPL_DST  (bash completion)"
else
    rm -f "$COMPL_DST" 2>/dev/null || true
    echo "skip     bash completion (run: source <(venice completion bash))"
fi

echo
echo "Done. Try: venice --help"
echo "First-time setup: venice login"
echo "Shell completion: source <(venice completion bash)   # or zsh"
