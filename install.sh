#!/bin/sh
# install.sh -- idempotent symlink installer for `venice`.
# Re-runnable. Replaces stale symlinks. Refuses to clobber real files.

set -eu

SCRIPT="$(readlink -f "$0")"
REPO="$(cd "$(dirname "$SCRIPT")" && pwd)"

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

echo
echo "Done. Try: venice --help"
echo "First-time setup: venice login"
