#!/bin/sh
# uninstall.sh -- removes our symlinks only if they point at THIS repo.
# Never touches ~/.config/venice (would delete credentials).

set -eu

SCRIPT="$(readlink -f "$0")"
REPO="$(cd "$(dirname "$SCRIPT")" && pwd)"

unlink_if_ours() {
    dst="$1"; expected="$2"
    if [ -L "$dst" ]; then
        current="$(readlink "$dst")"
        if [ "$current" = "$expected" ]; then
            rm "$dst"
            echo "removed  $dst"
        else
            echo "skip     $dst (points elsewhere: $current)"
        fi
    else
        echo "skip     $dst (not a symlink)"
    fi
}

unlink_if_ours "$HOME/.local/bin/venice" "$REPO/bin/venice"
unlink_if_ours "$HOME/.local/lib/venice" "$REPO/src/venice"

echo
echo "Credentials at $HOME/.config/venice/credentials were NOT removed."
echo "To wipe: shred -u ~/.config/venice/credentials && rmdir ~/.config/venice"
