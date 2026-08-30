#!/bin/sh
# uninstall.sh -- removes our symlinks only if they point at THIS repo.
# Never touches ~/.config/venice (would delete credentials).

set -eu

# This bootstrap is intentionally kept in sync with install.sh.  It cannot
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

# Remove the bash completion install.sh wrote -- but only if it's ours (carries
# the generated `complete -F _venice venice` line), never a hand-placed file.
COMPL_DST="${XDG_DATA_HOME:-$HOME/.local/share}/bash-completion/completions/venice"
if [ -f "$COMPL_DST" ] && grep -q "complete -F _venice venice" "$COMPL_DST" 2>/dev/null; then
    rm -f "$COMPL_DST"
    echo "removed  $COMPL_DST"
elif [ -e "$COMPL_DST" ]; then
    echo "skip     $COMPL_DST (not ours)"
fi

echo
echo "Credentials at $HOME/.config/venice/credentials were NOT removed."
echo "To remove the file (secure overwrite is not portable):"
echo "  rm -f ~/.config/venice/credentials"
echo "Then remove the directory if it is empty:"
echo "  rmdir ~/.config/venice"
