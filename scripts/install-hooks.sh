#!/usr/bin/env bash
# Install this repo's git hooks.
#
#     ./scripts/install-hooks.sh
#
# .git/hooks is not version-controlled, so a hook committed to the repo does
# nothing until it is linked into place -- which is why this script exists
# rather than a note in the README that everyone forgets.
#
# Symlinked, not copied: an edit to scripts/hooks/ then takes effect without
# anyone remembering to reinstall, and `git status` shows the hook's real
# source. Uninstall by deleting the link in .git/hooks/.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p .git/hooks

for hook in scripts/hooks/*; do
    name="$(basename "$hook")"
    target=".git/hooks/$name"
    if [ -e "$target" ] && [ ! -L "$target" ]; then
        echo "SKIP $name -- a real file is already there, not overwriting it."
        echo "     Move it aside and re-run if you want this repo's version."
        continue
    fi
    ln -sf "../../$hook" "$target"
    chmod +x "$hook"
    echo "installed $name -> $hook"
done

echo
echo "Bypass any hook for one push with: git push --no-verify"
