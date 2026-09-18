#!/usr/bin/env bash
# Apply single file from stash@{0} and commit with message.
set -euo pipefail
STASH='stash@{0}'
FILE="$1"
MSG="$2"

if git show "${STASH}^3:${FILE}" &>/dev/null 2>&1; then
  git checkout "${STASH}^3" -- "$FILE"
  git add -A -- "$FILE"
elif git diff "${STASH}^1" "${STASH}" --name-status -- "$FILE" 2>/dev/null | rg -q '^D'; then
  git rm -f "$FILE"
else
  git checkout "${STASH}" -- "$FILE"
  git add -A -- "$FILE"
fi

git commit -m "$MSG"
