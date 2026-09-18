#!/usr/bin/env bash
set -euo pipefail
STASH='stash@{0}'
BASE="$1"
BRANCH="$2"
shift 2
git checkout -b "$BRANCH" "$BASE"
while [[ $# -ge 2 ]]; do
  FILE="$1"
  MSG="$2"
  shift 2
  if git show "${STASH}^3:${FILE}" &>/dev/null 2>&1; then
    git checkout "${STASH}^3" -- "$FILE"
    git add -A -- "$FILE"
  elif git diff "${STASH}^1" "${STASH}" --name-status -- "$FILE" 2>/dev/null | rg -q '^D'; then
    git rm -f "$FILE" 2>/dev/null || git rm -f --cached "$FILE" 2>/dev/null || true
    if [[ -f "$FILE" ]]; then git rm -f "$FILE"; fi
  else
    git checkout "${STASH}" -- "$FILE"
    git add -A -- "$FILE"
  fi
  git commit -m "$MSG"
done
echo "Branch $BRANCH: $(git rev-list --count ${BASE}..HEAD) commits"
