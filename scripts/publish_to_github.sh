#!/usr/bin/env bash
set -euo pipefail

# Run this script from the repository root after reviewing the files.
REMOTE_URL="${1:-https://github.com/BfsorDfs6/AgeSafer.git}"
BRANCH="${2:-main}"

if [[ ! -d .git ]]; then
  git init
fi

git branch -M "${BRANCH}"
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "${REMOTE_URL}"
else
  git remote add origin "${REMOTE_URL}"
fi

git add .
if ! git diff --cached --quiet; then
  git commit -m "Release lightweight ML-1M + GMF reference implementation"
else
  echo "No staged changes to commit."
fi

git push -u origin "${BRANCH}"
