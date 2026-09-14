#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -eq 0 ]; then
  printf 'Usage: %s "consultation request"\n' "$0" >&2
  exit 64
fi

if ! command -v codex >/dev/null 2>&1; then
  printf 'Codex CLI is not installed or unavailable on PATH. Install and authenticate it first (e.g. `brew install codex`, then `codex login`).\n' >&2
  exit 127
fi

request="$*"
prompt="You are providing a read-only engineering consultation. Analyze the current repository for the request below. Do not modify files, run commands that mutate state, commit, push, access secrets, or suggest broad unrelated refactors. Return concrete findings, relevant paths, risks, and a focused next-step plan.

Request:
${request}"

# Capture just the agent's final message so callers get a clean answer.
last_message="$(mktemp)"
trap 'rm -f "$last_message"' EXIT

codex exec \
  --sandbox read-only \
  --skip-git-repo-check \
  -c model_reasoning_effort="high" \
  --color never \
  --output-last-message "$last_message" \
  "$prompt" </dev/null >&2

cat "$last_message"
