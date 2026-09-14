#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -eq 0 ]; then
  printf 'Usage: %s "consultation request"\n' "$0" >&2
  exit 64
fi

if ! command -v claude >/dev/null 2>&1; then
  printf 'Claude Code CLI is not installed or unavailable on PATH. Install and authenticate it first (e.g. `npm i -g @anthropic-ai/claude-code`, then `claude login`).\n' >&2
  exit 127
fi

request="$*"
prompt="You are providing a read-only engineering consultation. Analyze the current repository for the request below. Do not modify files, run commands, commit, push, access secrets, or suggest broad unrelated refactors. Return concrete findings, relevant paths, risks, and a focused next-step plan.

Request:
${request}"

claude -p \
  --model sonnet \
  --effort high \
  --max-turns 30 \
  --permission-mode plan \
  "$prompt" </dev/null
