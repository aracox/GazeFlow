#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -eq 0 ]; then
  printf 'Usage: %s "consultation request"\n' "$0" >&2
  exit 64
fi

if ! command -v agy >/dev/null 2>&1; then
  printf 'Antigravity CLI is not installed or unavailable on PATH. Install and authenticate it first.\n' >&2
  exit 127
fi

request="$*"
prompt="You are providing a read-only engineering consultation. Analyze the current repository for the request below. Do not modify files, run commands, commit, push, access secrets, or suggest broad unrelated refactors. Return sections for findings, relevant paths, risks, and a focused next-step plan.\n\nRequest:\n${request}"

agy \
  --mode plan \
  --print "$prompt"
