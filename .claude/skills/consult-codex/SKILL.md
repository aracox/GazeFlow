---
name: consult-codex
description: Consult a locally authenticated OpenAI Codex CLI session for a read-only second opinion — code review, implementation plan, debugging investigation, or test strategy. Use when an independent analysis from Codex (GPT models) would strengthen a decision before Claude makes changes.
---

# Consult Codex

Use the OpenAI Codex CLI as a read-only specialist. Treat its output as advice: verify it against the repository and make the final implementation decisions yourself.

## Workflow

1. Work from the target repository root.
2. Form a precise request that includes the goal, relevant paths, constraints, and questions to answer.
3. Run the bundled script:

   ```sh
   .claude/skills/consult-codex/scripts/consult-codex.sh \
     "Review the authentication flow in src/auth. Identify defects, risks, and focused fixes. Do not edit files."
   ```

4. The script prints Codex's final answer to stdout (its progress/reasoning goes to stderr). Cross-check recommendations before changing files.

## Guardrails

- Keep consultations read-only. Do not ask Codex to edit files, run mutating commands, commit, push, or access secrets.
- Provide only necessary repository context; never include API keys, credentials, or private tokens in the prompt.
- The script runs `codex exec` with the `read-only` sandbox, high reasoning effort, `--skip-git-repo-check` (this template is not a git repo), and captures only the final message. The model comes from your Codex config default (`~/.codex/config.toml`); pass `-m <model>` inside the script if you need to pin one.
- If `codex` is unavailable or unauthenticated, explain the prerequisite (`brew install codex`, then `codex login`) and continue with local analysis rather than retrying indefinitely.

## Good Requests

- "Investigate why `tests/api.test.ts` is flaky. Return likely causes and a minimal test plan."
- "Review the proposed migration in `db/migrations/`. Identify compatibility and rollback risks."
- "Map the code paths affected by adding rate limiting to `src/api/`. Return a focused implementation plan."

## Relationship to the `.codex/` skills

This is the mirror of `.codex/skills/consult-claude` (which lets Codex consult Claude). Together they let each assistant get an independent second opinion from the other, always read-only.
