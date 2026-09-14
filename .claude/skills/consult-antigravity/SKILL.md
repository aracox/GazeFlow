---
name: consult-antigravity
description: Consult a locally authenticated Antigravity CLI (agy, Gemini models) session for a read-only second opinion — code review, implementation plan, debugging investigation, or test strategy. Use when an independent analysis from Antigravity/Gemini would strengthen a decision before Claude makes changes.
---

# Consult Antigravity

Use the Antigravity CLI (`agy`) as a read-only specialist. Treat its output as advice: verify it against the repository and make the final implementation decisions yourself.

## Workflow

1. Work from the target repository root.
2. Form a precise request that includes the goal, relevant paths, constraints, and questions to answer.
3. Run the bundled script:

   ```sh
   .claude/skills/consult-antigravity/scripts/consult-antigravity.sh \
     "Review the authentication flow in src/auth. Identify defects, risks, and focused fixes. Do not edit files."
   ```

4. The script prints Antigravity's response to stdout. Cross-check recommendations before changing files.

## Guardrails

- Keep consultations read-only. Do not ask Antigravity to edit files, run mutating commands, commit, push, or access secrets.
- Provide only necessary repository context; never include API keys, credentials, or private tokens in the prompt.
- The script runs `agy` in non-interactive print mode (`--mode plan --print`), which favors analysis and planning over execution.
- If `agy` is unavailable or unauthenticated, explain the prerequisite and continue with local analysis rather than retrying indefinitely.

## Good Requests

- "Investigate why `tests/api.test.ts` is flaky. Return likely causes and a minimal test plan."
- "Review the proposed migration in `db/migrations/`. Identify compatibility and rollback risks."
- "Map the code paths affected by adding rate limiting to `src/api/`. Return a focused implementation plan."

## Relationship to the other consult skills

This is the mirror of `.codex/skills/consult-antigravity` (which lets Codex consult Antigravity) and sits alongside `.claude/skills/consult-codex` (which lets Claude consult Codex). Together they let each assistant get an independent second opinion from another, always read-only.
