---
name: consult-antigravity
description: Consult a locally authenticated Antigravity CLI session for thorough, read-only repository analysis. Use when Codex benefits from an independent code review, implementation plan, debugging investigation, or test strategy before making changes.
---

# Consult Antigravity

Use Antigravity CLI as a read-only specialist. Treat its output as advice: verify it against the repository and make the final implementation decisions yourself.

## Workflow

1. Work from the target repository root.
2. Form a precise request with the goal, relevant paths, constraints, and questions to answer.
3. Run the bundled script:

   ```sh
   .codex/skills/consult-antigravity/scripts/consult-antigravity.sh \
     "Review the authentication flow in src/auth. Identify defects, risks, and focused fixes. Do not edit files."
   ```

4. Review the structured plain-text response and cross-check recommendations before changing files.

## Guardrails

- Keep consultations read-only. Do not ask Antigravity to edit files, run commands, commit, push, or access secrets.
- Provide only necessary repository context; never include API keys, credentials, or private tokens in the prompt.
- The script uses Antigravity CLI's non-interactive print mode and plan execution mode.
- If `agy` is unavailable or unauthenticated, explain the prerequisite and continue with local analysis rather than retrying indefinitely.

## Good Requests

- "Investigate why `tests/api.test.ts` is flaky. Return likely causes and a minimal test plan."
- "Review the proposed migration in `db/migrations/`. Identify compatibility and rollback risks."
- "Map the code paths affected by adding rate limiting to `src/api/`. Return a focused implementation plan."
