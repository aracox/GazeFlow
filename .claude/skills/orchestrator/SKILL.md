---
name: orchestrator
description: Route a task to the appropriate coding model when the user says the literal phrase "do think". Use Claude itself for very complex work, consult Codex for complex bounded work, and consult Antigravity for generic work.
---

# Orchestrator

Classify the task before acting. State the selected route and a one-sentence reason, then perform that route.

## Routing

| Task level | Use when the task involves | Route |
| --- | --- | --- |
| Very complex | Cross-cutting architecture, large migrations, difficult multi-system debugging, security-critical changes, or long-horizon implementation with substantial ambiguity | Implement it yourself. Use the highest reasoning effort the current surface exposes. |
| Complex | A bounded multi-file feature, non-trivial regression, focused code review, or implementation plan | Invoke the `consult-codex` skill with a focused, read-only request. Use its findings as advice, then continue the requested work yourself. |
| Generic | A straightforward question, summary, explanation, small lookup, or otherwise narrow task | Invoke the `consult-antigravity` skill with a focused, read-only request. Use its findings as advice, then continue the requested work yourself. |

Prefer the more capable route when the task sits between categories: choose Codex over Antigravity, and doing it yourself over Codex.

Honor an explicit user model choice unless it is unavailable.

## Delegation Rules

- Preserve the user's goal, constraints, and relevant paths in each consultation prompt.
- Keep Codex and Antigravity consultations read-only. Do not forward secrets or authorize edits through them.
- Treat consultation output as advice. Verify it against the repository before making changes.
- Do not delegate an empty greeting or acknowledgment; answer it directly.

## Relationship to the `.codex/` skill

This is the mirror of `.codex/skills/orchestrator` (which keeps Codex as the implementation owner and routes to Claude/Antigravity instead). Both fire on the same trigger phrase and follow the same routing logic, swapped for who's holding the pen.
