---
name: orchestrator
description: Route a task to the appropriate coding model when the user says the literal phrase "do think". Use GPT-5.6 Sol for very complex work, consult Claude Code for complex bounded work, and consult Antigravity for generic work.
---

# Orchestrator

Classify the task before acting. State the selected route and a one-sentence reason, then perform that route.

## Routing

| Task level | Use when the task involves | Route |
| --- | --- | --- |
| Very complex | Cross-cutting architecture, large migrations, difficult multi-system debugging, security-critical changes, or long-horizon implementation with substantial ambiguity | Use Codex. Select GPT-5.6 Sol (`gpt-5.6-sol`) with high or higher reasoning effort when the current surface exposes model selection. |
| Complex | A bounded multi-file feature, non-trivial regression, focused code review, or implementation plan | Invoke `$consult-claude` with a focused, read-only request. Use its findings as advice, then continue the requested work in Codex. |
| Generic | A straightforward question, summary, explanation, small lookup, or otherwise narrow task | Invoke `$consult-antigravity` with a focused, read-only request. Use its findings as advice, then continue the requested work in Codex. |

Prefer the more capable route when the task sits between categories: choose Claude over Antigravity, and Codex/Sol over Claude.

Honor an explicit user model choice unless it is unavailable. Do not claim to switch the active Codex model when the current surface provides no model selector; explain that GPT-5.6 Sol is the recommended selection and proceed with the active model.

## Delegation Rules

- Preserve the user’s goal, constraints, and relevant paths in each consultation prompt.
- Keep Claude and Antigravity consultations read-only. Do not forward secrets or authorize edits through them.
- Treat consultation output as advice. Verify it against the repository before making changes.
- Do not delegate an empty greeting or acknowledgment; answer it directly.
