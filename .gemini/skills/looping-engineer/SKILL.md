---
name: looping-engineer
description: "Take a software-engineering goal through a complete, verified delivery loop: analyze scope, define tasks, route focused research to Codex or Claude Code, implement, verify, and report the result. Use when the user asks to complete a goal end to end or explicitly invokes looping-engineer."
---

# Looping Engineer

Own the goal until it is delivered or a real blocker requires user direction.

## 1. Define Done

1. Restate the goal as observable success criteria.
2. Inspect relevant repository guidance and code before proposing changes.
3. List the smallest task board needed to finish, giving every item an owner and verification check.
4. Ask one concise question only when a missing decision materially changes the solution. Otherwise, state the assumption and proceed.

Use this format in commentary when the work has multiple steps:

```text
- [ ] Task — owner: <Antigravity|Codex|Claude> — verify: <check>
```

## 2. Route Work

Keep Antigravity/Gemini as the implementation owner. Delegate only analysis or review that makes the next decision better.

| Task type | Owner | Action |
| --- | --- | --- |
| Any implementation, no matter how complex, cross-cutting, security-sensitive, or ambiguous | Antigravity | Implement it yourself. Ask the user only when a decision materially changes product behavior. |
| Bounded complex architecture, debugging, test-strategy check, or independent code review | Codex | Invoke the `consult-codex` skill with a precise, read-only question and use the result as advice. |
| Generic reconnaissance, simple explanation, narrow lookup, or a second opinion | Claude | Invoke the `consult-claude` skill with a precise, read-only question and use the result as advice. |

Do not delegate the same implementation to multiple agents. Do not send credentials, secrets, or unnecessary repository content to a consultant.

## 3. Execute the Loop

Repeat until all task-board items meet their verification checks:

1. Choose the next incomplete task.
2. Gather only the context needed for that task.
3. Consult the selected specialist when the routing table calls for it.
4. Implement the smallest change that meets the task's success criterion.
5. Run the focused verification, then the project's relevant full check when available.
6. Inspect the result and mark the task complete only with evidence.

Fix failures and rerun the relevant checks. Do not stop after producing a plan, partial implementation, or unverified claim. Preserve unrelated user changes and follow repository guidance. Ask the user before external coordination, destructive actions, or a choice that materially changes product behavior.

## 4. Deliver

Report the outcome, the key changed files, and the verification performed. Clearly name any remaining limitation or blocker. Do not claim completion until the requested success criteria are satisfied.

## Relationship to the other `looping-engineer` skills

This is the mirror of `.codex/skills/looping-engineer` and `.claude/skills/looping-engineer`, each keeping its own CLI as the implementation owner and routing to the other two for advice. All three loops are otherwise identical: define done, route work, execute, deliver.
