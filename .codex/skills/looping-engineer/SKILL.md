---
name: looping-engineer
description: "Take a software-engineering goal through a complete, verified delivery loop: analyze scope, define tasks, route focused research to Codex, Claude Code, or Gemini/Antigravity, implement, verify, and report the result. Use when the user asks to complete a goal end to end or explicitly invokes $looping-engineer."
---

# Looping Engineer

Own the goal until it is delivered or a real blocker requires user direction. Use concise reasoning summaries; do not expose private chain-of-thought.

## 1. Define Done

1. Restate the goal as observable success criteria.
2. Inspect relevant repository guidance and code before proposing changes.
3. List the smallest task board needed to finish, giving every item an owner and verification check.
4. Ask one concise question only when a missing decision materially changes the solution. Otherwise, state the assumption and proceed.

Use this format in commentary when the work has multiple steps:

```text
- [ ] Task — owner: <Codex|Claude|Gemini> — verify: <check>
```

## 2. Route Work

Keep Codex as the implementation owner. Delegate only analysis or review that makes the next decision better.

| Task type | Owner | Action |
| --- | --- | --- |
| Very complex, cross-cutting, security-sensitive, or highly ambiguous implementation | Codex | Use GPT-5.6 Sol when the current Codex surface makes it selectable; otherwise continue with the active Codex model. |
| Bounded complex architecture, debugging, test strategy, or independent code review | Claude | Invoke `$consult-claude` with a precise, read-only question and use the result as advice. |
| Generic reconnaissance, simple explanation, narrow lookup, or second opinion | Gemini | Invoke `$consult-antigravity` with a precise, read-only question and use the result as advice. |

Use `$orchestrator` when the route is unclear. Do not delegate the same implementation to multiple agents. Do not send credentials, secrets, or unnecessary repository content to a consultant.

## 3. Execute the Loop

Repeat until all task-board items meet their verification checks:

1. Choose the next incomplete task.
2. Gather only the context needed for that task.
3. Consult the selected specialist when the routing table calls for it.
4. Implement the smallest change that meets the task's success criterion.
5. Run the focused verification, then the project’s relevant full check when available.
6. Inspect the result and mark the task complete only with evidence.

Fix failures and rerun the relevant checks. Do not stop after producing a plan, partial implementation, or unverified claim. Preserve unrelated user changes and follow repository guidance. Ask the user before external coordination, destructive actions, or a choice that materially changes product behavior.

## 4. Deliver

Report the outcome, the key changed files, and the verification performed. Clearly name any remaining limitation or blocker. Do not claim completion until the requested success criteria are satisfied.
