# MobileInferenceLab Development Rules

These rules apply repository-wide. Project-level `AGENTS.md` files may add stricter rules.

## Workflow

Use lightweight Spec-Driven Development: `SPEC -> PLAN -> TASKS -> IMPLEMENT -> TEST -> REVIEW / UPDATE DOCS`.

Before modifying a project, read the applicable repository/project `AGENTS.md`, project `README.md`, `docs/SPEC.md`, `docs/PLAN.md`, and `docs/TASKS.md`.

- `SPEC.md` is the requirements source.
- `PLAN.md` is the implementation route.
- `TASKS.md` is the current execution record.
- Implement only the current phase and task. Do not over-design for hypothetical needs.
- Keep tasks small and independently verifiable, and keep the project runnable.
- After every task, run tests/validation and update `TASKS.md` with changed files and results.

## Task Ledger Discipline

Every project must maintain its complete execution state in `docs/TASKS.md`.

- Before implementation, add the active task and split its acceptance criteria into independently verifiable Markdown checkboxes (`- [ ]`).
- Mark the task `in progress` while work remains; identify the next task explicitly in the remaining-work section.
- Change a checkbox to `- [x]` only after the corresponding work has objective evidence (test, command output, artifact inspection, or device result). Never pre-check planned or assumed work.
- As each item is verified, update `TASKS.md` during the task rather than reconstructing progress only at the end.
- When a task completes, record its completion date, changed files, exact validation commands/results, limitations, and the next task.
- Keep `SPEC.md`, `PLAN.md`, `TASKS.md`, and implementation status consistent. If scope or deployment choices change, update the applicable documents before or with the implementation.
- Do not mark a phase complete while any required task or acceptance checkbox remains open.

## Public Repository Safety

- Never commit credentials, private code, restricted files, proprietary SDKs, or Qualcomm SDK binaries.
- Ignore generated artifacts, large models, local SDKs, and build output.
- Document external SDK installation instead of vendoring SDKs.
- Record source and license for third-party code, models, data, and media.
- Prefer simple, stable, mobile-friendly operators.
- Before implementation, document in `PLAN.md` choices that materially affect ExecuTorch or QNN deployment.
