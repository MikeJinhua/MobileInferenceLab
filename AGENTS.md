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

## Public Repository Safety

- Never commit credentials, private code, restricted files, proprietary SDKs, or Qualcomm SDK binaries.
- Ignore generated artifacts, large models, local SDKs, and build output.
- Document external SDK installation instead of vendoring SDKs.
- Record source and license for third-party code, models, data, and media.
- Prefer simple, stable, mobile-friendly operators.
- Before implementation, document in `PLAN.md` choices that materially affect ExecuTorch or QNN deployment.
