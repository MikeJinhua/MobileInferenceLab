# TemporalNeuralUpscaling Project Rules

This file extends the repository-level `AGENTS.md`.

## Scope and Order

- Follow `SPEC -> PLAN -> TASKS -> IMPLEMENT -> TEST -> REVIEW / UPDATE DOCS`.
- Read both `AGENTS.md` files, `README.md`, and all three SDD documents before implementation changes.
- Work only on the active task in `docs/TASKS.md`.
- Stabilize spatial SR before ExecuTorch, Android, QNN, Vulkan, or temporal work.
- Do not add temporal logic during Phase 1 or pursue SOTA image quality/training complexity.

## Engineering Constraints

- Prefer operators with plausible ExecuTorch/QNN paths, but test compatibility rather than assume it.
- Phase 1 tensors are float32 NCHW RGB.
- Benchmarks state device, warmup, repetitions, shape, and inference-only versus end-to-end scope.
- Later benchmarks separate preprocess, inference, transfer, synchronization, composite, and total latency.
- Keep Android UI minimal and focus on C++, Vulkan, inference runtime, and GPU/NPU interop.
- Do not commit generated models, proprietary SDK contents, large assets, or unlicensed resources.

After each task, update `docs/TASKS.md` with status/date, changed files, commands/results, limitations, and next task.
