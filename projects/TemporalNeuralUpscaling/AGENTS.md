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

## WSL and External Source Discipline

- Keep every project-specific script, source change, test, and learning note created during WSL/Linux work inside this repository (prefer the mounted project path) so it is reviewed, tested, documented, committed, and pushed with the task.
- Before completing a Linux-dependent task, audit external WSL checkouts for project-specific edits and reproduce or sync every intentional edit into the repository. Do not leave the only copy of project logic under `$HOME`, caches, SDK directories, or external build trees.
- Do not vendor upstream ExecuTorch checkouts, Qualcomm SDK files, third-party binaries, generated models, or build output. Record their exact version/commit, purpose, commands, integration points, and any required external patch instead.
- Repository automation must derive external paths from environment variables or documented `$HOME` defaults; never commit a developer-specific absolute path or device serial.

After each task, update `docs/TASKS.md` with status/date, changed files, commands/results, limitations, and next task.
