# RoboTwin2 C2 Open-source Cleanup Implementation Plan

> **Goal:** Reduce the repository to the reproducible 75-query RoboTwin2 C2 training and evaluation path while removing unsupported experimental branches.

## Task 1: Establish the dependency and recovery baseline

- Enumerate the selected training scripts, Python entry points, model modules, configs, deployment paths, and tests.
- Search imports and CLI/config field usage before classifying any source file as removable.
- Record syntax/test baseline.
- Archive the exact removal set outside the repository before deletion.

## Task 2: Create the two-stage public training interface

- Replace the multi-variant launcher with fixed Warmup and Full entry points.
- Fix the method at three cameras and 75 Action/Segmentation queries.
- Keep only operational overrides such as paths, resources, schedule, seed, and logging.
- Make Full consume the Warmup `checkpoint-20000` by default.
- Add early validation with actionable error messages.

## Task 3: Create the public validation and evaluation interface

- Retain and rename the dataset validator.
- Reduce the online server/client scripts to one C2 pair.
- Replace local paths and historical checkpoint names with environment-based configuration.
- Verify the trained checkpoint is loadable through the retained inference path.

## Task 4: Remove scripts and assets outside the selected product path

- Delete query-count and component ablations.
- Delete A3U, tri-plane, historical training launchers, duplicate online launchers, and experimental figures/reports.
- Remove generated outputs and caches from the source tree; add appropriate ignore rules where needed.

## Task 5: Prune unreachable implementation branches

- Remove experiment-only modules after verifying no selected-path imports remain.
- Remove their config fields, CLI arguments, model branches, serialization handling, and tests.
- Simplify C2 components to the selected context/query behavior without changing checkpoint-compatible parameter names needed by Warmup and Full.

## Task 6: Document the minimum reproducible workflow

- Document prerequisites and expected RoboTwin2 dataset modalities.
- Document Warmup and Full commands, directory layout, resuming, and evaluation.
- List the fixed method hyperparameters separately from user overrides.

## Task 7: Verify the cleaned repository

- Run `bash -n` over retained shell scripts.
- Compile retained Python sources.
- Run focused config/model/launcher tests.
- Run the broad test suite that is supported in the current environment.
- Search for deleted module names, obsolete variants, and developer-machine absolute paths.

