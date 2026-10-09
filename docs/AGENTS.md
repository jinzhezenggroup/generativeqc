# Documentation instructions

These rules apply to files under `docs/`.

## Organize documentation by audience

- `learn/`: the minimum quantum-chemistry concepts needed to use GenerativeQC correctly.
- `user/`: task-oriented public behavior and workflows.
- `reference/`: authoritative lookup material such as units, names, options, and generated capabilities.
- `developer/`: architecture, numerical algorithms, compiler/code generation, implementation contracts, and extension interfaces.
- `maintainer/`: validation, performance qualification, evidence, ownership, generated artifacts, CI/release/project-health material.
- `agent/`: explanatory workflow for coding agents. Normative instructions remain in repository `AGENTS.md` files.

Keep `docs/index.md` as the stable audience router. Keep this file at the docs root so it continues to scope all documentation.

## Keep docs current-state focused

Documentation should describe the system as it exists now: public behavior, scientific contracts, ownership, supported paths, invariants, setup, and current validation procedures.

Historical benchmark snapshots, migration narratives, discarded designs, and one-time debugging findings belong in `.agents/notes/` when their rationale is worth preserving.

## Update rules

- Update current-state docs in the same change as behavior.
- Preserve non-trivial rationale in Agent Notes rather than chronological prose in current docs.
- Keep measured evidence in docs only when it is a current acceptance criterion or reproducible qualification procedure.
- Prefer stable repository-relative links and commands.
- Do not manually copy generated method/capability tables into prose; link the authoritative reference.
- Keep machine-readable checker inputs and repository inventories under `manifests/` unless they are themselves rendered reference documentation.

## One authority per topic

- A current scientific, API, method-capability or operational rule has one
  authoritative guide/reference owner. Link to that owner rather than
  repeating the same contract under several audiences.
- Distinguish three lifetimes: current behavior and runnable procedures in
  `docs/`; durable choices and rejected alternatives in `.agents/notes/`;
  source-matched raw measurements and past acceptance in
  `benchmarks/results/`. Historical one-time timing/object-size tables should
  not grow indefinitely in a current-state guide.
- When consolidating, preserve existing public documentation paths as a
  concise current contract or forwarding page. Verify links, toctrees,
  code references and generated-ledger evidence paths before deleting
  or relocating material.
- Do not write issue state or production-gate status from a stale snapshot.
  Confirm the current issue/PR state, name the measurement source SHA,
  and distinguish delivered structural work from unresolved numerical
  or runtime qualification.
- Avoid making a new summary page the authority for facts owned by the
  generated method manifest, public API reference or scientific CUDA ledger.
  Keep the existing Sphinx warnings-as-errors and documentation tests.
