# Documentation work for agents

The authoritative scope and update rules are in the repository's
[docs/AGENTS.md](https://github.com/jinzhezenggroup/generativeqc/blob/master/docs/AGENTS.md).
Follow that file, the root `AGENTS.md` and the nearest scoped instructions.
Use [documentation home](../index.md) to choose a reader and destination;
do not create a second audience taxonomy in this page.

## Before changing a guide

1. Find the **current-state owner** of the fact: public workflow,
   API/capability lookup, scientific implementation, or operational
   qualification. Link to it instead of copying a second specification.
2. Compare any status sentence with the current implementation and
   issue/PR state. A pinned historical failed gate is not a claim about
   current `master`, and issue closure is not proof of unrelated runtime gates.
3. Put past choices, rejected alternatives and dated observations in a
   purpose-specific `.agents/notes/` decision record; retain raw evidence
   with immutable source/protocol identity in `benchmarks/results/`.
4. When consolidating a page, **preserve its path** for ledger, external and
   older-source references. Summarize the stable contract at that path,
   redirect readers to the authoritative topic and verify relative links.
5. Check the Sphinx build with warnings as errors and the applicable API,
   link or inventory tests. Do not change generated reference data manually.

For examples of the intended split, compare the current
[SCF ownership map](../developer/scf_module_boundaries.md) with its
[historical decision record](../../.agents/notes/implemented/architecture/2026-10-09-scf-decomposition-provenance.md),
and the current
[CUDA ownership rules](../maintainer/cuda_ownership.md) with their
[retirement evidence index](../../.agents/notes/implemented/architecture/2026-10-09-cuda-retirement-provenance.md).
