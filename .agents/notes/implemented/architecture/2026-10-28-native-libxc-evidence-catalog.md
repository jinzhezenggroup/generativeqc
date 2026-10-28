# Decision: native Libxc measurements feed the existing evidence DAG

Status: implemented
Date: 2026-10-28

## Problem

The canonical bulk capability inventory had exact stage/endpoint resolution and
revision checks, but the current native CPU/CUDA force campaigns were independent
capsules. Import/compilation counts could be confused with molecular evidence.
The old production-domain producer also executes a historical raw candidate;
its GGA/tau-MGGA recipe is not the installed automatic work-domain worker.

## Decision

Keep ingestion and qualification in offline `tools/` code. Use the existing
capability DAG, endpoint resolver, summary renderer and exact regression tokens.
Do not make a second selector allow-list or import reference code into production.

Reuse the independent CPU full-force and CUDA prepared-lifecycle producers.
Retain previously assertion-only native SCF/fixed-density gates and actual loaded
module identities, so CI can reject incomplete coverage or mismatched objects.
Bind a campaign to compiler/frontend/native input bytes and the actual cache-
wrapped build, loaded ELF and native callback owners. Different source inputs
remove claims and expose blockers; historical measurements are never restamped.

The native boundary adapter uses the existing profile and numerical cases, the
same raw independent Libxc oracle and original tolerances. It executes the current
common point emitter, with LDA's explicit U1 restricted worker and the existing
P2 equal-spin differential for other ingredients. It preserves the existing
compiler control cases and independently checks native invalid-input rejection.
This measures the installed mathematical recipe without changing its equations.
Physical empty-spin particle birth and second derivatives remain separate gates.

A current-tree report contains both measured scopes and qualified DAG endpoints.
It records only the exact molecule/basis/grid/geometry/spin/precision/Hamiltonian
scope. Full-catalog AOT builds and historical timing claims are not molecular
evidence. The retained previous summary is a real catalog snapshot, not a
hand-written support matrix. Updating a changed report requires rotating that
previous summary; an older intrinsic-only baseline cannot hide a later demotion.

## Rejected alternatives

- Attaching only stage pass labels: loses numerical, binary and coverage checks.
- Reusing historical successful graphs/binaries under current identities: invalid
  after source, domain, derivative ABI, parameter/schedule or loader changes.
- Treating old candidate boundary failures as current worker failures, or silently
  replacing their identities: mixes distinct work-domain recipes.
- Wiring evidence into public selection: would reverse structural default-allow.
- Adding production orchestration: the native prepared owner already owns runtime
  admission, convergence, derivatives and resources.

## Evidence and maintenance

The reproducible campaign, its exact build identity, current report and baseline
are referenced by `manifests/libxc/native-qualification.json`. The checker is a
normal CI step, and tests mutate real retained receipts, not production source
spelling or generated-source fingerprints. Scientific input digests invalidate
claims; they do not forbid refactoring. A reviewed demotion uses exact tokens.

Current execution domains remain documented in `docs/user/libxc.md`; the generated
report is the sole count table. Revisit the conservative input inventory when a
native owner provides a narrower, equally complete execution identity, or when
independent response/expanded-context producers become available.
