# Decision: separate portable DF derivative lowerings from device tuning

Status: implemented
Date: 2026-10-31

## Problem

The derivative manifest registered Rys/compact for SM120 only. On A100 the
runtime therefore selected polynomial derivatives, even though the generated
Rys arithmetic and subgroup schedule do not use SM120-specific instructions.
Device tuning coverage had inadvertently become mathematical availability.

## Decision

Add an optional `portable` profile to the compiler-owned derivative manifest.
The generated selector checks exact target/class rows first, then portable
rows for any nonzero CUDA architecture. Zero remains the explicit legacy or
unprobed sentinel. Missing mathematical classes retain polynomial fallback.
The generated `rys_available` registry and schedule bounds still constrain
kernel instantiation; this does not introduce another integral implementation.

Portable profiles have the same independent-evidence and candidate/baseline
requirements as exact profiles. `candidate` can execute an unqualified portable
profile; `auto` requires qualification. Existing exact tuning stays authoritative
and explicit schedule controls continue to work. Runtime/driver errors remain
errors, rather than being hidden by switching mathematical algorithms.

An unqualified exact-target experiment without a qualified baseline cannot
shadow the portable default under `auto`; it wins only under `candidate`.
Executable selector tests cover both qualification states and arbitrary CUDA
architectures. This final boundary fix emits byte-identical C++ for the measured
production manifest, whose exact-target rows are already qualified.

The portable default uses the existing legal compact schedules. Their original
SM120 evidence is retained as origin provenance, not relabelled as a measurement
on every architecture. A portable default is not a claim of universally optimal
batch sizes, occupancy, or endpoint performance. Other CUDA devices may have
their own measured overrides without requiring a change to mathematical
availability. This CUDA policy does not claim an AMD/CPU runtime implementation.

## Evidence and current limits

Executable host policy/dispatcher tests exercise arbitrary unregistered CUDA
architectures, exact-profile precedence, explicit legacy, missing classes and
unchanged evidence gates. The device admission suite compares panel, grouped
and packet consumers with independent PySCF RHF/UHF forces, including practical
auxiliary f shells and the explicit portable candidate. It accepts the real
Slurm or LSF allocation identity without synthesizing either scheduler's fields.

The 23 schedules require at most 128 threads and a declared shared-memory bound
of 24,288 bytes. The generator's existing 48-KiB legality ceiling remains intact.
No architecture-specific instructions or new arithmetic were introduced.

A100 and both RTX 5090/PRO 6000 passed 18 independent device tests each, including
RHF/UHF panel/group/packet execution, practical auxiliary-f shells, ordinary
hybrid RKS forces, moved geometry and constrained-memory fallback. Native source
transport/controller tests cover eight compiled profiles. Each device completed
32 README DF populations at 3/12/48/96 atoms, with one cold and three warm calls.
The 48/96-atom def2-SVP cases exercise 384/768 orbital functions. The original
SM120 384/768 controls remain separately identified historical evidence.

Each matrix retains 27 qualified and five failed independent default-grid gates:
four HF fitting-energy failures and M06-2X DF3's grid-force failure. No acceptance
gate changed. Paired energies/forces agree with previous implementations to
2.28e-12 Eh / 3.96e-12 Eh/Bohr. A100 PBE0 DF96 improved from 7.163701 to 4.250577
seconds with coalesced J/K plus portable Rys and the original grid tile/cap.
The Blackwell matrices use automatic exact SM120 tuning; their preceding full
matrices forced the Direct portable profile. This distinction prevents treating
that comparison as an isolated coalescing ablation. No warm increase above 5%
was observed in those comparisons. Initial small A100 timing increases did not
reproduce in two fresh ABBA rounds, retained in full.

Both Blackwell devices pass nine unfiltered memcheck tests. A100 passes nine
memory checks and three race checks with `--report-api-errors no`: CUDA 12.9
cuBLAS on driver 535 makes an unsupported `cuCtxGetLimit` feature query, reproduced
by standalone cuBLAS create/destroy with successful creation. The unfiltered
errors and vendor probe remain retained; this is not an all-API-errors pass.

The [evidence bundle](../../../../benchmarks/results/ecnu-portable-df-20261031/README.md)
records final automatic-selection integration separately from the candidate
measurements, including source/binary identities and post-measurement formatting.
Large-grid experiments are a distinct rejected arm and are not production code.
Raw source-matched receipts remain under `.artifacts/a100-catchup-20261031/`.

Integration with master `2b64a412e` adds its source-bound native Libxc catalog.
The PR's changed compiler/native inputs invalidate its old exact transition
tokens. Compare against the last retained summary before refreshing them:
all 221 changes are identity changes, with zero new stage demotions and exactly
unchanged qualification counts. Native stages were already unqualified and the
historical campaign remains stale, with zero current observed native contexts.
Retain that summary as the new previous baseline, regenerate the exact tokens
and snapshot, and run the unchanged catalog checker. This does not requalify
historical Libxc measurements or alter public structural functional selection.

## Rejected alternatives

Adding an SM80-only entry repeats the whitelist defect. Unconditionally treating
SM120 qualification as universal optimal tuning hides missing evidence. Removing
the legacy polynomial path would lose an explicit control and the fallback for
classes without a generated Rys lowering.

## Revisit when

Introduce measured target overrides when complete endpoints demonstrate a
benefit; do not turn missing tuning into a reason to disable portable math.
