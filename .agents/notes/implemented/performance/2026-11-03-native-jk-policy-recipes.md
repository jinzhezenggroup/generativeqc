# Decision: isolated endpoint-qualified native DFT J/K recipes

Status: implemented
Date: 2026-11-03

## Problem

The native independent Coulomb/exchange owner already has several exact
lowerings and task schedules. HF shell-class profiling is not a KS capability,
and XC tile identities cannot describe native J/K execution policy. Many
existing controls are process-scoped environment variables, not per-Calculator
state. Temporarily mutating those variables inside Calculator would leak policy
between concurrent prepared owners and make runtime evidence ambiguous.

## Decision

Extend the explicit tuner with `--component jk`. Search only existing audited
native full-range execution controls in fresh child processes. Keep the
Hamiltonian, spin, grid, precision, screening, derivatives and XC schedule fixed.
Require observed complete J/K application counts, unchanged semantic work,
independent PySCF/libxc E/F checks, native allocation admission and all-sample
complete-endpoint gates. A selected proposal must also beat the original
incumbent in a fresh holdout, not just its last incremental comparison.

Publish an explicit exact-workload recipe and adjacent raw evidence, never an
automatic process-global profile. Replay revalidates the raw holdout, workload,
allowlisted policy, library bytes and actual device/source identity before
executing the same native complete endpoint in a child. Preserve Slurm visibility
and do not mutate the parent's environment. Existing native capability/resource
fallbacks remain in force; requested policies are not resolved kernel census.

## Rejected alternatives

- Reuse HF profiling flags: KS rejects them and they do not own native J/K work.
- Mutate environment variables during ordinary Calculator preparation: unsafe
  across concurrent callers and native policy caches/lifetimes.
- Transfer a molecule's winner to arbitrary geometries/bases/batches: complete
  endpoint profitability is workload-specific.
- Search precision, screening or fitted/exact approximations: changes science.
- Promote from a J-only/K-only microbenchmark: force and XC work can dominate.
- Trust only stored pass flags on replay: recompute every holdout gate from raw
  samples and the retained independent reference.

## Evidence and invariants

CPU tests exercise successful exact replay, independent/work/memory/performance
failure, fresh holdout rejection, input/binary/evidence tampering, allowlisting,
parent-environment preservation and absent-exchange selection. Opt-in Slurm GPU
tests exercise every candidate for semilocal RKS, hybrid RKS and hybrid UKS
against independent analytic forces. No synthetic fixture is installed as a
production profile. GPU endpoint measurements are retained separately.

Final source-matched Slurm job 1443 selects a J-lowering policy for water's
PBE-RKS, PBE0-RKS and PBE0-UKS complete warm endpoints; fresh incumbent holdout
speedups are 1.159x, 1.103x and 1.115x. All three accepted recipes actually
replay with raw-evidence validation. K candidates are measured but do not
supersede the J winner. Preserve all raw candidate/holdout observations in
`benchmarks/results/dft-local-autotune-20261103/jk-receipts.json.xz`; the
companion `JK.md` records scope and reproducer rather than transferring these
measurements to arbitrary production workloads.

## Revisit when

Native per-owner execution-policy descriptors and resolved-plan diagnostics can
replace process-scoped controls. That can authorize safe ordinary Calculator
profile selection without weakening exact workload and endpoint qualification.
