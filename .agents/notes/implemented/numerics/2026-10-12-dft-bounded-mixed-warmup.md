# Decision: bounded approximate CUDA KS warmup, without default promotion

Status: implemented
Date: 2026-10-12

## Problem

Convergence-only AUTO can spend the full primary iteration budget near its FP32
noise floor before starting the strict solve. A target FP64 energy/density
tolerance is not an appropriate optimization objective for this approximate
accelerator. The objective is complete endpoint time at unchanged final accuracy,
not approximate-stage convergence or isolated FP32 kernel throughput.

## Decision

Add immutable, opt-in prepared-owner diagnostic controls for handoff on the
current maximum absolute AO commutator entry and/or a bounded approximate
iteration count. Residual handoff needs at least two approximate observations;
incumbent approximate convergence and budget exhaustion remain triggers. The
controls do not broaden component precision admission or change final tolerances.
Strict FP64 does not parse these approximate-only controls.

Admit positive early-handoff controls only for restricted owners. Additional OH
open-shell PBE/PBE0/r2SCAN probes passed native convergence but their independent
PySCF reference did not satisfy its pinned strict target, including a bounded
Newton retry with unchanged tolerances. Preserve both failed qualification
receipts rather than relaxing the oracle or treating native convergence as
independent proof. UKS retains unset/zero incumbent handoff criteria and rejects
positive early controls explicitly. This is not removal of existing UKS AUTO.

Clear DIIS and occupation-stabilization history on handoff and retain the full
strict iteration budget. Also reset the previous electronic energy: the first
strict observation has no same-precision predecessor. It cannot pass an energy
stopping test against an approximate energy, even if their rounded totals agree.
Its diagnostic history uses the existing unavailable-energy-change sentinel.
Zero/unset controls therefore preserve incumbent handoff criteria, not the old
cross-precision energy comparison.

Keep the global precision default FP64. Do not promote a component or functional
because AUTO becomes less slow than its own convergence-only implementation.
The frozen DFT-MP-v1 complete E+force and lifetime gates remain authoritative.

## Rejected alternatives

- Wait for FP32 to satisfy the FP64 target: unnecessary approximate tail work.
- Let a warmup residual/cap certify final convergence: no strict physical proof.
- Carry approximate DIIS or energy history into strict stopping: precision epochs
  describe different operators and cannot certify consecutive FP64 observations.
- Promote from energy-only or cold-only speed: forces, same-owner warm replay and
  geometry rebuilds are separate mandatory endpoint boundaries.
- Claim that AUTO lowers exact K, tau or XC algebra: their admission is unchanged;
  the measured build lowers Coulomb J recurrence, not every DFT operator.

## Invariants

- Handoff uses the current physical residual before DIIS extrapolation, not a
  proposed/extrapolated Fock residual. It is the maximum absolute AO entry, not
  the per-spin RMS `ks_diagnostic.physical_residual_max`.
- Nonfinite/negative diagnostics do not authorize handoff. Existing explicitly
  counted nonfinite retries, final audits and bounded resource refusals remain.
- Prepared-owner controls are resolved once. Set them before preparing a new
  owner; changing process environment does not mutate an existing owner.
- Explicit FP64 semantics, precision ABI and scientific kernels remain unchanged.
- An invalid comparator, missing forces/lifetimes, timeout or inaccurate repeat
  invalidates a policy summary; failed samples are not filtered from its median.
- Oracle work stays outside production and timing. Preserve frozen inputs,
  source/patch/library/artifact/runner identities and every attempted outcome.

## Evidence

The source-matched experiment is based on master
`160bf3d2358847ec2e9ee101ba80e0283b117abc`, plus the retained native patch, with
CUDA 12.9.1/SM120 portable CUDA and a Slurm-allocated RTX 5090 on n1. Stationary
s/p/d force AOT artifacts are compiled before timing. Local host compilation uses
verified ccache 4.5.1; runtime cache-miss preparation uses verified sccache 0.16.0.
No maintenance-node n3 GPU is used.

The experiment scans PBE0, PBE and r2SCAN using the frozen spherical def2-SVP
DFT-MP-v1 water grid/geometry, target energy `1e-10`, density `1e-8`, screening
`1e-12`, approximate cap 24 and residual thresholds 1 through `1e-6`.
Independent PySCF 2.14.0 energies and moving-grid central directional differences
at `1e-4`/`5e-5` Bohr supplement strict native full-force comparisons. This is
exploratory evidence, not the contract's finer-grid/full-case product PASS.

Completed water cold screens favor the earliest tested threshold (1), but all
three remain slower than strict FP64. PBE0 five interleaved lifetime pairs also
show a substantial same-owner warm regression, not merely startup variability.
PBE0 water8 strict E+force completes while the first AUTO sample hits its
600-second watchdog; subsequent work is explicitly stopped rather than
discarded or relabeled as passing. Initial attempts with missing force AOT
artifacts are retained as endpoint failures, never energy-only performance wins.

Native policy tests cover legacy criteria, residual/cap bounds and malformed
controls. Slurm GPU tests compare early handoff with an independent energy
oracle, strict physical residual and fresh strict energy history, including a
residual trigger with no cap. See the retained experiment for exact raw times,
work counts, failures, numerical comparisons and reproduction commands.
The final RKS-only guard is rebuilt and passes 21 GPU tests, including explicit
UKS refusal and unchanged incumbent preparation. Timings retain the earlier
exploratory source/binary identities rather than being relabeled as this later
guard revision.

### Larger-atom follow-up on the final guarded binary

Source-matched single-pair screens now cover water16 (48 atoms/384 spherical AOs),
water32 (96/768), caffeine (24/246) and ace_glygly_nme (26/247). Completed PBE0
FP64 E+force endpoints require 16--21 SCF iterations and take 43.50--55.15 s;
caffeine PBE/r2SCAN require 25 iterations and take 50.71/95.92 s. Every AUTO cap-2
screen reaches its 180/240 s process watchdog. Caffeine/PBE0 still times out at
300 s with cap 1. Water32 FP64 also fails the 240 s E+force and 600 s energy-only
watchdogs; its remaining energy-only candidate is explicitly cancelled after
the failed comparator. Unknown work counts stay unknown, not replaced by the
requested cap. These are negative exploratory observations, not paired speed
or independent larger-force qualification.

A finite Slurm GDB diagnostic with synchronous CUDA launches places water8's
long approximate work in the independent mixed-J Direct-J/K kernel. The earlier
asynchronous stack in an XC launch could instead be waiting for preceding GPU
work and must not be treated as proof that XC was the cause. The Direct owner
intentionally refuses generated mixed Fock, resident strict values and MD-J for
the narrower recurrence-only DFT contract: generated mixed Fock also narrows
the density product, which this DFT admission does not allow. Do not remove
those guards to manufacture acceleration. A new recurrence-only generated
owner requires its own numerical/work gates before default promotion; the
strict refinement's source/chunk eligibility must likewise remain legal.

Separate four-thread PySCF water16/PBE0 and caffeine/PBE0 references reach their
600 s watchdogs after entering SCF. The larger tiled exporter preserves exact
points/weights and strict reference targets; no accepted larger oracle is
claimed. The final binary additionally passes 24 Slurm GPU correctness tests,
including cap-1 handoff for PBE/PBE0/r2SCAN against independent water energies.
The original water archives remain unchanged; additional exact captures and
derived work are in `large-captures.json.xz` and `large-summary.json.xz` beside
them. Their README binds the distinct native/runner identities and failures.

## Consequences

The approximate stage can be tuned without demanding FP32 convergence, but is
not automatically beneficial. Extra strict recovery work and warm replay matter
even when only a few approximate Fock builds execute. Larger endpoint timeouts
require diagnosis, not an unsupported claim that tuning a residual alone fixes
preparation or backend costs. This change intentionally does not redesign those
owners or silently override an explicit precision request.

## Revisit when

The dense mixed-J dispatch diagnosis is superseded by the
[canonical-source bug fix](../performance/2026-10-12-dft-mixed-j-canonical-source.md).
The negative measurements here describe the pre-fix binary and remain unchanged.

A source-matched production revision beats FP64 on all mandatory complete
endpoints with the existing independent numerical gates, five interleaved pairs,
warm regression bounds, memory budget and reviewed DFT-MP-v1 receipt. Qualify
validated-warm approximate refusal or backend improvements separately before
using them to justify a broader default.

## References

- `src/dft/mixed_precision_warmup.hpp`
- `docs/developer/ks_diagnostics.md`
- `docs/maintainer/dft_mp_v1_contract.md`
- `benchmarks/dft_mixed_thresholds.py`
- `benchmarks/results/dft-mixed-warmup-20261012/README.md`
