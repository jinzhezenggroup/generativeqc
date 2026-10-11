# Decision: reuse scalar Coulomb and relative-pair atomic responses

Status: implemented and qualified on the frozen high-order workload
Date: 2026-10-11

## Problem

The retained order-seven-through-eleven force fallback evaluates a contracted
`Dual3` ERI independently for each participating atom. Its nuclear-response
domain contains N-1 atoms, but every response repeats the same primal geometry,
radial moments and Coulomb recurrence. Full-range f-containing residuals and
the generic LR consumer have large private frames; smaller CTA widths alone do
not eliminate that work.

The first pilot used `a9b991f723ae0482c4496d20c9e2dc0649b6ffe5`, after the
order-three/128-thread LR scheduling work. During qualification master added
order-four-through-six spatial reuse. The accepted paired baseline is the newly
frozen master `858fe52f1057bf77f30ad56f3bef9b6ecf00e353`, including that work.
The pilot is not substituted for this complete endpoint cohort. The decision
does not revisit rejected blanket angular partitions or active-Hermite-only
initialization.

## Decision

Generalize the existing scalar spatial-response view, already qualified by the
dddd consumer, to the instantiated L+1 simplex. Retain the same Cartesian
contraction body and radial moment owners. The component-local consumer fills
one scalar simplex per primitive quartet, then lends its raised states to every
independent atomic response.

At fixed exponents, a one-dimensional pair coefficient depends only on A-B.
Its second-center response is minus its first-center response. Store one `Dual`
per Hermite axis, including the Gaussian response, and scale that response by
`[atom=A]-[atom=B]`. The product-center response is separately computed as
`(alpha*[atom=A]+beta*[atom=B])/(alpha+beta)` and the corresponding ket term.
This also handles repeated atoms without subtracting independently recomputed
large derivatives or dividing a Gaussian factor to recover coefficients.

Bra responses survive all ket products; ket responses and scalar Coulomb states
survive all atomic consumers of one product. No shared-memory allocation, global
derivative tensor, cache lease, new queue, or CPU/reference production consumer
is introduced. New mathematical composition remains compiler-owned.

## Boundaries and invariants

- Preserve AO/shell/source ordering, normalization, signed density coefficients,
  scientific screening, physical primitive domains and translational scatter.
- Select full/LR orders 7--11 by recurrence capability, not method, molecule,
  basis name, device name, or an opt-in. Lower orders retain qualified specialized
  owners; short-range production and optional reachable/convolved AD retain
  their existing owners.
- L+1 must fit the existing maximum Coulomb order 12. Order twelve remains AD;
  do not enlarge a global workspace limit merely to remove that fallback.
- Full Hermite initialization remains intact. No division by combined cache
  coefficients or Gaussian factors, including zero/underflow cases.
- A single-atom quartet does no derivative preparation. Each live AO quartet
  performs one scalar Coulomb preparation per primitive, not N-1 preparations.
- Report capacity, logical primitive contraction and recurrence-preparation
  counts in their own units; never infer executed force work from shell bounds.

## Independent qualification

The host probe covers orders 7--11, three Cartesian orientations, signed
two-primitive contractions, four distinct/repeated-atom layouts, full/LR
operators and omega 0/0.37/10000, with exponent scales 1e-3/1/1e3. Across 4860
atomic-coordinate checks, maximum normalized retained-AD error is 4.52281e-13
and independent displaced-value error is 3.7247e-8. Acceptance remains
3e-11*(1+abs(AD)) and 3e-7*(1+abs(displacement)). It independently checks four
bra and sixteen ket/Coulomb preparations per live AO quartet, and sixteen
contractions per independent atom. These are exact fixture execution counts,
not a complete endpoint primitive-work census.

A fixed-step two-point difference was cancellation-limited for the highly
diffuse unnormalized fixtures. The retained independent oracle uses four-point
differences with step 1e-3/sqrt(exponent_scale), rather than relaxing acceptance.
Full Hermite initialization and zero/underflow Gaussian cases remain covered.

Real-GPU probes cover all five orders, four atom layouts, full/LR and 64 sampled
Cartesian components per case. Retained AD also supplies the omitted atom as
an independent translational check. Exact preparation/contraction counters pass.
Memcheck, initcheck, synccheck and racecheck report zero errors/hazards.

The source-matched Release sm_120 library is
`d1a3e6079f278bf3678c47291c55853a98b1ac28225ac981150d2147d6ce4e54`.
Qualification passes 141 targeted/expanded host cases, 515 compiler modules
with zero dependency errors, six native CUDA suites, separate SR/LR and mixed
s/p/d/f gates, eight independent Libcint J/K cases and eight independent
two-electron FCI physical-force cases. The FCI probe initially skipped because
threadpoolctl was absent; that is retained, not counted as a pass. An isolated
3.7.0 overlay supplies the prerequisite, and the replay requires eight passes.

## Complete endpoints and tradeoff

Fresh owner/density, exact Direct WB97M-V/RKS FP64, complete spherical
def2-TZVPD H/O, grid 48x16x32, Becke3, tolerances 1e-12/1e-10/1e-12,
VV10 density cutoff 1e-8, and 4 GiB host/device force allowances remain fixed.
These are water proxies, not OMol25-distribution samples. Compilation/artifact
caches are preserved; imports/context/Calculator precede timing, while prepare,
SCF, physical forces, synchronization and host publication are inside it.

One candidate-only-primed initial cohort was invalid for throughput: baseline-3
first force took 18.7868 s versus 3.37141 s on repetition. Stop that task's own
job after its baseline-12 endpoint, retain all five completed captures, and
label first baseline 3/12 calls as priming. Rerun the whole ABBA after both
artifacts have been primed; do not manufacture a win by dropping just the slow
baseline point. All subsequent complete samples remain in the receipt.

On node1's Slurm-assigned RTX 5090, two clean ABBA samples/arm give:

- 12 atoms: complete 189.466910 -> 184.278075 s (-2.74%); physical force
  32.991263 -> 27.745079 s (-15.90%). SCF/publication stays 155.542374 ->
  155.535303 s; both native arms retain 21 iterations / 21 Focks and identical
  available semantic work/traffic. Maximum primary physical force error is
  5.014128e-11.
- 3 atoms: complete 11.955799 -> 12.061497 s (+0.88%). Four supplementary
  ABBA cycles, eight samples/arm, give 11.950039 -> 12.078239 s (+1.07%) and
  force 3.374666 -> 3.462618 s (+2.61%). Most of the measured difference is in
  preparation (median 2.073808 -> 2.166398 s); this is not a small-shape speedup
  or a universal non-regression claim.

The retained default decision accepts this explicitly measured small-shape
startup/preparation tradeoff for the high-order force improvement. It does not
hide the unfavorable samples or add a water/functional/basis/atom-count selector.
Further preparation optimization is a separate work item, not an unmeasured
fix folded into this source-qualified producer.

Two fresh full-Fock GPU4PySCF reference endpoints on the same job/device give
median complete 183.499979 s and force 11.032725 s. Native complete time is
near parity, but native force is still about 2.51x slower. Reference retains
45 iterations / 46 Focks, native 21 / 21; no per-Fock equality is implied.
The reference reports CUDA LibXC. This is not a stock-default DF comparison.

No complete 96-atom E/F claim follows. During measurement master advanced to
`98263e06ca13aa88cee504f1f105f5cf3ef6d92a`; an additional 59 host/source numerical
compatibility cases pass there, but no later-master GPU/timing claim is made.

The compact receipt, reconstruction patch and reproduction runner are under
`benchmarks/results/wb97mv-shared-spatial-response-20261011/`. Matched raw work
remains local under `.artifacts/shared-response-latest-20261011/` and on node1
under `/data/jzzeng/wb97m-shared-response-latest-20261011/`. The old a9 pilot
remains separate. All real-GPU work uses finite Slurm jobs with assigned device
visibility; verified ccache is reused. Both qualification builds disable the
unrelated native CLI because node1 lacks its JSON SDK prerequisite.

## Residuals and rejected shortcuts

The intrusive 12-atom profile is numerical-gated but not a throughput sample.
Its ordinary LR force kernel takes 10.274591 GPU seconds and full f-containing
force kernel 8.222631 GPU seconds. These remain the first cross-component/CTA
reuse candidates; prefer one bounded queue traversal over blanket angular
passes. The clean first matched integral-derivative region is 26.417960 ->
20.920601 s, while geometry/pair drain remains about 4.41 s.

Do not equate compact mathematical state with a smaller compiled frame.
Restricted FullSources angular diagnostic symbols retain 255 registers/thread
and 4108 shared bytes/CTA; reported private stacks grow from 26824--65208 to
33368--65688 bytes/thread. Optional AD remains compiled. Device-function REG=0
is unavailable resource metadata, not zero use. These are static symbols, not
measured endpoint peak allocations.

Force preparation also merits investigation: cProfile records 1.214 s in 878
file-hash calls and 1.089 s in stationary compilation/artifact resolution.
Any future reuse of verified identity needs an explicit lifetime/immutability
contract and bounded revalidation fallback, not weaker scientific cache keys.

## Revisit or reject when

Revisit the default if independent physical gates fail, the bounded high-order
complete endpoint regresses, or the explicitly retained small-shape cost grows.
Cross-component CTA reuse, preparation identity reuse and elimination of
single-atom specialized derivative work need their own qualification. Any
extension to lower orders or degree-thirteen responses requires numerical,
resource and complete endpoint evidence, not extrapolation.
