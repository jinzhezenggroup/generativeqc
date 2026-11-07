# Rejected: retain immutable selected AO panels across cold XC work

Status: rejected; prototype and default selector removed
Date: 2026-11-07

## Problem and measured boundary

The source-bound current A100 diagnostic LSF227018 uses native source0644bc07
and ELF2dd1887b from PR3033. Its intrusive complete cold endpoint is82.543947s,
with16 Focks (11 provisional/5 target), four SCF VV10 sweeps and one frozen
warm Fock. CUDA-event scopes attribute33.083123s to XC, including21.369741s
to nonlocal pair features, while immutable Direct-source preparation takes
14.327907s. J/K and stationary-force scopes are11.528867s and20.770240s.
These nested scopes must not be added to the endpoint as independent costs.

The older226178 diagnostic's16.414s `structured_basis` intervals belong to
full-range J/K matrix projection, not XC AO evaluation. Composite exchange
already removes that route. They do not establish today's AO-cache opportunity.
The proposed cache targets repeated geometry work within the remaining XC
cost; only complete same-image endpoints can establish its value.

## Candidate and boundaries

Keep exactly the existing tile-major selected FP64 AO/jet values. Prepare a
complete-tile prefix only after effective composite Direct-source admission.
Reuse it for current-density features and nonlocal AO pullback, without caching
any density-dependent output or changing point/reduction order. A batch that
straddles the prefix uses its ordinary producer; the nonlocal tile consumer can
use every admitted prefix tile. The same scheduled AO evaluator supplies both
paths. Maps, geometry, basis and jet layout remain immutable.

Admission is resource-based, not a GPU-model rule: at most8GiB, at most half
the actual `cudaMemGetInfo` free bytes, and no consumption of the known following
force quote. `device_resource_available` is a request allowance, not free VRAM;
without a ledger it returns SIZE_MAX. The initial compile-only b4cde3c8 image
predates that correction and is not a GPU performance qualification.

OOM retains the bounded ordinary producer. Numerical/driver failure drains
and releases the unpublished lease before propagating. Force pressure retires
this optional geometry cache before the higher-priority ERI lease. Captured
borrowers must be invalidated before retirement; density generations and
published XC outputs remain valid. Mixed, response, public-ledger, independent
grid and graph/chunk owners retain their existing production routes. Explicit
`GENERATIVEQC_CUDA_XC_AO_CACHE=0` is the same-image incumbent control.

## Required evidence

Independent CPU E/V and nonlocal cotangent contraction; RKS/UKS and displaced
Cartesian/spherical f shells; dense/nonidentity/empty maps, tails, partial
budgets and different point batching; exact changed-density replay and fallback
after retirement; transactional OOM/driver/numerical/capture refusal; complete
A100/RTX5090 cold energy+analytic-force+host-export and own-density warm
nonregression with unchanged scientific work and controls. No microbenchmark
or old `structured_basis` subtraction substitutes for those gates.

## Actual outcome and rejection

Measured source5efc4cf4 / ELF5a933173 is frozen over9ec79b779, the current
PR3033 head including its CI-only radial-shim repair. A100 LSF227134 exits0:
complete cold82.695270 ->82.042750s (-0.79%,0.652520s saved), and three-repeat
own-density warm median13.816298 ->13.686628s (-0.94%). Both arms retain
16 cold Focks (11 provisional/5 target), four SCF pair sweeps, one warm Fock,
zero warm pair sweeps and identical complete composite Direct construction.
Independent static E/F and tighter paired-native gates pass. These are one
cold sample per arm, not a latency distribution or a fresh gpu4pyscf ranking.

The cache retains6,989,834,240 bytes on A100. RTX5090 Slurm16644 passes the
independent native cached E/V/features/nonlocal-pullback/retirement probe and
memcheck with0 errors, but its complete candidate endpoint fails with OOM.
Its completed rollback is168.310181s cold and12.755009s median warm. There
is no candidate endpoint timing. Separate intrusive Slurm16645 reproduces the
failure after entering stationary forces with6,280,069,120 cached AO bytes.
A quoted following consumer is not a complete reservation for subsequent
provider lifetimes; the force-pressure-only retirement boundary is inadequate.

An unconditional cold-SCF-to-force retirement correction builds as a distinct
source de9848f4 / ELF d0255e88. Its921-case scoped host owner/force cohort
passes, but no real-GPU job executes that image. Pending qualification drivers
are stopped before submission once the original A100 cost/benefit is known.
Do not call the compiled correction a validated fix or reuse5efc timings for it.

Reject rather than retain an opt-in or publish an unqualified default: a
sub-second observed A100 saving does not justify nearly7GB of optional storage,
the failed5090 endpoint and this additional owner/lifetime complexity. This
is not a universal minimum-percent policy. The complete prototype overlays,
source/image/build receipts, probes, clean outputs, failures and separate
diagnostics remain in the compressed replayable evidence. Production code,
native test registration and promotion inventory return to the incumbent.

## Revisit when

Require a materially different work or lifetime argument before retrying full
AO storage. The current diagnostic's21.37s SCF VV10 pair work is the larger
separate opportunity. Its existing compaction, bilateral symmetry, strict
reciprocal lowering and rejected packet-width/FMA/force-output experiments
must be respected: a new experiment should remove real pair/sweep work, not
repeat spelling/width tuning or infer savings from nested scope subtraction.

## Retained development failures

LSF226979 fails before GPU execution because a login-node /tmp script is not
compute-visible. LSF226993 reaches Python but shared-home disk quota prevents
the initial output write, before the endpoint. LSF227018 uses owned compute
/tmp and a finite login-node transfer; authenticated download succeeds.
Host map/publication doubles require the new optional geometry lease argument;
their failed compile populations are setup failures, not scientific passes.

## References

- Tracking issue2976, plan6703119171, diagnostic6703367842, refusal6703456868;
  PR3033 and its CI-only9ec79b779 fix.
- `benchmarks/results/cold-xc-ao-cache-20261107/` preserves both rejected source
  overlays, including their executable host and independent native tests.
- `docs/developer/xc_native_cuda.md` for the shared XC owner contract.
