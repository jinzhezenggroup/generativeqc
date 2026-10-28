# Decision: automatic CUDA forces use the shared native prepared owner

Status: implemented
Date: 2026-10-27

## Problem

Automatic Libxc CUDA energy providers previously had no admitted stationary
force consumer. Adding a Python force driver or functional-specific equations
would bypass the single native runtime required by #2227. A full catalog also
must not multiply every automatic profile into legacy Python force libraries.

## Decision

Derive `libxc_<code>_<rks|uks>` profiles from the structural semilocal catalog.
Adapt the existing canonical automatic point worker to the existing stationary
geometry interface, including the unpolarized total-density LDA convention.
The common native controller owns all seven stationary sources, current D/W,
moving grid, partition response, AO derivatives, direct Coulomb response and
publication. There are no new named-functional force formulas or admission
whitelists. Explicit defect and unsupported-domain exclusions remain in force.

Admission intersects the exact linked energy/force model, domain and spin with
strict FP64, device-fused, all-electron s/p, direct J and three partition
iterations. Resource discovery and preparation use the same native predicate.
An absent artifact or unsupported context rejects explicitly. Automatic default
outputs remain energy-only; the caller requests forces explicitly.

The automatic AOT option remains opt-in. `auto` force selection follows the
selected energy package; explicit force selection can narrow it. Automatic
profiles do not create legacy Python s/p or s/p/d DSOs. Integral primitives and
source reductions remain shared. Installed execution requires no Python,
compiler, Maple, Libxc or reference engine.

## Canonical closure diagnosis

The first broad cold-force matrix passed, but LTA RKS warm/moved calls sometimes
failed. After the requested convergence gates passed, the old closure applied
another tiny DIIS proposal and then unextrapolated fixed-point corrections.
Tracing showed roundoff amplification by roughly 12–45 per correction, followed
by rejected warm seeds and cold retries, sometimes exhausting 300 iterations.

The accepted policy audits the current physical F[D] with the existing native
canonical proof before publication. It retains the original requested density,
energy and residual tolerances. A failed proof continues the intact DIIS solve
inside its original iteration budget. An accepted proof does not perturb D.
The attempted alternative of simply holding D before the old closure failed
cold convergence and was rejected. Loosening tolerances and adding an LTA branch
were also rejected. Curated closure behavior remains independently qualified.

## Artifact and resource evidence

The private diagnostic requires a current successful native state and reads the
actual point and force descriptors. Successful force work is bound to the
current token; later energy calls and failed items cannot borrow old force
success. Loaded-module identity comes from the descriptor address and `dladdr`,
with Linux mapping inode validation and a stat-bracketed binary hash. The mapped
inode must stay immutable while loaded. This explicit diagnostic is outside
timed production calls.

The offline audit joins that identity to generated sources, objects and cubin
resources. Cached geometry reports `none`; the bounded schedule reports
`minimum-workspace`. Retained-controller work counters are not advertised as
whole-endpoint all-attempt work accounting. Cubin stack/local counts do not
prove zero compiler spills. Auxiliary zero-register device-function records
are retained, while an all-empty resource scope fails.

Rebasing changed the upstream RHF integral-AD compiler input. The stationary
contract conservatively hashes the whole method compiler family, so force
identities correctly changed even though generated force bodies changed only
their identity literal. The first reference-reuse run rejected that mismatch.
An explicit source-transition audit then checked all 22 executed profiles and
the compiler input difference. Fresh forces and center oracles reuse the
unchanged scientific reference task, with old/new identities kept distinct.
This is independent-reference reuse, not an old artifact accepted under a new
cache key. The full automatic AOT package is rebuilt on the new baseline.

## Evidence and limits

The retained campaign is in
[`benchmarks/results/automatic-native-cuda-20261027`](../../../../benchmarks/results/automatic-native-cuda-20261027/README.md).
It covers 11 non-curated LDA/GGA/tau-MGGA registrations in both spin layouts,
full-vector reconverged native and independent PySCF/Libxc finite differences,
replay, moved coordinates, MethodIR, s/p water, low-density and near-iso-orbital
points, exact/below-minimum budgets, ragged failure/recovery, nonconvergence,
installed CLI and GPU memcheck. All GPU execution uses finite Slurm allocations.

The p-AO fixture is neutral water in both spin layouts. Initial correlation-only
water-ion Hcore trials for LDA and TPSS did not converge; they showed large SCF
oscillations, not near-tolerance closure failure. These failures are retained.
Open-shell H4 supplies the polarized gate; neutral-water UKS does not replace it
or claim that those water-ion solves now converge.

Full AOT compilation covers 197 energy registrations and 394 automatic native
force profiles. It is a compile/size/resource census, not 197 molecular numerical
qualifications. Representative execution receipts and the full-catalog resource
census are separate. The earlier narrow NVCC 12.9/sm80 persistent-cache closure
and real CPU CHACHIYO alias-reuse evidence retain their original limits; static
sm120 packaging does not imply a universal persistent CUDA cache closure.

Paired complete E+F endpoint measurements use the earlier automatic-force
implementation before the closure repair as baseline, not an admitted automatic
force endpoint on master. Curated PBE is the unchanged control. Keep every cold,
warm and moved sample, SCF work and failure/fallback observation.

## Revisit when

Additional angular momentum, empty-spin birth derivatives, mixed precision,
nonlocal/exact exchange, response/HVP or other architectures are proposed.
Each requires its own semantic and endpoint evidence. Do not turn source
emission, an AOT object or this representative matrix into a blanket capability.

References: #1118, #1122, #1123, #1124, #2227;
[current native CUDA contract](../../../../docs/developer/xc_native_cuda.md).
