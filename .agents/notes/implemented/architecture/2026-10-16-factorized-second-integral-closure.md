# Decision: share compiler-AD factors across the finite f/g Hessian closure

Status: implemented; finite full package and independent device gates qualified
Date: 2026-10-16

## Problem

The native stationary endpoint is Python-free, but independently expanding AD
for every high-angular quartet component is not a practical distribution. A
single ffff component previously emitted 377,862 bytes, while full ordered
orbital-through-f ERIs contain 480,000 Cartesian components across three radial
operators. Finite enumeration alone neither compiles nor qualifies that closure.
The original selected package and its numerical/performance receipts remain
separate from this extension.

## Decision

Keep mathematical ownership in the existing compiler scalar algebra. Reuse
`_pair_expansion` for one-dimensional Hermite coefficients and
`build_coulomb_derivative_algebra` for Cartesian Coulomb states. The existing
`_eri_differentiate` produces their gradients and gradient-JVPs. A product-jet
assembly is itself emitted by `Graph.differentiate`, not handwritten native AD.
Runtime loops convolve pair orders per axis and combine the three axes,
Coulomb state and Gaussian prefactor. This changes intra-primitive factoring,
not radial quadrature, normalization, center conventions or DF fitting policy.

ERI and physical raw three-/two-center DF reuse one compiler-emitted CUDA
factor library. DF attaches zero-exponent, zero-angular unit-Gaussian partners
only at explicitly compiled internal slots. The default geometry template
retains its positive-exponent domain; the two allowed masks require zero in
exactly the masked slots and strictly positive physical exponents elsewhere.
The compiler retains all real centers before molecular incidence scatter.

An explicit inventory schedule selects this backend. Each integral tile still
has exact scientific metadata, component selectors, identity and a context-free
native callback. Tiny host-only template forwarders copy bounded selectors into
the launch argument. Groups of at most 256 entries replace one NVCC translation
unit per tile. For the full finite selection this is 31,469 integral programs,
31,489 total programs, and 246 generated compilation units: 123 host metadata
groups, one shared CUDA factor library, and 122 existing one-electron/scalar/AO
units. The registry is additional host-only source. Emitting one 256-program
metadata group took about 0.12 s in a diagnostic, not an endpoint benchmark.

The original registry aggregate expanded 31,489 returned Program copies into
one optimizer CFG. Its host compiler ran for over four minutes before that
owned compilation was deliberately canceled. The replacement stores a finite
getter-address table and fills the final array with a loop directly in static
storage. It neither allocates nor creates a multi-MiB temporary on a caller's
stack. This preserves immutable sorted-span lookup and exact descriptors while
bounding compilation control-flow size. The interrupted attempt is not a
completed full-build receipt.

The native ABI, outer 64-thread primitive batches, ordered FP64 radial summation
and uniform late-failure barrier are unchanged. No tree summation, floating
atomic, runtime compiler, interpreter, reference computation or Direct-for-DF
substitution is authorized. CUDA compiler local/stack storage remains a real
resource cost; finite metadata or bounded arrays do not establish good scaling.

The sm120 compiler resource report for the shared kernel records 255 registers,
10,560 bytes of stack per thread and 13,824 bytes of shared storage per block.
This is not zero-cost scratch and is not an endpoint performance claim. Native
callbacks still allocate nothing; device stack/local backing is managed by CUDA.

## Rejected alternatives

- Do not ship tens of thousands of separately expanded NVCC units or claim a
  finite inventory file proves that a full package was built.
- Do not handwrite derivative recurrences or let native runtime code become a
  second scientific algebra. The established scalar AD owns product derivatives.
- Do not relax all ERI exponent validation to admit internal unit Gaussians.
- Do not use a runtime wildcard or JIT miss path to fill absent linked tiles.
- Do not silently replace the default expanded development selection. Both
  schedules are explicit build selections; neither bypasses molecular gates.
- Do not infer GPU4PySCF speed acceptance from the smaller compilation graph.
  The earlier matched comparison still fails the unchanged numerical gate.

## Evidence

The first CPU prototype passed six direct independent libcint controls,
including ffff full/LR/SR, f/f/g raw three-center and g/g metric. The formal
CPU suite subsequently passed all 16 cases in 47.44 s with verified sccache
0.16.0: independent primitive values/gradients and Richardson-extrapolated
gradient JVPs, translation and rejection of invalid real exponents. Python is
an optional test oracle, not the native runtime.

Host package tests require finite full component inventories, bounded source
grouping, stable mathematical identities and unchanged default generation.
The Slurm GPU suite checks one explicit Cartesian tile in every ordered
ERI-through-f full/LR/SR and DF-through-g class; it does not claim every
Cartesian component is independently sampled. The full sm120 package compiled
and linked all five native targets, with ccache visibly in the NVCC invocation.
The immutable factorized deployment (CLI SHA256
`5e708a6c5d351ecf0426fa2389ac663c33d9bc44c82924a9b31a5629df0d5043`,
inventory SHA256 `c349bc9f63d543f5a1a33bd9ca2552a568d6846c9d5f4d83b5222d15a04770c6`)
passed the following finite Slurm node2/pro6000 gates with PTX JIT disabled:

- Job 3203: 52 native AOT gates plus five top-angular independent cases.
- Job 3207: all 40 raw DF source/projection gates, including joint f/f/g.
- Job 3208: all 25 independent, reconverged values-only molecular oracle cases,
  including spin/backend/layout/ECP, separate spherical d/g auxiliaries and the
  complete H2 matrix. Disjoint worker counts were 7/6/6/6.
- Job 3215: all three actual C++ CLI gates, including UKS/DF/ECP with d/g
  auxiliary layouts, with Python/compiler discovery unavailable in native children.
- Job 3211: synccheck, racecheck, initcheck and memcheck all reported zero errors
  or hazards on the accepted UKS/DF/ECP+spherical-g endpoint.
- Job 3218: all 873 ordered ERI/DF classes passed independent libcint gates,
  with complete nonoverlapping linked selectors; workers passed 219/218/218/218.

The original all-class job 3206 failed 56 short-range JVP reference comparisons:
the libcint short-range gradient's small quadrature noise was amplified by
2e-4/1e-4 finite differences. The unchanged native primitive agreed with libcint
values/gradients to about 1e-14. Job 3214 retained stencil convergence and the
independent full-minus-long identity: for sdfd, direct short-range FD errors grew
from 1.52e-12 at step .02 to 8.29e-10 at .0002, while the radial-identity stencil
agreed to 1.86e-12 at .0002. The maintained gate now requires **both** .02/.01 and
.01/.005 Richardson stencils to pass the original atol=3e-10, rtol=2e-9; no
native arithmetic, oracle kernel or acceptance tolerance was changed.

Retained local logs/receipts use `.artifacts/wb97mv-hessian-aot-20261015/` and
immutable remote deployments under
`/data/jzzeng/wb97mv-hessian-native-response-20261015/`. These are development
qualification receipts, not an accepted GPU4PySCF speed comparison.

## Revisit when

Complete endpoint profiles show that shared factor evaluation, device calls,
local-memory traffic or metadata startup dominate. Any specialization or reuse
must retain independent numerical gates, deterministic radial contraction,
bounded work/storage, exact selectors and whole-result failure behavior.
