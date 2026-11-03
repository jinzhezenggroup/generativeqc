# Decision: one physical unrestricted CBS response on CPU/CUDA and fitted sources

Status: implemented
Date: 2026-11-03

## Problem

The initial native unrestricted CBS promotion deliberately admitted CPU energy
only. Removing its historical 12-AO check is insufficient to qualify forces,
CUDA or density fitting: each needs a complete physical response and an explicit
resource/backend contract, not a prototype or silent reference-package fallback.
This note supersedes the energy-only scope in
`../architecture/2026-11-02-open-shell-cbs-native-companions.md` and extends the
resource decision in
`../compatibility/2026-11-03-unrestricted-cc-resource-admission.md`.

## Decision

- Generate primal, amplitude VJP, raw-Hamiltonian and composed physical
  Lagrangian programs from the existing unrestricted TensorIR/AD owners.
  CPU and CUDA execute the same mathematical programs with runtime spin extents.
- Keep the packed independent same-spin amplitude convention. Its residual
  dual lift has a factor of 1/4; the amplitude pullback sums all four signed
  entries. Picking one entry gives an incorrect Lambda equation.
- Differentiate standard (T) with full occupied/virtual same-space Fock bars,
  using paired resolvent moments on all six legs. Diagonal epsilon bars alone
  omit off-diagonal density contributions and fail analytic force oracles.
- Replay the retained CC parent and triples energy, solve corrected Lambda and
  the same live UHF-provider adjoint, and check fresh full orbital stationarity
  before contracting derivatives or publishing any result.
- Compose fitted AO interactions from raw three-center integrals and the shared
  fixed-branch Coulomb-metric inverse square root. The generated Gram map and
  VJP own both integral legs; the existing metric Frechet owner supplies retained/
  discarded cross terms and rejects unresolved rank crossings.
- Correlation-only DF has independent reference and correlation ERIs in the
  Hamiltonian DAG. Reference Fock/energy and its orbital response stay exact.
  Full DF uses fitted ERIs throughout. Never obtain the former by substituting
  fitted ERIs into an exact-reference response.
- Admit dense raw-MO/AO weights, fitted source, arenas, transport, derivative
  staging and Krylov capacity with checked arithmetic. Remove molecular AO-count
  ceilings, not memory/work refusal. The dense fitted source is intentionally
  not advertised as a low-memory DF-CC algorithm.

## Backend and lifetime contract

CUDA graphs, spin-aware integral transforms and nuclear integral generation use
the requested device. Native host packed DIIS/Krylov control, MO-to-AO weights,
small matrix projections and spectral metric conditioning are explicit hybrid
consumers. UMP2 retains its existing generated host scalar cotangent assembly.
Conventional CUDA UHF also retains native host SCF/eigen control with device J/K.
These are not CPU-backend retries, PySCF calls or fully resident execution.

Method composition supplies synchronous integral callbacks through the neutral
`integrals/correlation_backend.hpp` contract. The retained CUDA SCF adapters
remain bound in `methods/unrestricted_integrals.hpp`; post-HF and CC response
must not acquire new dependencies on those SCF implementation headers. Missing
callbacks reject CUDA execution rather than selecting CPU work. This preserves
the existing schedules, coordinate conventions and drains without raising the
architecture checker's frozen post-HF-to-SCF dependency ceiling.

Every CUDA graph action drains its stream on success and exception before host
input/output buffers can die. Detached publication follows finite/error checks.
An empty spin block must skip its zero-output CUDA launches, but scalar
reductions over empty inputs must still execute and publish their zero result.
Skipping all actions merely because one occupied spin extent is zero is wrong.
Typed CUDA correlation-auxiliary handles are admitted for unrestricted correlated
methods only with a compatible selected CUDA fitting mode; their role does not
silently admit them as conventional or CPU auxiliary handles.
Conventional derivatives are shell-streamed. CUDA fitted derivatives currently
use admitted coordinate-major raw integral tensors; fitted CUDA diagnostics must
therefore not set shell-streaming/no-global-derivative-tensor provenance bits.

One physical UHF frame serves HF/UMP2/UCCSD(T) companions. A failed force,
denominator, stationarity, allocation or convergence gate must not publish a
partial group or leave a successful diagnostic from the failed call.

## Evidence

- Independent PySCF energies are reconverged at displaced geometries; production
  native code does not import or differentiate the oracle.
- CPU qualification includes neutral unequal-spin doublets, charged empty-beta
  triplets, full and correlation-only DF, complete CBS forces and nonzero
  14-AO spherical d-shell LiH forces against independent energy differences.
  The final endpoint regression run passed 225 tests (15 optional checks skipped),
  including force-budget refusal/recovery and standalone/grouped DF-UMP2 parity.
- Restricted CCSD/(T) and conventional CBS regression passed 39 tests (31
  optional device/frontier checks skipped), protecting the shared emitter.
- Shared Hamiltonian, triples-Fock and RCC generator qualification passed 30
  tests (6 optional CUDA checks skipped). The independent AO Gram/VJP and
  separate-reference Hamiltonian tests both passed.
- The final CUDA qualification selection passed all 14 tests: 12 real-device
  endpoints and two CPU fitted-policy controls selected by `-k cuda`. It covers
  neutral unequal-spin doublets, charged empty-beta triplets, conventional and
  both fitted policies, complete CBS forces, three nonzero 14-AO force cases,
  standalone/grouped UMP2 parity, and convergence failure/recovery. Independent
  displaced-energy force gates are 2e-7 to 3e-7 Hartree/bohr; translational sums
  are bounded by 2e-9 Hartree/bohr.
- CUDA 12.9/sm_120 qualification ran on an RTX 5090 through Slurm on `n1`
  (`node1`, `main`, `gpu:5090:1`, one task, 30-minute limit). The reproducible
  selection is `GENERATIVEQC_OPEN_SHELL_CUDA_TEST=1 pytest -q
  tests/python/test_cbs_open_shell_native.py -k cuda --maxfail=1`, with the
  source-matched native library selected by `GENERATIVEQC_LIBRARY`. Local raw
  logs are retained under `/tmp/qc-cbs-open-shell-20261103/` (final device log:
  `cuda-pr-qualification.log`, final CPU log: `cpu-pr-regression.log`). No GPU
  execution occurs on the maintenance node3.

## Rejected alternatives

Do not promote internal Python prototypes into the public runtime, differentiate
shifted iterative denominators, use finite differences as production forces,
silently refit the UHF reference, or reintroduce a small AO cutoff to hide an
unaccounted dense allocation. A fully resident low-rank CC implementation is a
separate future performance change requiring complete endpoint qualification.

## Revisit when

A factored/streamed correlation Hamiltonian or fully resident response owner can
replace dense transport while preserving independent energy/force oracles,
fixed-branch metric response, resource refusal and transactional publication.
