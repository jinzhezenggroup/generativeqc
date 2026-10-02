# Native xTB execution and compiler ownership

`Calculator(method="gfn2-xtb")` executes the native molecular GFN2 runtime in
`src/xtb/native/`. Its private descriptor types and CPU/CUDA implementation use
GenerativeQC identity. The imported xTBloom C entry-point declarations, version header
and `include/xtbloom/` interface are retired. The public GenerativeQC ABI is unchanged.

The private execution adapter accepts one molecule, fresh SCC, energy and
analytic forces. It rejects periodic cells, external charge/response/field
attachments and other property flags before staging any input bytes. The
unsupported ALPB model, native periodic topology/integrals/Ewald/multipole
owners, standalone CUDA request/plan API, diagnostic snapshots, DLPack/result-arena API and CPU batch
worker/checkpoint infrastructure are removed. The internal SCC graph and its
bounded fallback remain part of CUDA execution.

## Runtime lifetime

Each public GFN2 `Calculator` retains one native context until `clear_cache()`
or calculator collection. The context owns a single method-neutral workspace
slot; GFN2 prepared calculations share its bridge with strong references. The
bridge retains one committed CPU/CUDA prepared cache, replacing incompatible
topology or scientific-control identities and refreshing coordinates before
execution. Transactional replacement may temporarily own one candidate alongside
the committed cache. Every request still uses fresh SCC. Energy/force output selection,
charges, multiplicities and geometry changes are validated on every call;
retention never substitutes a previous result or converged density.

Python serializes the whole prepare/execute/result-read transaction and cache
cleanup per calculator. The native context mutex serializes preparation, and
the bridge mutex covers execution and any orbital snapshot. Distinct calculators
have independent contexts. Backend/device changes retire the old context before
creating its replacement; failed creation leaves an empty owner for retry.
Prepared calculations keep their own immutable molecule/controls and a strong
bridge reference, so replacing the context slot cannot invalidate them.
`clear_cache()` waits for in-flight singlepoint calls, frees resident storage,
and leaves the calculator usable. This bounds the number of retained entries,
not the memory required by the current molecule. Native SCC graph resource
fallbacks remain unchanged.

CUDA topology preparation builds parameter plans, stable storage and descriptors
without evaluating an artificial molecule on the CPU or executing a synthetic
energy/force calculation. Numerical binding images are zero-filled except for an
identity overlap. Setup retains the identity factorization and its asynchronous
provider-error check, then invalidates that factor's geometry generation. The
public numerical epoch starts at zero: the first admitted request must refresh
all real geometry-dependent values on CUDA, refactor its overlap and commit
epoch one before SCC and energy/force publication. Fresh SAD initialization,
transactional failure behavior and the bounded SCC Graph fallback are unchanged.

## Scientific ownership

The compiler emits the following production mathematics. Backend owners retain
ragged storage traversal, validation, accumulation and publication, and launch
the compiler-selected schedules where available.

| Science | Compiler owner | Production consumers |
| --- | --- | --- |
| CN and repulsion | `geometry/gfn2_pair.py` | CPU and CUDA geometry/classical terms |
| S/D/Q Cartesian primitives | `integral/gfn2_sdq.py`, `integral/gfn2_sdq_cpu.py` | CPU and CUDA integral values/coordinate response |
| Electronic pair Hamiltonian and S/D/Q adjoints | `method/gfn2_electronic_runtime.py` | CPU and CUDA electronic owners |
| ES2, ES3 and AES2 | `method/gfn2_es2_runtime.py`, `method/gfn2_es3_runtime.py`, `method/gfn2_aes2.py`, `method/gfn2_aes2_schedule.py` | CPU/CUDA electrostatics and bounded AES2 CUDA scheduling |
| H0 shell factors, CN/radial/Cartesian adjoints and AO adjoint updates | `method/gfn2_h0_force_runtime.py`, `method/gfn2_h0_force_schedule.py` | CPU H0 values/VJP and CUDA H0 values/forces with bounded shell-pair scheduling |
| Shell spin energy and potential | `method/gfn2_spin_runtime.py` | CPU and CUDA spin owners |

Paths in this table are relative to `python/generativeqc_compiler/`. H0 value and force
consumers share `generated_gfn2_h0_native.hpp`. Spin consumers share
`generated_gfn2_spin_native.hpp`; generation preserves the ordered FMA
accumulation and records the reverse-AD identity of the symmetric shell energy.
Restricted zero-output admission remains backend policy. The public method's
restricted/shared-orbital open-shell capability is unchanged; generated spin
science does not admit a new public unrestricted endpoint.

SCC iteration/mixing/convergence, occupations, generalized eigensolver provider
selection, workspace/cache lifetime, per-system errors and public method
admission remain native runtime responsibilities. Generation needs no installed
GenerativeQC runtime, GPU, or scientific oracle.

CPU S/D/Q generation evaluates a complete Cartesian shell block per primitive
pair, sharing the Gaussian prefactor and recurrence intermediates across its
up to 36 outputs. Native contraction order, screening and spherical transforms
remain unchanged. CUDA keeps one Cartesian pair per lane and hoists only DAG
nodes common to every component alternative before the component switch.
Both routes retain FP64 arithmetic and the checked primitive entry points.

CUDA electronic Hamiltonian assembly and density contraction use the policy in
`method/gfn2_electronic_schedule.py`: 256 threads per block and up to 128 tiles
per system. Host-visible mean matrix size selects the tile count; each system
strides over its actual device extent. Small matrices retain one tile. This
requires no additional storage or device-to-host metadata transfer, including
for imbalanced ragged batches. One triangular pair owns both matrix
directions, preserving scalar arithmetic and spin packing. Density contraction
retains the full orbital sum in each lane; orbital and trace reduction orders
and finite-range checks are unchanged. Native validation
and whole-system publication remain separate launches; an error in any tile
suppresses the entire system's output. Final Hamiltonian/density matrix copies
reuse the same bounded tile count after all compute errors have settled.
Only tile zero publishes scalar and channel diagnostics.

The same compiler schedule selects one occupation solve when a restricted system
has exactly equal alpha/beta populations. Both spins have already passed native
admission and consume the same spectrum and temperature; the second output is a
copy of the first solve's occupations, chemical potential, electron sum and
entropy. Unequal populations and unrestricted spectra retain two independent
solves. Root finding, finite-range/degenerate fallbacks and reduction order remain
native policy and are unchanged. This removes repeated work within one SCC
iteration and does not reuse occupation results from an earlier iteration.

AES2 CUDA potentials and coordinate/CN derivatives use compiler-selected atom
tiles when the rounded-up mean atom count exceeds 32. At most 256 blocks per
system evaluate independent 32-peer chunks in bounded shared storage. Potential
components accumulate independently in their original peer order; the derivative
owner also preserves the first failing peer. Inputs and pair caches are validated
once per system before evaluation, and a separate publication launch suppresses
the complete output of a failed system. Smaller means retain fused validation and
evaluation. Neither path adds allocations, host synchronization or SCC reuse.
The native schedule harness compares these paths by adding independent single-atom
systems to select the fused policy, including ragged tails, Graph replay and
failure publication. Potential outputs must be bitwise equal; VJP outputs have a
tight FP64 roundoff gate because materialization changes NVCC's FMA boundary.

H0/Pulay CUDA force contraction distributes ordered shell pairs across at most
256 blocks of 128 threads per system. The compiler chooses the width from the
rounded-up mean pair count; native traversal strides over each actual ragged
extent. Small means retain a single block. Each pair owns a disjoint AO block
and preserves its AO reduction order. Atom gradients and coordination adjoints
retain FP64 atomic accumulation with unspecified inter-pair order. Input scans,
seed initialization and whole-system publication each remain outside the tiled
contraction, with no additional storage or synchronization. Qualification checks
bitwise AO equality against the single-block route, atom adjoints against an
independent long-double analytic oracle, and complete molecular forces against
tblite references. Failed or gated systems preserve all public accumulators.

Integral-force admission uses the compiler policy in
`integral/gfn2_force_schedule.py`: one block per system with 64 threads up to a
rounded-up mean of 4096 matrix elements, or 256 threads above that boundary.
Every actual ragged extent is still scanned once; atom/shell/primitive validation
and gradient-seed initialization remain in that block. The schedule adds no
storage, launches or synchronization. Scientific shell-pair force evaluation and
its fixed 64-lane reduction remain unchanged. Native qualification checks late
adjoint and metadata faults, gated peers and Graph replay at both widths.

`benchmarks/compare_xtbloom.py` compares public molecular energy/force calls with
matched fresh-SCC settings. It records cold, repeated and changed-geometry
timings, every SCC iteration count, numerical outputs and loaded binary hashes.
Comparisons retain the separate construction and first-call measurements and
also report `cold_total`, their sum, because the public APIs assign setup to
different phases. A cold call is the first call of a new calculator in the
measurement process; it is not a new process for each molecule.
Calculator/result cleanup after all samples is timed separately, before the
next case's constructor. Reports record this timing contract and reject a mix
with older receipts that included preceding-calculator cleanup in construction.
Run CUDA measurements inside Slurm as described below; compare the resulting
JSON files using `--reference`, `--candidate` and `--output`. Both reports must
use the same geometries and settings, and every sample participates in the
energy/force gate regardless of its iteration count. xTBloom's high-level API
also returns atomic charges; that additional output is retained in the
comparator's endpoint contract.

## Remaining native scientific work

Compiler ownership is not complete. Integral contraction/representation
transforms and multipole translation, parameter/basis binding, the D4 SCC
charge-response hot loop, and optional interaction primitives shared with the
remaining lower-level descriptor machinery still contain native scientific
arithmetic. These
need their own generated replacements and independent gates. Molecular final
D4 derivatives already reuse the shared GenerativeQC D4 provider.

The CUDA ownership ledger continues to count remaining handwritten science;
renaming or relocating a source is not a scientific retirement gate. Public
ragged-batch, CUDA wheel and complete endpoint performance acceptance also
remain separate from compiler replacement.

## Provenance and qualification

Upstream parameter snapshots, licenses, oracle attribution and revision IDs
retain their historical xTBloom names. In particular, the GFN1 header generator
checks its native output against the original audited digest after reversing
only the namespace substitution. Scientific table bytes and method parameter
identities are unchanged. `src/xtb/native/CUDA_SOURCE_PROVENANCE.json` retains
upstream hashes and records the current adapted source hashes.

Native CPU execution requires LP64 OpenBLAS with LAPACKE; a development build
can set `GENERATIVEQC_XTB_CPU_LINALG_LIBRARY` to the provider's absolute path. Wheel
builds retain their pinned private OpenBLAS provider and native shim. The former
`XTBLOOM_CPU_LINALG_LIBRARY` build setting has been retired.

Relevant gates are `test_gfn2_h0_force_codegen.py`,
`test_gfn2_spin_native_codegen.py`, `test_gfn2_runtime_bridge_boundary.py`,
`test_gfn2_xtb.py`, `test_gfn2_runtime_retention.py`, `test_gfn2_cuda_bootstrap.py`, and
`test_gfn2_xtb_force_qualification.py` under
`tests/python/`. GPU endpoint tests require `GENERATIVEQC_TEST_GFN2_CUDA=1` inside a
Slurm allocation on `main` with `--gres=gpu:5090:1` and a finite time limit.
The additional compiler graph CUDA gates use `GENERATIVEQC_GFN2_CUDA_TEST=1` in the
same allocation; these are separate from the native endpoint opt-in.
Run `python tools/check_compiler_structure.py` and
`python tools/report_cuda_ownership.py --check` for ownership validation.

The rationale and retained boundaries are recorded in the
[retirement note](../../.agents/notes/implemented/architecture/2026-09-22-xtb-native-compiler-retirement.md).
