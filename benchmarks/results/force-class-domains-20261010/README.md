# Exact mixed-f force class-domain qualification

This source-matched cold ethane CCSD(T) E+F campaign qualifies execution-domain
partitioning, not a new integral equation, precision policy or universal speedup.
Every timing starts a new process from geometry/basis and retains all 24 forces.
Compilation/installation are excluded. The predecessor RHF-auto cache is enabled
by its default on both baseline and candidate.

## Frozen source

Base: `e55dcd2b2ec4acd52d9f45d3f9fa2ebf709d6d7d`, including master `742dff879`
and the qualified RHF-auto implementation. The candidate is this base plus the
retained reconstruction patch, not a clean new Git commit.

The final 2026-10-10 fetch observes newer master
`7f342546d887796e6a92a005ba033029ed73ce2a`. Its later canonical-pair/DFT changes
are not part of this frozen measurement. Requalify after integrating them;
do not present these numbers as measurements of that newer tree.

Candidate library: `9b42eccb04df382615355f9338f12c1c0fa7ff8d9423590db708386021b03e8c`.
Candidate endpoint: `9fb79949cc6503fceff1e3ada433989b82df46e3698ff320efc946f44551f77d`.
Baseline library: `6d771d67542d82e47d4442b19f6a4b9ad74afd971d2f4ff32b80485b03c20ded`.
Input: `9428f2b1d1db38ffa374387705099e8d57fde98e0e068faed2861b04604a1c6e`.

The input and retained independent/qualified ethane reference files are already
tracked in `benchmarks/results/rhf-phase-values-auto-20261010/`; this bundle
references their exact identities rather than duplicating those fixtures.

## Results and boundaries

| Clean native median | Baseline | Candidate |
| --- | ---: | ---: |
| Complete cold E+F | 349.802091 s | 326.742221 s |
| Two-electron nuclear derivative | 89.908148 s | 67.233617 s |
| Lambda | 87.764197 s | 87.740530 s |

Three n1 RTX 5090 UUID-stratified ABBA allocations, jobs 7143/7144/7145, retain
six samples per side. Complete E+F improves 6.5923% (1.07058x); derivatives improve
25.2197%. All strata and the descriptive MAD floor are positive. Sampled 1-Hz
NVML total-device peaks are 31,151 MiB on both sides, not exact allocator peaks.
RHF12+final Fock1, CC20/38, Lambda21/22/Q32, Z12/13 and two exact physical
derivative passes are unchanged. No replay/audit is omitted.

Separate node-level profile job7155 shows two residual f launches and ten
specialized launches, still only two physical passes. F residual kernel sum is
55.027298 s, cooperative 9.832603 s and dddd 1.982722 s. The three scalar domains
sum to 0.683180 s. Instrumented timings are not pooled with clean samples.
Linked f/scalar kernels still report 255 registers. NCU hardware counters are
unavailable under the current profiling-admin policy; no occupancy/spill claim
is made from static resource counts.

Fresh independent Cartesian/spherical mixed-basis two-electron FCI energy and
all-six-coordinate force tests pass on n4/job720. Native arbitrary-density,
source-mask, opposite/zero-density, both-spin and both-representation J'/K'
gates also pass on the default and old fallback routes. Full mixed-provider
memcheck on n4/job721 has zero errors and zero leaked bytes. The preceding
sanitizer attach-timeout attempt is retained, not counted as a passing run.
Ethane energy/FD and 24-force comparisons use retained references, not fresh
full ethane force oracles.

## Reproduction

1. Check out the base, decompress/apply `measured-source.patch.gz`, and verify
   source/fixture hashes. Build baseline before applying the patch and candidate
   afterwards. Use Release, GCC11.4, CUDA12.9, sm120, portable_cuda AOT and
   `GENERATIVEQC_CUDA_FAST_COMPILE=OFF`; verify/reuse ccache. The recipe's existing
   compiler cache makes its build times incremental, not uncached compile costs.
2. Stage each library/endpoint into the `baseline`/`candidate` folders expected by
   `run-abba.sh`, including the library's `.so.0` alias. Copy the predecessor's
   ethane input to the recipe root. New builds need new identities and new
   observations, never relabelled measurements.
3. Run `srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1
   --time=00:35:00 bash run-abba.sh` on three distinct n1 GPUs. Preserve assigned
   `CUDA_VISIBLE_DEVICES`. n1/n4 provide real-GPU evidence here; n2 is CPU-build
   only because of its driver mismatch. Do not use maintenance node n3.
4. Run `run-qualification.sh` via a finite GPU allocation. Its context-first
   native probe and finite attachment timeout protect sanitizer startup during
   CPU oracle preparation. Test-only `P1_PROFILE_ONLY`, `P1_SKIP_MEMCHECK` and
   `P1_MEMCHECK_ONLY` split independent qualification stages when needed.
5. `validation.json.gz` retains all endpoint outputs, numerical gates, semantic
   work, hardware/build identities and the separate profile. The shared
   publication reader validates/decompresses it. Raw traces/logs remain under
   the local `evidence-force-class-domains-20261010/` audit directory.

The patch reconstructs production inputs, the native/independent mixed-basis
tests, ownership metadata and rationale. The bundle's host publication test is
a separate evidence guard, not an input to the measured CUDA library.
