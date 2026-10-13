# r2SCAN-3c capability contract

GenerativeQC exposes canonical r2SCAN-3c through the Python selectors
"r2scan-3c", "r2scan-3c-rks", and "r2scan-3c-uks". These names resolve to the
same inspectable R2SCAN-3c MethodIR used by the compiler; they do not add a
second native scientific driver or a separate native method ID.

The canonical method binds four defining pieces: r2SCAN electronic structure,
the pinned spherical H-Ar def2-mTZVPP basis, the r2SCAN-3c D4(BJ)-EEQ-ATM
profile, and the matching gCP profile. Changing any defining component requires
a different method identity. The first supported element domain is Z=1..18.

| Path | Energy | Forces | Batch/replay | Notes |
| --- | --- | --- | --- | --- |
| CPU RKS/UKS, direct | yes | not promoted | yes | exact canonical basis; D4 and gCP are explicit components |
| CPU RKS/UKS, DF | yes | gated | yes | complete electronic DF response plus separately checked D4 and gCP gradients; bounded CPU force domain |
| CUDA RKS/UKS, direct | yes | gated | yes | s/p force uses packaged AOT; all-electron s/p/d force uses bounded JIT when NVCC is available |
| CUDA RKS/UKS, DF | yes | gated | yes | electronic DF response including auxiliary and metric; D4 and CPU gCP remain explicit, separately timed components |

method_capabilities() intentionally reports the conservative backend-neutral
surface (energy plus batch support). Force availability is resolved by
Calculator after the device, basis, and stationary-gradient domain are known.

For density fitting, the canonical r2SCAN-3c MethodIR retains the same pinned
def2-mTZVPP orbital basis, D4(BJ)-EEQ-ATM and gCP parameters. Its complete
electronic DF force uses the selected auxiliary basis and differentiates both
three-center integrals and the fitting metric, as well as the full moving grid.
The D4 and gCP geometry derivatives are evaluated exactly once by the separate
correction owner and added after the electronic response. This composite is
admitted only where the electronic backend has a prepared analytic force owner;
other D4/DF compositions are not promoted by this exception. The CPU consumer's
8-atom, 64-AO and 128-primitive limits still apply, and CUDA's native
per-item resource and generated derivative domain remain authoritative.

Real-device qualification for the canonical DF endpoint (including changed
geometry, translation and an independent reference) is in
`tests/python/test_r2scan3c_execution.py` and the source-matched
`benchmarks/run_readme_dft_benchmarks.sh` harness from #2401. CPU or ordinary
CI without a scheduler-allocated CUDA device does not establish CUDA
performance or force qualification.

Production total energy is assembled once as E_r2SCAN + E_D4 + E_gCP.
Where forces are admitted, the corresponding force is
F_r2SCAN - dE_D4/dR - dE_gCP/dR. Prepared replay keeps the correction owner
separate and supports changed coordinates without silently applying either
correction twice.

The method remains fail-closed outside H-Ar, for a changed defining basis, or
when a requested backend/property combination has not been qualified.
The first CUDA s/p/d stationary-force domain admits at most 32 atoms, 128 AOs,
1,000,000 grid points and 100,000,000 grid-pair visits. Stationary integral work
is streamed through bounded native pages: each page admits at most 4096 task
descriptors and, by default, 16,000,000 primitive records. The complete force may
execute more than 16,000,000 primitive records across multiple admitted pages;
the cumulative count is retained as coverage evidence rather than used as a
whole-force allocation gate. These are execution bounds, not a claim that every H-Ar
system has a qualified total force. A resource or compiler precondition that
fails produces an explicit item failure rather than an incomplete total.

Open-shell forces are conditional on the converged spin-density state. In
linear OH, a prepared changed-geometry run and a fresh run can converge to
different stationary states with nearly equal energy and different transverse
forces. The current method does not promise automatic root following across
that case; compare the final states before interpreting a cross-run force
difference as an error in one state's analytic force.
