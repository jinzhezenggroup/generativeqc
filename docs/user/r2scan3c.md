# r2SCAN-3c capability contract

GenerativeQC exposes canonical r2SCAN-3c through the Python selectors
"r2scan-3c", "r2scan-3c-rks", and "r2scan-3c-uks". These names resolve to the
same inspectable R2SCAN-3c MethodIR used by the compiler. The C/C++ named
preparation API and CLI also expose complete CPU/CUDA energies and
context-qualified analytic forces, using a compiled
projection of that graph without adding a separate native method ID.

The canonical method binds four defining pieces: r2SCAN electronic structure,
the pinned spherical H-Ar def2-mTZVPP basis, the r2SCAN-3c D4(BJ)-EEQ-ATM
profile, and the matching gCP profile. Changing any defining component requires
a different method identity. The first supported element domain is Z=1..18.

| Path | Energy | Forces | Batch/replay | Notes |
| --- | --- | --- | --- | --- |
| CPU RKS/UKS, direct | yes | gated | yes | complete native owner; s/p/d Direct derivatives |
| CPU RKS/UKS, DF | yes | gated | yes | complete native electronic DF, D4 and gCP response |
| CUDA RKS/UKS, direct | yes | gated | yes | native s/p/d electronic force, D4 and device gCP |
| CUDA RKS/UKS, DF | yes | gated | yes | native resident DF response including auxiliary and metric; device D4 and gCP |

method_capabilities() intentionally reports the conservative backend-neutral
surface (energy plus batch support). Force availability is resolved by
Calculator after the device, basis, and stationary-gradient domain are known.

For density fitting, the canonical r2SCAN-3c MethodIR retains the same pinned
def2-mTZVPP orbital basis, D4(BJ)-EEQ-ATM and gCP parameters. Its complete
electronic DF force uses the selected auxiliary basis and differentiates both
three-center integrals and the fitting metric, as well as the full moving grid.
The D4 and gCP geometry derivatives are evaluated exactly once and added after
the electronic response. Both backends retain the corrections inside the native
prepared KS owner. Admission follows actual electronic and correction providers,
checked workspace sizes and the final stationary token. Historical electronic
atom/AO/primitive ceilings no longer determine force availability.

Real-device qualification for the canonical DF endpoint (including changed
geometry, translation and an independent reference) is in
`tests/python/test_r2scan3c_execution.py` and the source-matched
`benchmarks/run_readme_dft_benchmarks.sh` harness. CPU or ordinary
CI without a scheduler-allocated CUDA device does not establish CUDA
performance or force qualification.

For an exact canonical FP64 graph without a global ResourcePlan, native
execution materializes the compiled basis, checks actual normalized basis data,
runs the r²SCAN-3c D4 profile and gCP, and publishes component receipts. Python
presents those receipts without correction arithmetic. Energy-only execution
omits correction derivative work. Offline Python basis metadata remains
available independently of the native library.

CUDA native gCP uses the same audited pair algebra and parameter tables as CPU,
with device pair evaluation and reduction. Its retained workspace is linear in
atom count and pair work is quadratic. Coordinate-index representability and
the explicit dispersion budget bound gCP allocation. The separate D4 owner
retains its EEQ pivot-capacity bound and must fit the same correction budget. Native CUDA receipts identify gCP
as `r2scan3c-gcp-cuda-v1` and report the CUDA backend.

CPU and CUDA Direct forces use the packaged s/p/d derivatives. DF uses its
qualified orbital and auxiliary domains. CUDA native FP64 requires three
partition iterations. Exact prepared property queries are authoritative; see
the [shared derivative contract](native_cpp.md#density-fitting-and-derivative-boundary).

Custom graphs, older libraries and contexts outside native admission retain
established explicit compatibility routes. The native CLI refuses unsupported
force contexts before execution.

Production total energy is assembled once as E_r2SCAN + E_D4 + E_gCP.
Where forces are admitted, the corresponding force is
F_r2SCAN - dE_D4/dR - dE_gCP/dR. Prepared replay supports changed coordinates
without applying either correction twice. Native energy and force execution
retain both corrections inside the KS owner and revoke receipts after failed
replay. Independent PySCF, dftd4 and gCP references qualify the native endpoint;
`tests/python/test_native_r2scan3c.py` and
`tests/native/test_cli_cuda_named.py` exercise its ownership and numerical gates.

The method remains fail-closed outside H-Ar, for a changed defining basis, or
when a requested backend/property combination has not been qualified.
Integral work is streamed through bounded native pages; total work is counted
across pages rather than used as a molecular-size admission ceiling. Resource
or provider failures produce an explicit item failure, never an incomplete
force total.

Open-shell forces are conditional on the converged spin-density state. In
linear OH, a prepared changed-geometry run and a fresh run can converge to
different stationary states with nearly equal energy and different transverse
forces. The current method does not promise automatic root following across
that case; compare the final states before interpreting a cross-run force
difference as an error in one state's analytic force.
