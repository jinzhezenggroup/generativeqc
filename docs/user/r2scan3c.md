# r2SCAN-3c

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
| CPU RKS/UKS, direct | yes | gated | yes | complete native owner; bounded s/p/d Direct work domain |
| CPU RKS/UKS, DF | yes | gated | yes | complete native electronic DF, D4 and gCP response; bounded CPU force domain |
| CUDA RKS/UKS, direct | yes | gated | yes | native s/p and bounded s/p/d force; broader qualified Python compatibility contexts use bounded JIT |
| CUDA RKS/UKS, DF | yes | gated | yes | electronic DF response including auxiliary and metric; D4 and CPU gCP remain explicit, separately timed components |

method_capabilities() intentionally reports the conservative backend-neutral
surface (energy plus batch support). Force availability is resolved by
Calculator after the device, basis, and stationary-gradient domain are known.

For density fitting, the canonical r2SCAN-3c MethodIR retains the same pinned
def2-mTZVPP orbital basis, D4(BJ)-EEQ-ATM and gCP parameters. Its complete
electronic DF force uses the selected auxiliary basis and differentiates both
three-center integrals and the fitting metric, as well as the full moving grid.
The D4 and gCP geometry derivatives are evaluated exactly once and added after
the electronic response. CPU native execution retains them inside the prepared
KS owner; CUDA DF compatibility retains its separate correction owner. This composite is
admitted only where the electronic backend has a prepared analytic force owner;
other D4/DF compositions are not promoted by this exception. CPU electronic derivative and primitive-work limits still apply, and CUDA's
native per-item resource and generated derivative domain remain authoritative.

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
atom count; pair work is quadratic and bounded to 4096 atoms. Both that workspace
and the D4 owner must fit the explicit dispersion budget. This correction bound
does not enlarge the electronic force domain. Native CUDA receipts identify gCP
as `r2scan3c-gcp-cuda-v1` and report the CUDA backend.

CPU Direct forces retain the packaged derivative s/p/d and primitive-work gate;
CPU DF retains its complete electronic source and resource admission. Native
CUDA Direct forces require strict FP64 and three partition iterations. The s/p/d
source extension now relies on the generated stationary executor's checked
shape and device budget, rather than a separate 8-atom/64-AO/128-primitive
whitelist. Newly admitted larger contexts still require independent real-device
numerical and complete-endpoint performance qualification. Exact prepared
property queries are authoritative.

Custom graphs, older libraries, global ResourcePlan requests and electronic
force contexts outside native admission retain established explicit Python
compatibility routes. In particular, CUDA DF and broader JIT force contexts
still expose their separately timed CPU gCP component. The native CLI refuses
unsupported force contexts before execution.

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
The broader Python compatibility CUDA s/p/d stationary-force domain admits at most 32 atoms, 128 AOs,
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

See the [method table](../public_methods.md) for the current selector inventory.
