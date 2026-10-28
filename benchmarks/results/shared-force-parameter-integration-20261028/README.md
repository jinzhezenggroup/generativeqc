# Shared force and parameter integration qualification

The `pre-r1/` records qualify source commit
`84c30caafaa2e8393cbb5cb7a6406300f63a16f2`, before integration of the subsequently
merged R1 force owner. They are historical integration evidence, not a claim
about later source revisions. There is no performance promotion.

The build used GCC, CUDA 12.9.86, SM120 and verified sccache 0.16.0 on node2.
Every real-device command ran in a finite Slurm allocation with its assigned
device visibility. All 3,184 tracked scientific/build input blobs matched the
recorded Git revision. `pre-r1-manifest.json` records that check and original
and stored identities for every compressed receipt. XZ decoding restores the
exact original bytes; no timing sample or failure outcome is relabelled.

The automatic force matrix covers KTBM 0/1, VWN 1/4, APBE and LTA in both spin
layouts, with cold, repeated and moved-geometry endpoints. KTBM uses full-vector
finite differences; the other registrations use the established directional
gate. The existing test independently reconverges PySCF/Libxc and checks energy,
density, full Fock, forces, artifact ownership and work diagnostics. All twelve
cases passed. Production calls forbid Python scientific fallbacks.

Four native CLI compositions cover PBE-D3 and WB97M-V in both spin layouts.
Their complete forces agree with CPU and GPU energy finite differences at two
step sizes; WB97M-V also retains its independent PySCF/Libxc center-energy gate.
The CPU derivative AOT option was disabled in this build, so the CLI comparison
uses CPU **energy finite differences**, not an unavailable CPU force endpoint.
The original attempt with grid options on a base PBE route was rejected before
execution and corrected to the repository's existing named-composition fixture.

The four accompanying drivers reproduce the source-matched build and finite
Slurm test commands after adapting only checkout/environment paths. They do not
authorize rerunning GPU work outside Slurm or admitting these old receipts to a
different compiler subject. Complete build logs and binaries remain ignored
local artifacts; the manifest identifies the measured library through the raw
endpoint receipts.
