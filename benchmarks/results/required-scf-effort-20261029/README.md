# Required native SCF effort protocol — CPU qualification

Production frontend and native build: `fafe44f539699b188b7b13f68affe50708539d76`.
Baseline frontend: `131144549f91223cf751a27874c365834cc7c443`.
Both use the same current CPU library, SHA-256
`fce2a400dd1328ff6cdb589e3cb363bd39fd4705fd7b5488f2412e13d9410bfd`.
All native/build/compiler inputs are byte-identical between these revisions.
This changes missing-protocol behavior, not numerical equations or a CUDA domain.

The `samples` object in `evidence.json.xz` retains all 144 complete trajectories: H2, LiH and water in
STO-3G, plus water in def2-SVP, each at original and genuinely moved coordinates.
Each case has separate A/B first use followed by ABBA/BAAB/ABBA/BAAB, eight
measured samples per arm. Every observation constructs the policy and prepared
owners, executes up to six geometry steps with SCF/forces, performs strict final
cleanup and verification, and closes owners. Imports are outside endpoint timing.
The frozen measured calibration is common to both arms and never comes from a
production reference oracle. At most three cases run concurrently on distinct
CPUs 24–31; paired arms share one fixed CPU, with OMP/BLAS threads set to one.
No compilation ran during this campaign; no exclusive-host claim is made.

All eight complete-endpoint median ratios pass the unchanged 1.05 gate; maximum
ratio is **1.0073811551781504**. Every non-timing publication is exactly equal,
including energy, full force arrays, coordinates, decisions/transitions, SCF
iterations, convergence diagnostics and final disposition. All strict SCF
endpoints converge. Six optimizer steps do not establish a converged molecular
geometry: final force termination may correctly remain `observed_unmet`.
This runner publishes SCF iterations, not Fock-build counts or physical memory.

The cached current-source CPU Release build/install passes 127 CTests and 188
focused Python checks. The installed public-header SDK runs with
`env -i PATH=/nonexistent`; its executable identity and exit status are retained.
Compiler commands use verified sccache 0.16.0; before/after shared-cache counters
are provenance, not an attributed compilation-speed measurement.

Its `qualification` object retains both calibration receipts, the complete pilot,
exact worker/controller source, source/binary/sample identities, SDK and test
receipts, cache counters and hashes of ignored bulk logs. Decode the capsule
with Python `json.loads(lzma.decompress(path.read_bytes()))`. To replay, restore
the exact `qualification.replay_source` texts, adapt checkout/library paths, and write the
retained `qualification.calibration` object as `calibration.json` beside the qualification
directory before running its `qualify.py`. Keep all cases, six-step bound,
sample groups and numerical/work/1.05 gates unchanged.
