# Retained CUDA AOT default-policy input

`production-shell-classes.json.xz` losslessly retains the exact stock manifest from
`44f34575f0c91b846c7265e5c388f068de09826c` (the P1b policy-dimension repair).
Both its stored and original uncompressed identities are retained in
`manifests/maintenance/cuda_aot_default_decisions.json`.

`effective-defaults.json.xz` retains the corresponding semantic resolution for
the audited stock targets: schedules, recurrences, consumer/route masks,
preferences, packaging and target facts. The gate compares current resolution
with this immutable data rather than re-resolving the old manifest under the
new compiler, which could otherwise hide implicit policy changes. This is a
policy contract, not a generated-source or function-body fingerprint.

This is an immutable historical **input**, not a newly qualified result, new
benchmark campaign, or authority to promote additional defaults. It preserves
the existing tuned/portable decisions while the checker evaluates future
semantic changes against source-matched cold/warm/moved campaigns. Formatting
of the current manifest is not compared byte-for-byte with this input.

The preceding P1b migration compared generated CUDA, registry and derivative
outputs against `8d2ad48175b79f2a3cb5749d3552413cdf5b7d4e`: stock auto bundles
for SM80/SM90/SM120 and forced-portable SM120 were byte-identical. That was a
one-off migration observation, not a permanent generated-source fingerprint.
No GPU numerical or timing result is asserted by this retained manifest.

Existing scientific and performance records keep their original source and
device identity. In particular, retained provisional f-shell selections are
not certified merely because they are present in this manifest.

## Source-matched compile-only acquisition

`sm80-sm90-compile.json.xz` retains input identities, compiler logs, CUOBJDump resource
text, toolchain versions and acquisition index for 88 translation units built
from P1b source `6c0e89cfff07881d76f10f27de79b7bc509dd0d1`, native source identity
`788daab81655bece6d505d8aefe0efaf20bf56b69f3f70abf985292ca1cb0e1a`.
This includes global-AOT-disable and mainline LAPACKE repairs; previous
receipts are preserved at their own source identity, not relabeled.
All 44 stock SM120 primary selections were retargeted, audit-only, to SM80 and
SM90. Compilation succeeded for all 88 units. The generated primary full-range
Fock/force and declared mixed/streaming wrappers were included; optional
competitors and exact SR/LR derivative packages were not.

The acquisition used CUDA 12.9.86, GCC 11.4, C++20, the compiler adapter's release
optimization and sccache 0.16.0 on n2, through finite Slurm job 5316
(`main`, `gpu:pro6000:1`, one hour requested, visibility `0`). This node's actual
device is RTX PRO 6000 Blackwell/SM120: this was **not A100 or H100 execution**.
No candidate was enabled in production or executed on a GPU. Resources are
observations, not resource admission. The retained `scontrol` receipt records
successful completion and the finite limit, not accounting-daemon evidence.
An earlier-source missing-`cc1plus` GCC12 attempt remains retained locally;
it failed before CUDA compilation, not scientific execution.

Large objects and generated input sources remain in ignored acquisition storage;
their hashes, source/emission contract and raw resource output are retained here.
`python tools/check_cuda_aot_policy.py --verify-compile-inputs` optionally
regenerates inputs from matching source and checks their acquisition hashes.
This is not a permanent required-CI generated-source fingerprint. Shared sccache
statistics do not observe the adapters' owned per-invocation servers, so no cache
hit rate is claimed. The bundle protects integrity, not acquisition authenticity.

`tools/check_cuda_aot_policy.py` verifies retained log/resource hashes
and parses per-kernel resource observations before attaching only the covered
source-matched feature cells. Exact-package, optional-competitor, device-numerical
and endpoint cells remain not-run. After native source changes, this acquisition
becomes historical automatically rather than imposing a generated-source
fingerprint on future refactors. It never promotes a default.

## Source-matched SM120 endpoint diagnostics

`sm120-endpoint-smoke.json.xz` retains both Release build contracts, native
profile/ABI receipts, acquisition scripts, independent PySCF
2.14.0 Cartesian reference results and all raw stock RHF E+analytic-force
process records. Finite n1 build job 17486 rebuilds both binaries at the current
P1b acquisition identity above. Only the AOT profile differs. The native
profile test passes for both auto (`sm_120`)
and forced portable (`portable_cuda`), with a matched nonempty registry.
The common bounded build disables Hessian AOT leaves and CPU stationary-force
AOT; it is not a claim about a release distribution's full package/startup cost.

Finite Slurm job 17487 (`main`, `gpu:5090:1`, 30 minutes) runs four fresh
processes per arm for each basis, balanced in order, on the same physical
RTX 5090 UUID. Each process executes cold, warm, moved and moved-warm endpoints.
Cold includes import, profile probe, construction, preparation and force host
export; the other phases include synchronous execution/export. This is
application-fresh timing, not an OS/driver cache-cold experiment. The 16 process
records contain actual SCF/Fock counts and explicit invalid quartet counters;
unavailable counters must not be invented or relabeled as zero work.

| Stock case | Auto maximum force error | Portable maximum force error | Outcome |
| --- | --- | --- | --- |
| Water/def2-SVP, all four phases | 5.611e-8 | 5.611e-8 | All 32 phase observations pass |
| Water/def2-TZVP, cold and warm | 6.758e-7 | 6.758e-7 | All 16 observations fail the unchanged 1e-7 force gate |
| Water/def2-TZVP, moved and moved-warm | 4.484e-8 | 4.484e-8 | All 16 observations pass |

All energy and translation gates pass. The failed TZVP stock results remain
failed; they do not establish a portable-only defect. The acquisition jobs
correctly exit nonzero on those scientific failures, despite all child
processes completing. Their assertion failures remain in the raw receipts.
Earlier source `cacd7164584993ee789b4d97c789ed840a962110`, finite job 17470,
retains an explicitly nondefault diagnostic: both arms use
`energy_tolerance=1e-12`, `density_tolerance=1e-10`; initial iterations rise
from 9 to 16 and force errors fall to about 1.12e-10. This supports a
convergence-limited discrepancy, not a portable-only defect. Those historical
records never replace stock acceptance or enter a performance ranking.

The checker replays raw reference/run hashes, process ownership, source,
physical-device and library identities, complete phases and numerical outcomes.
It reports the failed arm/phase groups without failing mere evidence-integrity
validation or declaring a default qualified. No per-shell execution coverage,
A100/H100 numerics, cross-SM endpoint equivalence, full #3084 cohort rerun,
or global-default performance win is claimed. Small-water timings cannot settle
the production workload's portable-profile slowdown.

## Same-binary second-card diagnostic

`sm120-pro6000-endpoint-smoke.json.xz` retains finite job 5317 on n2
(`main`, `gpu:pro6000:1`, 30 minutes): another 16 fresh processes and 64 complete
phase observations. The RTX PRO 6000 Blackwell is a different physical card,
also SM120. Both ELF hashes exactly match n1; this is not a rebuild. All SVP
phases pass. TZVP cold/warm fails in both profiles at about 6.76e-7; moved phases
pass. This reproduces the discrepancy on both cards, not just the RTX 5090,
but supplies no SM80/SM90 execution or GPU-only performance attribution.

The earlier-source n2 loader attempt, job 5299, fails before any endpoint because its
system glibc 2.35 cannot load the n1-built ELF's glibc 2.38 requirement. Its raw
failure remains retained in published history. Job 5317 uses a
private copy of n1's glibc 2.39 loader
and runtime for Python and every child, with recorded commands and library
hashes. It changes neither system libraries nor Slurm device visibility.
Different host/Python environments still prohibit a card-only speed comparison.

## Actual global-disable build contract

Job 17487 also configures the actual CUDA CMake build for SM80/SM90/SM120 with
a custom portable manifest explicitly enabling derivative-shell packaging.
With global AOT enabled, generated packages match that opt-in. With
`GENERATIVEQC_ENABLE_AOT_SHELLS=OFF`, the actual generated derivative registry
is empty for all three targets, overriding the custom policy. The stock
configuration is restored before endpoint acquisition. Commands, source revision,
outputs and independently checked emitter results remain in the primary capsule.
Its archive descriptor recovers the exact custom input from the original capsule
at `ec205be986fe83824730fa6aa9fcd249b1fc304d`, checking both capsule/member hashes;
only identical input is shared, not old generated results or source/job identity.
This tests CMake/codegen routing, not
the mathematical or device qualification of optional derivative packages.

## Bounded retention

Identical n1 build receipts are stored once; n2 references that bundle and
checksums each transferred receipt. Heavy raw compile-command and CUOBJDump
files remain in ignored acquisition storage, identified by path, size and hash.
The complete preceding-source endpoint and compile bundles are recoverable at
Git commit `a6b15115a63c83df224045337826787f409e35cf`, using the same paths;
their original hashes are recorded in the current bundles. Its history chain
retains `f747d8079ada10d6af99536c5d0fbecbc4fa13b0` and
`3958635e7687537d50ae3f61689ba257416b872c`. Historical failures
and the old convergence diagnostic are not rewritten as current acceptance.
The frozen manifest is losslessly compressed with both stored and original
byte identities checked. No storage or scientific limit is increased.
