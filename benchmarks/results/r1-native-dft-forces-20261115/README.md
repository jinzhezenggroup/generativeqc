# R1 native DFT force product qualification

This source-bound receipt accompanies the installed SDK/CLI/Python force
contract, not universal force admission or completion of the other runtime
ownership slices. The generated `docs/reference/libxc_qualification.md` remains
the authority for current automatic endpoint qualification.

Publication alone is not acceptance: the reconstructed audit must report PASS
for a passing cohort. Rejected and incomplete cohorts remain separate.
The independent controlled CUDA cohort on job 2043 passes all 408 comparisons,
with no failures or incomplete endpoints. Its maximum ratio is
**1.0224901148355194**, below the unchanged 1.05 gate. The earlier controlled
node1 cohort on job 17446 remains FAIL; its two warm ABBA failures are not
reclassified by the separate clean-node result.

## Identity and protocol

The live paired cohorts use parent `df42f93bbace0f16fb5b3dec77552384c9da9101`
and independently frozen baseline/candidate frontends. Their original headers,
complete observables, elapsed times and semantic work counts are retained in
`endpoint-observations/` with the existing exact-FP64 v3 reader. Every measured
JSONL block is reconstructible byte for byte and has its original SHA-256.
The source reconstruction envelope retains each historical source patch and
measured tool rather than relabelling an earlier run as current.
The code-only envelope lives at
`manifests/maintenance/r1-force-source-reconstruction.json.xz`; numerical
observations remain in this result family.

The parent subsequently merged as squash commit
`c99e6543e1fe0f4ea6832d58381d30e002e04c67`. Its complete Git tree is identical
to the original measured parent (`3cbd9c6234725daabd1c373c550bd07dac2b83ab`).
The envelope keeps each original `base_revision` and records an explicit
`reconstruction_base_revision` for recovery from this published master ancestor.
Apply the unchanged patch to that equivalent reconstruction base; the original
measurement identities, observations and source censuses are not relabelled.

The final CPU cohort contains 40 method/spin/Direct-DF cases and the CUDA cohort
contains 68, including eleven automatic representatives and two larger DF
clusters. Each has cold, warm and genuinely moved endpoints, eleven measured
samples after three warmups, in both ABBA and BAAB order. D3 resolution uses the
already supported MethodIR interface in **both** variants and stays inside cold
timing; this is not a claim that the broken parent string selector worked.
Timing is separate from qualification and profiling. The unchanged acceptance
requires every comparison, energy error <=2e-8, maximum force error <=3e-7,
identical semantic work at the same sample age/input history, and a per-case,
phase and order candidate/baseline median ratio <=1.05.

## Offline replay

From the repository root, choose output files outside `benchmarks/results`:

```bash
python -m benchmarks.native_dft_force_audit \
  benchmarks/results/r1-native-dft-forces-20261115/endpoint-observations \
  --backend cpu --output /tmp/r1-cpu-audit.json
python -m benchmarks.native_dft_force_audit \
  benchmarks/results/r1-native-dft-forces-20261115/endpoint-observations \
  --backend cuda --output /tmp/r1-cuda-audit.json
python -m benchmarks.native_dft_force_science_audit \
  benchmarks/results/r1-native-dft-forces-20261115/endpoint-observations \
  --output /tmp/r1-scientific-audit.json
```

The same auditor reads `--cohort cpu-incomplete`, `--cohort cuda-incomplete`,
`--cohort cuda-rejected`, `--cohort cuda-head-rejected` and
`--cohort cuda-bound-rejected` with their corresponding backend. These must
remain non-PASS; their observations are not removed from the retained streams.
No compiler, GPU, oracle package, network fetch or whole-checkout census is
needed to replay the measured numerical/work/timing gates.

The scientific replay authenticates all 34 automatic force receipts (58 rows,
eleven functionals) and eleven r²SCAN-3c receipts emitted by the installed CUDA
matrix. It checks the original independent-oracle, finite-difference,
translation, native-owner and resource gates. Full-vector cold tests and
directional cold/warm/moved tests retain their actual scopes; replay does not
promote them to a stronger full-vector three-phase endpoint-catalog claim.

## Rejected and incomplete attempts

- The original CUDA attempt timed out with incomplete BAAB coverage, constructor
  failures for the parent D3 string interface and actual ratios over 1.05. Its
  original audit and all measured observations remain retained. The old work
  auditor also compared different warm histories; the maintained auditor fixes
  that join without relaxing observable, timing or completeness checks.
- The first CPU attempt stopped before BAAB when source verification rejected
  seventeen changed inputs. Its 120 completed comparisons are not a full
  qualification, even though none exceeded the timing gate.
- The expanded old-native-identity CUDA cohort completed 408 comparisons but
  failed the BAAB `libxc-MGGA_X_LTA-0-0` moved endpoint: ratio
  **1.0654482192513264**, unchanged scientific work and no numerical failure.
  Preserve that real failed performance gate; do not call it noise or PASS.
  The live cohorts independently measure the subsequently rebuilt identity
  after the parent added missing generator dependencies.
- The rebuilt-head cohort also completed all 408 comparisons but failed two
  BAAB warm endpoints: `m06-2x-1-1` at **1.0858688997266084** and
  `wb97m-v-1-1` at **1.0891689425778566**, with identical semantic work and no
  numerical failure. It remains FAIL. Instrumented diagnosis records the outer
  wrapper's thread CPU time, wall time, context switches and GC events; it is not another
  accepted timing population. Slurm requested eight CPUs but left the process
  eligible on all 128. The controlled cohort binds both variants to the same
  first Slurm-assigned logical CPU, records actual CUDA UUID/PCI identity and
  preserves scheduler GPU visibility. Binding is not assumed to remove every
  driver wait; the full unchanged performance gate must independently pass.
- The CPU-bound cohort completed 408 comparisons with no incomplete endpoints,
  but ABBA warm `pbe-1-1` is **1.0569072806471687** and `wb97m-v-1-1` is
  **1.0785614498702347**. Numerical errors and semantic work pass. This cohort
  remains FAIL; CPU binding did not resolve the performance failure. Separate
  job 17450 correlates actual C ABI FFI windows with driver-call/syscall waits;
  its focused instrumented observations are diagnosis, not acceptance.
- The clean-node loader setup on job 2041 fails before scientific observation;
  its original error is retained. Job 2042 independently probes the resource
  environment, not acceptance. Job 2043 then executes the complete unchanged
  protocol on an exclusive allocation, with a separately authenticated private
  glibc 2.39 runtime and the original installed ELFs. The passing result does not
  prove a unique cause of node1's stalls or claim equivalent latency on arbitrary
  unbound hosts. CPU/Python builds and actual assigned GPU identities remain
  explicit in `validation.json.xz`.
- A current CPU harness initially selected the build ELF for the installed SDK
  fixture. All 160 errors rejected that identity before scientific execution.
  The original failure remains recorded alongside the complete module rerun
  using the installed ELF; the fixture assertion was not weakened.

## Scope and storage

The campaign/resource receipt bind the build ELF and the same finite Slurm
allocation. Installed SDK/CLI/Python acceptance and paired timing bind installed
ELFs: installation rewrites RPATH, so equal source identity does not imply equal
ELF SHA-256. Profiling reports explicit CUDA work, transfers, graph launches and
simultaneously-live explicit allocation peaks; static images and opaque
driver/vendor storage are outside that peak, not asserted to be zero.

Unsupported physical/precision/provider contexts remain fail closed, including
the separate automatic empty-spin particle-birth and HVP boundaries. Structural
energy selector availability is not force qualification. The pre-existing
CUDA AUTO-provenance CTest and all-target COSX linker failure are retained as
separate baseline limitations, not disguised as R1 acceptance.

No retention budget increases. All independent inputs, full-precision accepted
and rejected qualification observations and scientific gates remain offline.
`diagnosis.json.xz` separately retains every scalar timing row from all sixteen
instrumented streams: 7,512 records, all three phases and every pause. Historical
`native_execute_*` labels describe the outer Python wrapper; only the later FFI
probes isolate the actual ctypes C ABI boundary, still including ctypes/GIL
costs. These observations are not acceptance votes or pure C++ body timings.
Their redundant E/F vectors and raw profiler/syscall logs are not published;
the original diagnostic/JSONL hashes and all qualification E/F remain retained.

Two bulky GFN2 whole-checkout hash streams and four build/source-only provenance
files move to an already published Git ancestor. Their exact original identities
and explicit offline restoration instructions remain in authenticated snapshots.
Present corrupt files and missing scientific packets never fall back to history.
This is provenance storage, not a new numerical codec, discarded qualification
population or scientific requalification.

The existing change-review exceptions are exact-SHA-bound to the rejected ABBA
and BAAB FP64 streams only. They retain all original failed/incomplete E/F observations
rather than selecting passing attempts; accepted streams remain charged to the
ordinary review budget. This does not waive the hard per-file or aggregate cap,
alter any numerical/work/performance gate, or grant general force admission.
