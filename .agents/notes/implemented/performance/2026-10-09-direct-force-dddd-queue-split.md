# Decision: separate materialized dddd from the generic Direct force queue

Status: implemented; final-source qualification passed for the measured endpoints
Date: 2026-10-09

## Problem

The exact AO/Becke follow-ups did not address the retained integral derivative
consumer. Fresh master `d5a3173cc89b399ed105b0750124473eb782e2e5` uses default-enabled
materialized pair derivatives when its optional cache is available. The mixed
consumer consequently needs 256 lanes for the materialized dddd consumer's six
fixed component packets. This also prevents the other classes from using the
qualified 128-lane generic force schedule. The historical notes describing these
pair switches as default-off do not describe this master's `enabled()` policy.

The explicit angular schedule additionally repeated exact shell screening in
all thirteen passes before testing angular ownership. Four passes are
unreachable for a proved s/p/d basis. Removing that repeated work is useful but
did not establish a production-default improvement: it only changes an already
selected experimental route.

## Decision

For the method-neutral full-range force source with Force screening, admit a
split only when the complete packed basis proves maximum shell angular momentum
two, the retained materialized pair derivative cache and recurrence mode are
available, and the caller requests the standard 256-by-1-by-1 launch shape.
Order eight is then exactly dddd. Launch the complementary queue at 128 lanes,
reset the existing cursor on the same stream, and launch dddd at its original
256-lane component width. Reuse the output, queue representation and optional
profile; allocate no new retained storage.

The pure dddd specialization does not instantiate the unreachable generic AD
consumer. This is a compile-time ownership promise backed by the host admission
and exhaustive physical-shell ownership test, not a numerical approximation.
Mixed f-containing angular passes still instantiate their complete fallback.
Combined and Separate retain the existing algebra and coefficient forwarding.

The method-neutral launch seam returns submission/reset errors through the
native source boundary. Launch-error inspection uses `cudaPeekAtLastError`,
preserving the legacy caller's last-error check. Missing caches or incompatible
recurrences retain the original generic consumer; f/unproved bounds and custom
launch shapes retain their prior dispatch and workspace contract.

For the separately selected angular route, ownership now precedes exact shell
screening. Its proved maximum bounds orders to 0..4*lmax; s/p/d need at most
1/5/9 passes. Sentinel 255 retains all thirteen. This does not promote the
angular route or change Fock ownership, scientific gates or physical orientation.

## Rejected alternatives

- Making every force consumer 128 lanes silently omits half of each fixed
  materialized component packet. Do not trade scientific work for occupancy.
- Enabling the full angular schedule is not a substitute for reducing its
  repeated traversal and generic frames. The first ownership/cap pilot did not
  justify changing the production selector.
- A split alone still instantiated the generic AD frame in the dddd worker.
  Frozen `candidate-v3` passed all numerical/work gates but improved the
  48-atom complete medians only 1.615%/1.629%, below the 2% gate. Removing that
  unreachable consumer is a separate frozen `candidate-v4`, not a relabeling.
- No register cap, recurrence rewrite, screening threshold, AO cutoff or
  Becke selection is changed. Saving capacity alone is not a performance proof.

## Invariants

- Every physical quartet has exactly one owner and still passes its original
  scientific gates, generated-class mask and canonical pair orientation.
- The ordinary unpartitioned owner reads no angular metadata. The special
  complementary owner is used only under the complete s/p/d proof.
- Keep the 256-lane dddd packet mapping, queue publication/retirement barriers,
  same-stream ordering and complete source/error boundary.
- Force source differentiation remains FP64 and method-neutral; public forces
  do not depend on CPU/PySCF oracle work.
- The recorded shell/AO/primitive-AO counts are shell-admitted domains. The
  latter are upper bounds before inner component screening and zero weights,
  not executed recurrence, atomic or hardware-traffic counts.

## Qualification protocol

The matched endpoint compares frozen master with frozen candidate-v4, both with
the ordinary bounded route, incumbent AO producer, Becke mode zero and intrusive
profiling disabled during timing. Two independently converged native owners
share one CUDA context. Each owner uses its same source selection whenever it
is active, including changed-geometry preparation. Public warm updates are
frozen; the seeds are not claimed byte-identical across arms. Each measured
call is synchronized, converged, one iteration/one Fock, and has no warm fallback.

Each five-repeat population retains 50 complete calls: two construction calls,
four geometry setups, twenty primes, twenty interleaved measured calls and four
separate intrusive work diagnostics. Every call is independently checked against
the retained, scientifically identical GPU4PySCF perf5 references at 1e-8 Eh and
1e-7 Eh/Bohr. These references keep their original provenance and are not newly
computed d5 references. CPU verification rechecks raw outputs, state, resource
bounds, AO work, actual per-class force admissions, and medians/MAD without
trusting the driver's accuracy or performance summaries. Acceptance requires
improvement greater than max(2%, twice the sum of relative MADs).
This is a descriptive repeatability gate, not a confidence interval or proof
of a speedup on every supported basis/device.

The initial two-process harness repeatedly lacked a retained derivative lease
and failed closed for the enlarged force domain. A single CUDA context succeeds;
this does not establish the precise failed allocation without an allocation
receipt. Early cleanup hid the original failure, so the revised harness retains
it. A later feasibility run let the other owner's ambient angular setting leak
into changed-geometry reconstruction; actual launch counts exposed it. Those
runs are not promotion evidence. Configuration only at initial construction,
or reconstructing only the Python gradient executor, is insufficient.

## Measured endpoints

The final-source primary population uses PBE0/RKS, def2-SVP water clusters,
48 radial by 16 polar by 32 azimuthal points per atom, unpruned equal-radius
Becke partitioning, FP64 forces and screening tolerance 1e-12. Energy/density
convergence tolerances remain 1e-12/1e-10. The workload is the complete native
`execute(properties=("energy", "forces"))` endpoint, including its host force
return, not a standalone integral kernel. Neither the AO nor Becke production
selection changes.

Primary five-repeat population, Slurm job 6779 on node1, visibility 0:

| Atoms | Phase | Master median (s) | Candidate-v4 median (s) | Improvement | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| 48 | warm | 6.240242 | 6.108659 | 2.109% | pass |
| 48 | moved-warm | 6.240880 | 6.076362 | 2.636% | pass |
| 96 | warm | 19.275325 | 17.896664 | 7.152% | pass |
| 96 | moved-warm | 19.222016 | 17.849743 | 7.139% | pass |

For 48 atoms the stationary-integral region medians are
1.941926/1.951489 s before and 1.861312/1.847666 s after the change. These
component timings are explanatory only; the acceptance gate uses the complete
endpoint. Retain all raw samples, including candidate warm's 6.259787 s sample.
For 96 atoms the integral region is 6.538708/6.549249 s before and
5.713016/5.688246 s after. Its warm relative MADs are 0.1526%/0.1277%,
and moved-warm 0.0766%/0.1526%; both noise floors remain 2%.
The primary campaign completes at `2026-10-09T18:40:53+08:00`.

A separate ten-repeat 48-atom confirmation, job 6781, visibility 1, independently
checks 90 complete calls. Do not pool it with job 6779 or relabel it as the same
population:

| Phase | Master median (s) | Candidate-v4 median (s) | Improvement | Gate |
| --- | ---: | ---: | ---: | --- |
| warm | 6.310378 | 6.133879 | 2.797% | pass |
| moved-warm | 6.321480 | 6.126496 | 3.084% | pass |

Both confirmation noise floors are 2%; relative MADs are
0.1085%/0.3547% for baseline/candidate warm and 0.0784%/0.3740% for moved-warm.
Its candidate warm 6.364829 s and moved-warm 6.395495 s samples remain in the
statistics. The original five-repeat comparison likewise has a 2% noise floor;
its relative MADs are 0.1137%/0.3537% and 0.1924%/0.4450%.

The admission observer confirms one baseline versus two candidate force worker
launches, for both geometries. Per-class admitted scientific domains match
exactly; the following are totals, not estimated launch capacities:

| Atoms | Geometry | Shell quartets | AO quartets | Primitive-AO quartets | Tiles |
| --- | --- | ---: | ---: | ---: | ---: |
| 48 | original | 23,220,006 | 320,981,006 | 1,511,432,246 | 23,267,510 |
| 48 | moved | 23,219,801 | 320,981,265 | 1,511,429,483 | 23,267,305 |
| 96 | original | 92,233,228 | 1,263,186,780 | 5,944,643,268 | 92,420,204 |
| 96 | moved | 92,233,517 | 1,263,188,812 | 5,944,656,598 | 92,420,493 |

The 48-atom device/host additional bounds remain
257,507,136/111,230,496 bytes against 536,870,912/268,435,456-byte budgets.
AO active-domain/work counts and charged resource bounds match both arms.
For 96 atoms the device bound is 526,877,568 bytes, within the same
536,870,912-byte budget; its host bound equals the 268,435,456-byte budget.
There is no claimed reduction in these reservations or additional host headroom.
The split adds an outer traversal and a cursor reset, not less admitted work.

Final-source diagnostic function attributes on the same RTX 5090:

| Consumer | Threads | Registers/thread | Static shared (B) | Dynamic shared (B) | Local/thread (B) | Theoretical active CTAs/SM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Master mixed | 256 | 255 | 3,088 | 26,888 | 91,272 | 1 |
| Candidate generic complement | 128 | 255 | 3,084 | 0 | 90,936 | 2 |
| Candidate pure dddd | 256 | 236 | 3,088 | 26,888 | 704 | 1 |

The generic private frame does not disappear. The pure dddd frame does shrink
because its specialization no longer instantiates the unreachable generic
consumer. The active-CTA numbers are theoretical CUDA occupancy calculations,
not measured achieved occupancy or hardware traffic. This distinction also
explains why the split-only v3 did not clear the small-domain endpoint gate.

## Validation

- The 50-call 48-atom and 50-call 96-atom primary populations, plus the separate
  90-call confirmation, pass independent numerical, state, work and resource
  checks: 190 complete calls total, without pooling performance populations.
  Maximum absolute errors are 1.000444e-11 Eh and 3.786371e-11 Eh/Bohr,
  against 1e-8/1e-7 gates. Every prime/measured/diagnostic call has one iteration,
  one Fock and no warm fallback. The reference remains independent and retained,
  not relabeled as newly generated from the benchmark master.
- Final-source host cohort: 1,998 passed, 482 skipped. Do not add earlier
  overlapping host cohorts to this count.
- Final-source real-GPU composition/oracle cohort: 84 passed, including both
  spins, Cartesian/spherical bases, Combined/Separate sources, cache on/off,
  bounded/angular/resident routes and zero-K semilocal cases. The independent
  Libcint/CPU composition covers heavy d classes on two oxygen atoms.
- Queue-claim/high-angular cohort: 10 passed. This includes a source-only
  barrier check; it is not ten independent complete scientific GPU oracles.
- Memcheck, racecheck and initcheck: each runs two targeted Separate,
  full-range spherical PBE0/RKS cases (bounded and angular), with no errors,
  hazards or warnings. All complete on final v4; test marker is
  `2026-10-09T18:28:34+08:00`. Every real-GPU invocation uses finite `srun`.
- The queue-claim fixture previously copied a Force-dependent production
  snippet without defining Force. The repaired fixture declares Force and
  checks the actual 128-lane packet width. Its earlier v3 7-pass/3-error run
  stopped before sanitizers and is retained as failed harness evidence.
- Earlier v1 had four missing generic packaged-AOT failures (80 cases passed).
  A v1 heavy three-case racecheck exceeded its 30-minute allocation; exit 143
  is not evidence of a device hazard. Neither attempt qualifies final v4.
- Compiler dependency gates: 504 modules, zero errors; SCF: 223 modules,
  zero errors; CUDA ownership: 338 files. Complexity: 635 files and 177
  high-order loop nests. Source-bound materialization review against full
  master commit d5a3173cc89b399ed105b0750124473eb782e2e5 finds six baseline
  and six candidate cases, none added/removed/changed. All repository hooks
  pass, including Ruff and clang-format.

## Evidence retention

Frozen source/binary hashes, patches, ccache receipts, complete histories,
reference provenance, worker logs and admission/resource TSVs remain under
`.artifacts/angular-precheck/` in the author's checkout and
`/data/jzzeng/qc-angular-screen-20261009-d5a3173cc/evidence` on n1. Earlier failed
build/fixture/harness attempts retain their original names. All real-device
work uses finite Slurm allocations on RTX 5090/sm_120 and preserves visibility.

Frozen identity receipts (SHA-256):

- Master library:
  `0346d2dde90a1e0f1b3b9568e028387229124ee6073f9ad3b6d9334bbc4d7c23`.
- Candidate-v4 library:
  `9563667c7b085c9ed67ac7f175cf27aee9077399584c52e56277005143a801f2`.
- Five-native-file source manifest, matched against the final working tree:
  `14b3fe91e73bb7a2941eed2583b08ef82783493dfe4284a4759273358dc9dbcb`.
- `paired-endpoint-final.py`:
  `d2ddd8331908480e298ef901b0450b1400a4b5eded01acd99e8c16b8ca39e2a3`.
- `verify-paired.py`:
  `ad47968dbbb957220f8839cae1dcb18fea8b854930d01b9066e87549e8e740b6`.
- Observer source/binary:
  `5ba72ea994f88213c4908a1580c8e9f73f84da0a0e8d50874854afbb57b27909` /
  `9a6a3eda7550d1ac9303cad6d1340264112c0e1fe2446cbb2dddf1277a2e4247`.
- Retained 48/96-atom perf5 reference JSON:
  `dea65338702bca3cea6676c3b0e26b17f767e730fcdcbc4f9f7852a7337f555e` /
  `b0961774cc1a4e2a729d205a3952c6d2ec94e6593d5ead98226e3ac982fe8d9e`.
- Verified primary 48/96 summaries:
  `33f1a4c5b96e4d1a6412839f0971481b592a911119caf3fb89a84c121dd7e7d9` /
  `b1ed705b4815d85f1afc1fa557bef802bb0ae540b9969ec66e8ba5cb792b8b0b`.
- Verified separate ten-repeat confirmation:
  `7a0ee1c855c8030e6a92f7cb32643b379da4aa3e1ffa2043739867672e389045`.

Reproduce in the retained n1 workspace with `env.sh`,
`VARIANT=candidate-v4` and `paired-v4-campaign.sh` inside a finite Slurm
`main --gres=gpu:5090:1` allocation (50 minutes for the 48/96 campaign).
The script records the complete endpoint history and invokes the independent
CPU verifier. `candidate-tests.sh` holds the real-GPU/sanitizer commands;
`build.sh` reuses ccache with CUDA 12.9.1 and the sm_120 build target. Do not
overwrite a frozen population after a native change; create a new one.

CPU verification is also reproduced in the local checkout from raw JSON and
retained references, with byte-identical verified summaries. The verifier's
legacy `promotion: false` field is not the repeatability result; use its
`performance_gate` and explicit scientific/work/resource checks. Its reused
angular-policy wording does not authorize a broad experimental-policy
promotion, a cold/moved speedup, or a byte-identical-seed causal comparison.

The accepted change is the guarded internal full-range queue split; the
explicit angular selector and AO/Becke experimental policies are not promoted.
Endpoint evidence is restricted to these warm/moved-warm water-cluster
populations on RTX 5090. Master advanced to
`ab5282f74c0acf98c18ec05b343a5494de85cf49` during qualification via the unrelated
CC Lambda change #2156. This note does not relabel d5's frozen source/library
measurements as a fresh measurement of that later master.

## Revisit when

A class-indexed queue can eliminate the extra outer traversal without losing
the canonical domain, or a producer can remove the generic private frame for
additional classes. Require complete endpoints, admitted work equality and
independent force gates; do not infer achieved occupancy from function attributes.
