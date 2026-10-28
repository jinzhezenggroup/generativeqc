# Shared automatic CUDA force AOT

This campaign qualifies #1123 against the per-functional AOT implementation of
#2827. Compiler mathematics, the native prepared owner, seven-source force
contract and original numerical tolerances remain the owners. Production
execution never invokes Python science, Libxc, Maple or NVCC.

The measured incumbent is `e59942137e48a5c6767a9d330915cbd97761adc5`; the shared
implementation is `1447e0667d84cd9b3cbaead86a98a44f2719ac29`. The subsequent parent
head at measurement, `92abcbfb231b3ceb1d3f6a931497b6c4b9f447da`, changes only CI shard selection and
its test. It has no different compiler, native runtime or build input. Source
verification checks 5,915 candidate files against the measured commit. Full
source/command inventories, compiler logs and binaries remain ignored; compact
verification records and their hashes are retained.

During evidence publication the parent advanced to
`80616411b4bebd0aef9dbcabfd9da9f12e06884d`, adding a registry-enumeration optimization
and tests. This reuses a capability snapshot within each enumeration and changes
compiler inventory/provenance, without changing native runtime or formulas.
That revision is not relabeled as the measured baseline; this campaign makes no
generation-time or artifact-identity claim for it. The stacked change preserves
the parent's later fix when merged.

## Compilation and installed artifacts

Both builds use GCC 11, NVCC 12.9.86, `120-real`, strict FP64/`--fmad=false`, four
CUDA compile slots, two split-compile threads and twenty Ninja jobs on node2.
All compiler invocations use verified sccache 0.16.0. A retained, initially empty
private cache measures cold compilation; the same cache measures replay. The
shared cache is never cleared. Internal reuse within a cold build is reported
by the before/after cache counters.

The timed scope contains selected CUDA point objects, force consumers/bindings,
shared primitives, archive assembly and device linking. Generation and core
linking are recorded separately. Existing unrelated core objects and CPU point
objects are not made cold. CPU point source/object sizes are still inventoried.
Aggregate peak RSS samples all descendant compiler/link/cache processes every
100 ms, counting shared pages in each process. These are package-build
measurements, not a whole-project clean-build or exclusive-node timing claim.
An earlier candidate diagnostic build overlapped part of incumbent full cold
compilation on the 192-thread host; no compile speedup confidence interval is
claimed from these single runs.

| Scope / arm | Cold AOT seconds | Cache-hit seconds | Linked / installed bytes |
| --- | ---: | ---: | ---: |
| Narrow incumbent | 145.82264193694573 | 28.886744018062018 | 772414440 |
| Narrow shared | 65.77607458294369 | 12.816422794014215 | 750508568 |
| Full incumbent | 1588.1477737330133 | 369.4989903079113 | 1416722000 |
| Full shared | 152.4606938509969 | 14.825404471019283 | 908807712 |

The narrow package contains eleven automatic registrations, both force spin
layouts and two curated PBE controls. The full package contains 197 automatic
registrations and 394 scientific force bindings. At fixed execution ABI it has
six shared CUDA force consumers and six host binding objects, versus 394
per-functional force CUDA objects. Shared force source bytes are 758,933 and
object bytes 5,191,664, with host bindings reported separately. The corresponding
incumbent force totals are 60,153,143 and 397,978,096 bytes. These values must not
be confused with the stationary archive, device-link object or installed core.

Generic automatic force kernel counts are 180 in both shared packages, versus
660/11,820 in the narrow/full incumbents. Point kernels grow with packaged point
programs: 58/1,010 in the shared packages. The full package has 391 exact-math
workers for 394 spin registrations. Four VWN registrations share one U1 worker
and retain four distinct P2 workers and scientific identities. CHACHIYO and its
modified variant have different floors and do not alias. Independent point-body
and physical-contract comparison finds no mathematical changes in 394 programs.
The existing [Graph census](../automatic-native-cuda-20261027/catalog.json.gz)
remains the graph/derivative-root reference; provenance identities still change
when compiler source closure changes.

Resource distributions retain actual cubin registers/stack/local/shared values
and ptxas spill counts. Gates are REG <= 255, STACK/LOCAL <= 4096 and static
SHARED <= 49152 bytes; executed dynamic shared requirements are audited too.
Spills are not inferred from LOCAL=0. The full incumbent has eight spilling
point kernels and the shared package sixteen, including staged entry points;
the shared force kernels have much smaller spill maxima. This is not a claim
that all spills or historical higher-derivative pathologies have disappeared.
Narrow aggregate cold RSS also increases slightly; only measured scoped memory
values in `compilation.json.gz` are authoritative.

## Numerical, execution and resource gates

Both final packages re-execute 58 H4/water cold, warm and genuinely moved force
endpoints in the established eleven-registration RKS/UKS matrix. Every force
coordinate is compared with the incumbent and retained independent oracle at
3e-8 Ha/bohr; every center energy has a new independent PySCF/Libxc solve with
the original 3e-9 Ha gate. The 300 reconverged finite-difference directions in
the parent [references](../automatic-native-cuda-20261027/references.json.gz)
are reused explicitly, not described as newly measured stencils. Their original
3e-5 Ha/bohr force gate, 1e-4 bohr step, pinned BSE inputs and PySCF 2.14.0 /
Libxc 7.0.0 provenance remain intact.

Two APBE correlation full-vector cases rerun independent finite differences
after the partition-response repair. Eight VWN alias RKS/UKS cases add fresh
full-vector finite differences and actual executable receipts; the full
incumbent also re-executes these eight vectors. The probe binary and final
package are identified separately, even when their scientific source is equal.
The total of 120 newly evaluated directions includes repeated existing cases.

The 48-atom APBE exchange RKS/UKS checks compare phased and minimum-workspace
forces, two reconverged native energy-difference steps and incumbent full force
vectors. Actual workspace and launches prove the phased allocation was used.
The staged producer replaces the existing phased point launch; on the bounded
path its extra launch is counted once per tile. This checks a larger scheduling
regime, not a claim of general large-molecule performance or broader chemistry.

Native DFT and KS suites, 124 point/fixed-density tests, 91 automatic SCF tests,
four resource/failure tests and exact source/object/cubin receipt audits pass.
Minimum capacity, one byte below, ragged failure/recovery and nonconvergence keep
their fail-closed semantics. Point initcheck includes deliberately uninitialized
unused channels and upstream-error inputs. Force memcheck/initcheck cover LTA
RKS/UKS cold/warm/moved; all sanitizer summaries report zero errors.

The installed CLI checks 22 narrow and 28 full H4 force cases with no toolchain
on PATH and PTX JIT disabled. The loader's actual library choice is verified
against the installed path and immutable before/after hashes. The narrow
omitted-provider control rejects. Compiled catalog entries are not equivalent
to molecular qualification: no new Hessian, response, empty-spin UKS, higher
angular momentum, ECP, density fitting, mixed, nonlocal or exact-exchange support
is granted by this campaign.

## Complete endpoint timing and work

ABBA followed by BAAB uses eight fresh processes per package scope, with two
prepared owners per method and cold + three warm + moved calls per owner.
The matrix covers H4 PBE controls plus automatic VWN4/APBE exchange/LTA in
RKS/UKS, and s/p water VWN4/APBE exchange/TPSS correlation in both layouts.
The full package omits PBE force controls. All 1,120 narrow and 960 full endpoint
samples are retained. Every method/system/phase median must stay <= 1.05 times
its incumbent; samples are never dropped or timed around kernels alone.
All 42 narrow and 36 full gates pass; their maximum ratios are respectively
1.0128607611926415 and 1.0098907739421532. The independent installed-process
campaign retains 96 endpoints per scope and passes all 12 gates per scope;
maximum candidate/incumbent ratios are 0.9639581683043936 and 0.5591032829275604.

SDK cold includes calculator construction, preparation, complete SCF and host
return of E/F. A common disposable PBE energy call initializes CUDA first, so
this is a prepared-owner cold measurement, not process/context cold. Separate
installed-process ABBA/BAAB measurements include loading, context creation,
parsing, SCF+E+F and process teardown with no runtime tools. They retain the same
5% median limit and full-vector/iteration checks. CLI uses its default tile size;
SDK uses the matched explicit tile of 37, so their timing populations remain
separate. With the CLI tile, LTA/UKS takes 29 candidate iterations versus 28 in
the incumbent. The extra work is retained explicitly and still satisfies the
same cold endpoint time and numerical gates; no per-kernel speed is inferred.

All paired SDK iteration counts, Fock builds, physical J/K work and force/grid
work agree. Force workspace changes are exactly the bounded point-tile bytes;
force transfers agree and extra launches are explicitly accounted. One cold
LTA/UKS SCF control-flow difference is retained instead of calling all work
identical. At iteration 27 the incumbent passes its ordinary density gate but
rejects its canonical probe; the shared worker reaches the existing UKS
occupation shift before finishing at the same 28 Focks. This adds two matrix
products and one stabilized proposal, while avoiding one rejected probe, one
synchronization and 224 bytes (two 112-byte validation packets) of scalar D2H.
Final tolerances, physical state, force vectors and requested precision are
unchanged. Every sample's operator and transport counters remain available.

Receipt hashing is excluded from endpoint timers. Inside each finite immutable
DSO benchmark bracket, every query still authenticates callback/descriptor
mapping and inode/size/mtime/ctime; only the digest is reused between queries.
Full before/after DSO hashes are mandatory before a run is marked complete.
The production diagnostic retains its per-query policy. No CPU or GPU compiler
build runs overlap accepted endpoint timing campaigns.

## Failures, promotion and reproduction

The initially selected cooperative `precomputed_point` specialization skipped
inline Becke response for staged tiles, producing a 0.0212945 Ha/bohr H4/APBE
error with unchanged energy and density. Those binaries are rejected. The
repair preserves point setup and partition contraction/publication. A separate
rejected target-wide visibility setting split shared resource-ledger/TLS
owners; only generated worker/callback definitions are now hidden. Both
failures and the initial exact-transport assertion are retained in compact form.

`capabilities.json.gz` exposes retained compilation, artifact and endpoint
evidence through Lane F's existing `compiled-cuda` envelope. It has no force
promotion authority. The optional offline object store still accepts only its
documented NVCC 12.9 Linux sm80 whole-program closure; this sm120/device-link
package uses CMake and the verified compiler cache. Seven opt-in immutable
toolchain closure tests were not executed here and are not represented as
passed.

Replay the retained measurements from the repository root after compiler changes:

```sh
PYTHONPATH=python:. python benchmarks/results/shared-automatic-aot-20261027/verify.py --retained-only
```

It checks retained hashes, numerical/FD vectors, aliases, resources, work,
capacity, artifact bindings and every timing gate. This is historical evidence
verification and grants no qualification to the current compiler. Without
`--retained-only`, the verifier also requires the executing compiler to match
the original capability subjects; use the historical compiler at `c3ddb8861`
for that full check. An integrated compiler deliberately fails that identity
gate until separately qualified. Original capability envelopes and measurement
identities remain unchanged. This optional verifier is not a required CI
fingerprint of production source. `drivers/`
retains formatted copies of the finite build/GPU/benchmark recipes. Adapt checkout/toolchain paths
on another host, preserve Slurm-assigned visibility, and use only available
nodes n1/n2/n4/n5. No release, binary upload or runtime compiler is required.
JSON uses lossless gzip or XZ through the repository's standard record reader;
compression preserves every sample, numerical value and original JSON byte.
