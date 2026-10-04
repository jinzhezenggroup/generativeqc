# Decision: bounded optional ERI residency for CUDA RHF reference export

Status: implemented
Date: 2026-10-03

## Problem

CCSD and CCSD(T) request a physical CUDA RHF reference. The shared owner
previously disabled ordinary small-system ERI residency for reference export
and recomputed unscreened integrals inside every matrix-direct Fock build.
After correcting CC value-source admission, the 56-AO energy endpoint still
spent 775/136/866 seconds in cold/warm/changed-geometry RHF on node2. The MO
provider and correlation solve together took only about 37 seconds.

## Decision and invariants

Reuse the existing dense ERI producer and Fock consumer within the shared RHF
owner for eligible s/p physical-reference exports. Do not add integral formulas,
a CC-specific RHF solver, CPU oracle work or another physical reference contract.
Keep the matrix-direct arena. After querying the actual device and host solver
workspaces, admit an optional device ERI allocation only if the complete
reference budget can hold it. Cap this quartic cache at 256 MiB independently of
the caller's budget. Higher angular momentum remains on the direct evaluator.
The normal HF small-system selection and generated force ownership are unchanged.

The optional allocation has its own bucket resource owner and device-capacity
accounting. Release it in stream order. A budget shortfall or an allocation
failure selects the existing bounded fallback; other CUDA errors propagate.
Recompute geometry-dependent values after coordinate changes. Recheck solver
and cache bytes on every cached execution, not only on first setup. Exported
physical P/F/C/epsilon still pass the same stationarity and canonical gates.

The existing optional progress journal reports completed reference work after
stream synchronization: AO count, SCF iterations, physical Fock builds, resident
ERI bytes/values constructed and complete reference capacity. These counts are
not host graph-construction calls. A zero resident-value count on the direct
route does not mean zero integral work; that route recomputes during Fock builds.

## Evidence

The [retained final-binary evidence](../../../../benchmarks/results/cc-rhf-resident-20261003/summary.json.gz)
contains every cold, twice-warm and changed-geometry endpoint, immutable
source/binary identities, Slurm provenance, work observations and independent
exact-basis PySCF 2.14.0 gates. Warm 28-AO energy decreases from 13.61 to 4.25
seconds (3.20x), and warm 28-AO forces from 27.19 to 17.92 seconds (1.52x) on
node1. Cold/changed 28-AO energy decreases from 61.54/55.46 to 4.79/4.28 seconds;
force endpoints decrease from 75.06/68.95 to 18.66/17.89 seconds. On node2,
56-AO cold/warm/changed energy decreases from 812.58/173.93/902.89 to
67.23/66.72/66.76 seconds; the warm improvement is 2.61x. Nodes were shared and
the pairs used separate allocations; there are two warm samples per variant.

Public CC work counts are unchanged. Each 28-AO reference emits 614,656 ERI
values into a 4,917,248-byte cache; each 56-AO reference emits 9,834,496 values
into 78,675,968 bytes. Warm calls reuse those values for four/three physical
Fock builds, respectively. The complete reference peaks are 215,319,548 and
291,941,964 bytes, including mandatory provider/workspace allowances. The
baseline RHF Fock counts were not instrumented; the candidate's completed
journal is not a retrospective baseline measurement.

All 40 public CPU/CUDA tests pass. Two remote pytest compiler wrappers skip
because node1 lacks ccache; the identical native probes were compiled locally
with ccache and run in Slurm job 5404 for Cartesian and spherical inputs. They
compare physical energies, density, Fock matrices and orbital energies against
independent CPU RHF, execute the exact optional-cache boundary and one byte
below it, execute the exact minimum fallback budget, reject minimum-minus-one,
and inject optional device allocation pressure through the real numeric ledger.
Both observe 19,208 resident bytes, an exact 8,435,684-byte fallback peak, one
optional rejection and zero live ledger bytes after release. Changed geometry
and higher-angular fallback also pass. The native resource probe and all four
14-AO complete force calls pass memcheck with zero errors. Host tests cover
overflow, the storage ceiling, force exclusion and angular/admission boundaries.

Maximum retained total-energy, triples and force errors are 5.4e-12 Eh,
1.6e-13 Eh and 6.7e-8 Eh/bohr. Memcheck timings are retained but excluded from
speed comparisons. No reference-oracle work enters production execution.

## Rejected alternatives and revisit conditions

Removing the old reference guard from ordinary persistent ERI selection would
only cover <=16 AOs and could turn optional storage into a mandatory rejection.
Raising that ordinary threshold globally would change unrelated HF workloads.
Neither preserves reference-specific queried workspace admission. Enabling a
second CC-owned integral/Fock implementation would duplicate scientific owners.

This cache is rebuilt for each standalone RHF call and is not borrowed by the
following MO provider. Revisit cross-phase or prepared-endpoint reuse only with
explicit geometry/source identity, charged overlapping lifetimes, independent
reference gates and full endpoint measurements. Revisit the storage ceiling or
angular domain only with complete workload and resource evidence. The public
CCSD(T) force boundary remains 28 AOs; the separate 56-AO degeneracy failure is
not addressed by this scheduling change.

The later [bounded reference-quartet decision](2026-10-03-reference-bounded-quartets.md)
supersedes the matrix-direct choice for d/f references and s/p topologies beyond
the cache ceiling. It preserves the optional-cache policy and low-budget
fallback documented here for cache-eligible s/p systems.

## Lossless evidence storage

The original records and summary use deterministic gzip. The
[storage map](../../../../benchmarks/results/cc-rhf-resident-20261003/storage.json)
pins their original Git blobs, original/stored SHA-256 hashes and byte counts.
Read either JSON with `tools.generativeqc_validation.record.load_json`, or recover
its exact original bytes with `gzip -cd FILE.json.gz`. Historical paths inside
the unchanged records resolve through this map; historical Git objects remain
available. This changes storage only, with no new experiment, measured build,
scientific value, failure record or performance claim.
