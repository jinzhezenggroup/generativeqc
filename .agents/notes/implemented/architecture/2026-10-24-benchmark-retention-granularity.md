# Decision: retain benchmark measurements separately from execution inventories

Status: implemented
Date: 2026-10-24

## Problem

At published master `8e74c66de97e2bfacd655976888c0e0abb252ace`, tracked
`benchmarks/results/` contained 4,437 files / 74,949,491 bytes (71.48 MiB).
The user's earlier `7973502a` audit measured 70.21 MiB; these are different
snapshots, not inconsistent units. JSON compression alone had not prevented
growth: source/environment censuses and execution streams were repeatedly
embedded alongside measurements. Standalone old experiment drivers also crowded
the maintained benchmark entry points.

## Decision

- Consolidate 245 complete historical JSON records from 26 completed campaigns
  into bounded, authenticated collections of JSON values. Preserve every number
  and string token and array order. Small decisions and Markdown remain readable.
- Retire old copied drivers, patches and supporting execution files into a
  checksum-verified snapshot of published master. Retire ten standalone historical
  benchmark entry points with no current maintained caller. All original bytes
  remain recoverable without a new upload or Release.
- Replace the Direct phase/moved source/SDK/environment file censuses with measured
  revision, library/SDK identities, versions, observed counts and original hashes.
  Its numerical audit does not consume those censuses and now runs without them.
- Replace repeated PBE0 schedule source-verification listings with original stream
  identities, checked-file counts and failed lines; retire raw build logs. Keep
  both complete timing populations, failures, independent vectors, source patches,
  work counts and all scientific qualification checks.
- Keep current ownership-ledger pointers valid and provide a benchmark entry guide.
  Future runs share source/environment identity rather than copying inventories.

Against integration base `8f153f4f9`, the results tree changes from 4,450 files /
75,519,799 bytes (72.02 MiB) to 4,121 files / 67,783,780 bytes (64.64 MiB),
a 7.38 MiB / 10.24% reduction. All of `benchmarks/` changes from 4,713 files /
78,898,198 bytes to 4,375 files / 71,070,161 bytes (75.24 to 67.78 MiB).
Upstream additions since the original audit snapshot are retained. The recovery
inventory stays pinned to the published original snapshot, whose selected files
are unchanged in the newer base. These are current tracked contents, not
compressed packfiles or historical clone size.

## Is JSON inefficient?

The read-only `tools/evidence.py audit-json --revision <SHA> --output <path>`
audited all 2,765 JSON files in the original snapshot, with no decoding failures.
Counts include manifests whose suffix is also `.json`.

| Original format | Files | Stored bytes | Expanded bytes | Minified bytes | Minified gzip-9 | Minified XZ-6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| JSON | 1,420 | 16,579,980 | 16,579,980 | 12,383,523 | 3,261,142 | 3,048,912 |
| JSON.gz | 777 | 12,771,687 | 107,081,618 | 72,320,167 | 11,382,128 | 8,592,060 |
| JSON.xz | 568 | 28,442,636 | 547,418,575 | 499,755,025 | 68,221,174 | 28,588,800 |

Whitespace removal saves 25.3% of plain JSON. Across the whole corpus, however,
applying identical XZ-6 settings before versus after whitespace removal changes
41,149,112 to 40,229,772 bytes: only 0.88 MiB. Some existing XZ files use stronger
settings, so the table is not a promised improvement over current storage.
Replacing XZ with gzip would substantially increase the largest category.
Repeated field names compress well; high-entropy source-file hashes remain costly.

Dense arrays deserve typed storage. A local comparison of the exact retained
`density-candidates/inputs/water4_svp.npz` arrays (including JSON dtype/shape
metadata and verified reconstructed array bytes) measured 2,430,607 bytes for
compact JSON, 732,070 for gzip JSON, 409,308 for XZ JSON and 475,350 for NPZ.
Thus even binary storage is not uniformly smaller than compressed JSON. Typed
arrays retain dtype/shape and avoid decimal parsing; small metadata and timing
records benefit from JSON's inspectability. Existing NPZ/NPY fixtures are unchanged.
This comparison does not qualify any GPU or scientific endpoint performance.

## Rejected alternatives and invariants

Bulk codec conversion alone retains unnecessary content, changes all storage
identities and may worsen compression. Escaped JSON inside JSON also prevents
direct structural inspection. The historical collection format stores actual JSON
values and is not an admission path for new raw bundles.

Do not infer that an old report is a permanent scientific fixture from a hash-only
test. Conversely, do not remove independent references or numerical acceptance
coverage because a campaign is old. Current numerical verification remains offline
and does not depend on historical Git objects. Old environment/build audits are
explicitly separate, with a full published recovery revision and exact digests.

## Validation

The migration audit restored and verified all 464 original files. It independently
compared all 245 retained records against restored originals after whitespace-only
removal. Direct phase/moved produces an identical original admission report; both
PBE0 schedule reports recompute identically, and rehashed corrupt samples still
fail their numerical/work/identity gates. Synthetic tests cover precise decimals,
large integers, signed zero, escapes, duplicate keys, malformed/trailing streams,
checksums and unsafe paths. Repository evidence/size/compaction checks retain their
existing 1 MiB per-file, 2 MiB incoming-change and 80 MiB checkout limits.

Revisit when a current scientific consumer needs an archived member, or a measured
format comparison justifies changing an active publication's storage contract.

Recovery and scope: [migration inventory](../../../../benchmarks/results/retention-granularity-20261024/README.md).
Current policy: [evidence retention](../../../../docs/maintainer/evidence_retention.md).
