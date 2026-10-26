# One source-bound NVIDIA provider initialization case

This engineering record preserves one H100 initialization experiment at
`e87cad557a81ff956e210efa2d76c770f0fdd41f`, with one vendor owner and two retained
allocation guards. The observed free-memory delta was **375390208 bytes (358 MiB)**.
The legacy 192 MiB combined allowance correctly refused it with the existing
`length_error`; this experiment's explicit 512 MiB allowance accepted it. Drain,
parameters, solver, BLAS and stream cleanup each returned zero. Current context,
device, four directly loaded build identities and library file identities were
stable before and after the case.

This is **initialization-only engineering acceptance**. There were zero numeric
solver, matrix, geometry, model, SCF or scientific endpoint calls. The 512 MiB
choice is confined to this experiment: it establishes neither a global H100 cap
nor a peak bound. The 1 GiB `B_ref` argument is only a reservation lower bound;
the complete RHF arena/workspace budget remains unavailable. The default stays
96 MiB per enabled provider. Ordinary SDK request transport is not implemented;
full SDK/RHF and scientific acceptance are not asserted.

The independently observed environment was H100 80GB HBM3, driver 595.58.03,
CUDA runtime API 12090, driver API 13020, g++ 11.4.0 and sccache 0.16.0. Exact
tool, entry-header, four DSO, container, physical-device and loaded-build
identities are in the retained records. Driver and runtime API versions have
different meanings. GNU build IDs identify linked builds; file SHA256 values
do not prove transitive allocation behavior.

## Inspect and restore without running the experiment

The reader uses only Python's standard library. It never compiles, dispatches,
loads CUDA/vendor libraries, fetches a source tree or executes restored files.
Run from the repository root:

```console
python benchmarks/results/provider-initialization-h100-20261026/reader.py
python benchmarks/results/provider-initialization-h100-20261026/reader.py --restore .artifacts/provider-init-restored
```

The output directory must not exist. `originals/` restores 52 captured public
records with the declared exact UTF-8 bytes. `executed/` restores all **53 actual
non-binary files**, including the native bridge, runtime translation unit and
header closure, runner, manifest, and the actual bound parameters. Each is
checked against the independently observed terminal inventory's name, byte
length, SHA256 and POSIX mode. The bound-parameter file retains mode 0600.
On POSIX the reader also checks permissions after restoration. Windows reports
that native POSIX permission validation is unavailable; its permission model is
not substituted for a POSIX check.

Historical parameters contain an expired finite deadline and an opaque decision
digest. They provide provenance, not permission to run or reuse a resource.
Any future execution needs its own authorized inputs and environment. The
account and notebook literals in the retained collector are placeholders.

## Retained information and precise limits

| Record | Contents |
| --- | --- |
| `inputs.json.xz` | Frozen source packet, literal source payload, manifest, preparation metrics, 35 staging commands plus one complete verification command, collector and actual bound parameters |
| `observations.json.xz` | All 38 captured CLI results, actual runner and bridge events, read-only software intake and terminal file inventory, and their transport receipts |
| `manifest.json` | Storage and expanded hashes, original/public member identities, redaction descriptions, source scope and three compiled-product inventory rows |
| `offline-verification.json` | Actual POSIX restoration result, reader/manifest identities and six rejected malformed or unsafe inputs; metadata validation only |

The XZ members preserve the captured record bytes, including numeric tokens,
failed legacy admission and successful selected admission. The reader verifies
the original 21-file LF/mode packet, actual 51-file staged manifest, all 53
restorable executed files, 56 terminal inventory rows totaling 838711 bytes,
38 CLI return codes, original staging-command digests, native event extraction,
stable identities, guard arithmetic and cleanup outcomes. Timing scopes remain
separate in their original records; their intervals are not summed into an
endpoint or billing measurement. Compiler-cache counters are not process-exact
attribution.

Three compiled products are **inventory-only and remotely retained**; their
contents are not in this publication and cannot be reconstructed from hashes:

| Name | Bytes | POSIX mode |
| --- | ---: | --- |
| `native_init_bridge` | 105208 | 0755 |
| `native_init_bridge.o` | 130280 | 0640 |
| `runtime_identity.o` | 38504 | 0640 |

Private decisions/controllers, account/session bindings, chats and personal
policies are excluded. The decision is an opaque SHA256 binding only. The
collector's two private transport literals and local user paths in 40 transport
stderr fields are redacted, together with the notebook-derived hostname prefix
in the read-only intake result and its captured stdout. Every affected member
retains both original and
public byte lengths and SHA256 values, with the exact redaction scope. These
public copies are not described as byte-identical private originals. All actual
executed source, bound-parameter, runner/bridge-event and inventory records are
unchanged.

Offline validation restored all 105 public-record/source files and checked all
105 POSIX modes, including the executed parameter file's 0600 mode. It also
rejected a corrupt stored source member, trailing and truncated XZ streams,
duplicate JSON keys, a traversal path and an existing output directory. These
checks do not rerun the device experiment or its earlier HOST tests.

There is one pre-existing information gap: CLI 37's original collector had
already replaced one URL with `[REDACTED_URL]`. Its pre-scrub stdout was recorded
as 131790 bytes with SHA256
`73317dd71c5a7e793c755a71605fa3a6fde016af2e7166e9466180b6cd2fc8fc`;
the captured text is 131771 bytes. The reader authenticates that captured text
and preserves the original digest as **UNAVAILABLE for exact reconstruction**.
The other 37 CLI stdout streams match their original byte lengths and hashes.
The actual vendor child's stdout is independently byte-identical: 5278 bytes,
SHA256 `34167df89da20deafcf7d44a5eee3cc1c3d7f3e28a277c2c7e49dab3e388a467`.
No missing original is inferred or filled. The private captured originals remain
unchanged outside this publication.

## Source mapping and earlier failures

The measured source packet is SHA256
`78b0faa04e19e773bf6223e4fb3650eaa0befcbc1eb6823757b647c9c06795f4`;
the v5 staged-source manifest is SHA256
`00c1baab1fd649f82454b5796f5a2645d299cbc2e53e11ec2241dffd90410023`.
The following `a50d1114a752fe211b97b3df3ef9e745311e0e4f` commit changes only the
device-facts HOST fixture, preserving the preceding 21 files and compilation
closure. Its four affected HOST tests passed after a real missing-driver-header
compile regression was reproduced. This record remains an e87 experiment;
Root independently verified that mapping, including the seven production files
in the actual compilation closure; no additional GPU execution was needed for
the test-only delta. CI and whole-head SDK/scientific acceptance remain separate.

Earlier ea0 namespace compilation failures, e87's separate HOST fixture failure
and cumulative 30-minute core-a cancellation remain in their original CI and
ignored diagnostic records. This publication does not relabel them or claim the
short fixture failure caused the cumulative timeout. It introduces no new
measurement, numerical oracle, benchmark default or permanent CI test of source
function bodies, AST shapes or hash spellings.
