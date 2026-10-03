# Larger-system CPU qualification

This six-file capsule retains selected scalar evidence and the exact protocol
for the larger-system follow-up to [PR #1762](https://github.com/jinzhezenggroup/generativeqc/pull/1762).
Full raw arrays, logs, checkpoints and binaries remain local: this is not their
lossless publication or a durable backup.

## Results and limits

All 44 energy endpoints pass: eight pentane pilot, 32 separate balanced pentane,
and four candidate-only hexane endpoints. Direct RHF/def2-SVP, spherical fp64,
one CPU thread, unchanged maximum 150 iterations and convergence/oracle gates.
Geometries are idealized all-trans n-alkanes, not optimized structures.

For pentane (17 atoms, 130 spherical AOs), final baseline/candidate medians are:

| Metric | Baseline | Candidate | Reduction |
|---|---:|---:|---:|
| Whole-process peak RSS, KiB | 4,879,292 | 2,288,508 | 53.10% |
| Process-first energy endpoint, s | 36.3839 | 31.1787 | 14.31% |
| Fresh-owner warm endpoint, s | 35.1914 | 34.9566 | 0.67% |
| Changed-geometry endpoint, s | 34.3925 | 28.5608 | 16.96% |

Warm timing has three of eight matched losses and one of four pair-median
losses; no robust warm speedup is claimed. Iterations change from 21 to 24 for
original geometry and from 27 to 20 for changed geometry. They are not used to
normalize endpoint time or infer complete scientific work.

Hexane (20 atoms, 154 AOs) passes at 4.2447 GiB peak RSS. Its baseline and both
heptane arms (23 atoms, 178 AOs) are statically blocked by the stated external
memory admission, not measured failures. No speed ratio is reported for them.
There are no large-system force, full native density, full work-history or
repository resource-accounting acceptance claims. All failed oracle attempts
remain distinguishable from the four qualified strict recovery states.

## Identities and reproduction

Measured baseline: `e47058a51d8a6b01727d71267be7290df67077d1`.
Measured candidate: local `4796d967858b09e58c20d33241af460b42c10cbb`, with public
exact-tree alias `1378cdbc4e02502c969a3d1e4cd71cc7e1bb6340` and tree
`24a695119a10c9fa2872c8e56a6fed8b24ccfa67`. Neither the later merged master nor
unpublished scalar-solver experiments replace those measured identities.

Run `python3 decode.py` to verify, or
`python3 decode.py --output NEW_DIRECTORY` to expand hash-checked text into a
new directory. The decoder never executes recovered scientific code. Read
`review/REPORT.md` for the independent audit and `REPRODUCE.md` for auditing versus
a genuinely new build/oracle/freeze. The frozen driver's absolute paths and
build hashes require reviewed adaptation for a new run; it is not portable
unchanged. A new build must never inherit an old binary identity.

The payload retains all endpoint scalars/losses, 11 supervisors, eight oracle
states, the full 372-file hash/byte inventory, eight exact Python and two shell
protocol sources, the reference endpoint, three input JSONs and their manifest,
freeze/provenance/plans, and the final independent review. Only a duplicated
packaging narrative was omitted. Standard-library XZ stores ordinary UTF-8
members; no bespoke patch codec is required. The original staging versions and
full raw remain separate. Hashes do not make omitted originals recoverable.

The final 09:14:38–09:33:19 UTC window includes one transcript-reported permitted
CPU7 metadata read at 09:23:51 (134,360 characters), plus CPU8 monitor reads.
No independent observation of that exception, measured-negligibility, perfectly
exclusive host, or hidden cgroup-limit guarantee is claimed. No sample was
excluded for the exception.

`publication.json` binds the five other files; `SHA256SUMS` excludes itself and
that envelope. Formal performance remains inconclusive. `validation.json.gz`
uses the adjacent evidence schema. The bounded decoder rejects unsafe paths
and more than 8 MiB of decoded JSON. XZ decoder memory is capped at 128 MiB;
declared expanded counts/bytes and the generated validation path are checked. Direct JSON inspection is also possible
with Python's `lzma` module; decoded length and hash are in the envelope.
