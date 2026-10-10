# Producer-bound restricted PBE0 default

Measured clean Git-archive source `99ebd196df529bdd45107af44b86b7879d3bb69f`,
aligned to master `4444d0376`, not subsequent master `4b330ff3d`.
CPU-only official build: Slurm 7098. Complete endpoints: Slurm 7106 on RTX 5090.
Same official binary, explicit `off` versus unset/default, five interleaved
samples per arm for warm and moved-warm E+F, including synchronization and host
return. Each measured call performs one SCF iteration/Fock build.

| Atoms | Replay | Off median (s) | Default median (s) | Reduction |
| --- | --- | ---: | ---: | ---: |
| 48 | warm | 5.532675 | 5.271843 | 4.714% |
| 48 | moved-warm | 5.559617 | 5.323170 | 4.253% |
| 96 | warm | 16.203113 | 15.665374 | 3.319% |
| 96 | moved-warm | 16.166703 | 15.627425 | 3.336% |

All four pass the existing >2%-and-robust-noise gate. Work and actual default
restricted-point counts are retained; general points are zero in the admitted
default arm. Independent maximum E/F errors are below `1.1e-11 Eh` and
`3.8e-11 Eh/Bohr`, within unchanged `1e-8`/`1e-7` gates. Each arm's seed remains
immutable; cross-arm densities are not bit-identical (maximum `1.27e-11`).
Coordinates and semantic AO/force work match. No cold/reconvergence,
GPU4PySCF-relative, UKS/HVP, or latest-master timing claim is made.

The publication envelope accepts numerical qualification. Separately retained
endpoint timing assessments pass, but its resource-complete performance gate
remains unavailable: global peak memory and complete build duration were not
measured. Free-device snapshots and file timestamps are not substituted for
those measurements. Bound evaluation borrows existing scratch without a new
allocation. Static registers/stack are not performance evidence.

## Contents and reproduction

Use the shared publication reader for checksum-bound `samples.json.gz` and
`validation.json.gz`. Samples retain both original records, setup/priming calls,
all 40 measurements, seed-checkpoint identities, and independent E/F references.
The verification summary includes successful raw NPZ checks; NPZ seeds and
binaries remain ignored local evidence, not tracked inputs or external assets.

Restore master `4444d0376` and apply `measured-source.patch.gz` to reconstruct
the measured source. Alternatively check out the exact measured revision.
`recipes.json.gz` retains the original build/endpoint drivers, their helper,
environment scripts, offline verifier, source-manifest identity and static
resources. Numerical vectors and diagnostic histories live in the original
sample records; the envelope does not duplicate them. The path-bound original
source manifest remains ignored raw evidence and is regenerated on relocation.
Extract its script map into scratch; the helper belongs at
`evidence/bulk-point-v7/paired-ao-endpoint.py` and reference members at
`evidence/bulk-point-v7/reference-{48,96}-retained-perf5.json`. Reference identity
digests refer to the original JSON bytes; relocation must generate fresh digests
from the extracted references. Use a fresh source/evidence directory, relocate
site-specific CUDA/Python/compiler-wrapper/cache paths and produce fresh source
manifests. Basis input is the measured source's
`benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json`.

Build official native/grid/PBE0 AOT modules with verified compiler-cache
launchers and `GENERATIVEQC_CUDA_FAST_COMPILE=OFF`. The build/endpoint receipts
pin measured artifact and archive hashes; a reproduction rebuild does not claim
their binary hashes. Run every GPU command through finite `srun`, preserving
assigned `CUDA_VISIBLE_DEVICES`; do not use maintenance node n3. The original
endpoint recipe uses 35 minutes, and the offline verifier needs only CPU/NumPy.
No Release, tag, or external archive is created by this publication.
