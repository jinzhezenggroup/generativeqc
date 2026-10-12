# CUDA DF complete-force physical replay batching

Numerical qualification of the bounded #2221 replay consumer extension,
independent of #2298. Measured clean parent:
`34dd51ed3ebc5b03b17f095de2cd7b39903923c8`; the lossless production overlay
reconstructs the one changed dispatch source exactly. Both staged endpoint and
probe binaries are identical. Generated equations, iterations, Lambda/Z,
precision and scratch ceilings are unchanged.

Four cold native force processes, ethane230/o9/v221/Q488, FP64, DIIS8,
64 GiB correlation budget, CC Q8 and Lambda Q32, Slurm 2900/node2:

| Median seconds | Original replay | Bounded replay |
| --- | ---: | ---: |
| Complete process wall | 275.611584 | 274.532346 |
| Complete CCSD | 61.066811 | 59.878997 |
| Fresh physical replay | 9.668271 | 8.414243 |

Complete observed saving: **0.3916%**. Both candidate samples are faster than
both controls; this is descriptive, not statistical/global performance or a
latest-master endpoint claim. Actual replay Q batch is four (488 -> 122 tiles),
not sixteen. CC GEMMs 54,862 -> 45,468; semantic contraction work unchanged.
Complete logical capacity unchanged at 19,509,839,897 bytes; CCSD bounded host
bindings add 56,870 bytes. Physical peak VRAM is not measured.

All 38 independent small force/public gates and four large energy/residual/
Lambda/Z/force-direction records pass. Maximum independent direction error
7.11e-9 Eh/bohr; matched force difference 7.64e-14. Memcheck reports zero errors.
The byte-identical baseline's previous independent qualification is reused.

`raw-receipts.json.xz` stores every receipt with exact lengths and SHA-256,
including timing runner, finite Slurm recipe, binary hashes, original JSON,
resources, test XML, sanitizer, work queries and baseline source. Reconstruct
the source with `production-source.patch.gz`; run the retained qualifier with
`srun --partition=main --gres=gpu:pro6000:1 --nodes=1 --ntasks=1
--cpus-per-task=2 --time=00:30:00 bash qualify-feature.sh batch-replay
candidate-batch-replay`. Use matching staged libraries and preserve assigned
`CUDA_VISIBLE_DEVICES`. No Release, tag or external archive is created.
