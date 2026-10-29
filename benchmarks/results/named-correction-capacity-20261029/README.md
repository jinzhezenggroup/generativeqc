# Named correction capacity-promise guard

Qualified production source: `9ce18971351c2bda69e4e5732c67097e11969d04`.
The guard's native source is unchanged from `9f3b587fd`; the qualified build
includes merged master build/compiler inputs and has its own binary identity.
Baseline native source is `4fc8782cfa3b8ddbcaad8bbda4eed002e8aea5c8` (#2870).
Exact source, binary, input, log and profiler identities are in `identities.json.xz`.

Slurm job **14861** first reproduced the defect against the baseline: named
PBE-D3(BJ) preparation exposed 105,701,171 unspent promised bytes from an
electronic-only bound that omitted correction owners. The expected probe failure
is retained. Numeric allocations still obeyed their cap; this was an incomplete
future-owner promise and optional-storage admission defect.

The fixed build passed **127 CPU CTests**, an installed independent D4 SDK replay,
and finite one-hour Slurm job **14863** on one assigned RTX 5090 (node1/main).
The GPU job passed the native capacity suite, six installed named D3(BJ)/D3(0)/D4
RKS/UKS E/F endpoints with Python/compiler absent from PATH, the independent
pinned D4 SDK reference, sanitizer memcheck with zero errors/leaks, and 30 Python
checks. Native regression coverage includes no incomplete scalar/fleet promise,
E/F parity, moved-coordinate replay, one-byte refusal and zero teardown charges.
Scheduler visibility was preserved; the node was not exclusive. The exact SDK
compiler inputs are retained in `sdk-measured-source.json.xz`; the text copies
received only clang-format formatting after measurement.

Ordinary unbudgeted complete endpoints passed **1,792 calls in eight ABBA/BAAB
groups**: maximum median ratio **1.0181265511751556**, below the unchanged 1.05
gate. The matrix covers PBE/PBE0-D4 RKS/UKS, scalar and ragged H2/water fleets,
E/E+F, cold/warm/repeated and displaced replay, including preparation and cleanup.
Both use def2-SVP spherical basis, a 24/12/24 grid, default CUDA SCF controls and
poisoned Python correction construction. Published semantic work matches;
maximum E/F differences are 8.53e-14 Eh / 4.36e-14 Eh/Bohr. Raw populations and
all ratios are in `cuda-samples.json.xz`.

The complete untimed profiles match: 172,628 kernels, zero graph calls, 5,648
explicit dynamic allocations, 119,201,475-byte live peak and zero final live
bytes; all copy counts/bytes match. Static and opaque provider/driver memory are
excluded from that dynamic peak. Bulk profiler databases, XML and logs remain
ignored, with hashes retained here. All 13,595 frozen inputs passed before/after
digest verification, and the running Slurm controller receipt is retained.

This ordinary timing pass does **not** qualify the affected raw NumericCap path
for performance: named corrections retain capped ordinary allocation and may
lose optional XC storage until a complete correction-inclusive bound is proven.
Selected-inventory behavior already had no such promise. This guard does not
repair #2872's separate budget regression or complete #2227 R2/R6; R3 is outside
scope. Replay `run-gpu.sh` only through finite `srun`, adapting frozen paths.

To keep the existing 80 MiB evidence limit, the historical independent-reference
JSON named in `storage.json` is stored losslessly as XZ. Its original Git blob,
decoded/encoded checksums and every reference value are preserved; no numerical
coverage or acceptance criterion is removed.
