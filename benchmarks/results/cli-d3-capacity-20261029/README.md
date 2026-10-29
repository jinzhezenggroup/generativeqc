# Native CLI D3 resource consumer qualification

Production source: `8b681c7e8` (full revision in `identities.json.xz`).
The native D3 capacity owner is unchanged; this adds its installed CLI consumer
on top of #2873's D4 composition. It does not execute or newly qualify molecular
CPU/CUDA science. All CUDA electronic test leaves are explicitly synthetic.

The current CPU build, with verified sccache 0.16.0, passes all **127 CTests**,
including **50 CLI runtime checks**. D3(BJ)/D3(0), RKS/UKS and batch sizes 1/3
compare CLI capacity fields and total peaks with the real public native D3
query. Exact and one-byte-below global host/device and local correction caps,
native shape refusal, unsupported hybrid/force/fitted contexts and inherited D4
behavior are covered. Eight additional installed CLI checks run with
`PATH=/nonexistent` and independently compare the installed library query.

Complete installed command timing includes process startup: **6,720 calls** in
eight alternating ABBA/BAAB groups, 21 calls per case/block. Existing RHF, PBE
RKS/UKS and PBE-D4 RKS/UKS inventories cover batches 1/4. Baseline is the frozen
#2873 production installation `1f9920fa5`; the candidate is the source above.
Every semantic JSON field matches except the intentional source/plan identity.
The maximum median time ratio is **1.0045711913970181**, below the unchanged
1.05 gate. No GPU device, SCF solve, Python runtime or compiler is used by these
installed metadata commands. Another CPU numerical campaign was active on the
same host; the node was not exclusive.

`samples.json.xz` retains every timing and output reference. `identities.json.xz`
retains exact source/binary identities, installed-query results, log hashes and
the measured replay-script text. The readable `qualify.py` may receive formatting
only; adapt its frozen installation paths before replay. Bulk build/CTest logs
remain ignored. No runtime performance or #2227 whole-slice closure is implied.

`storage.json` records lossless XZ storage of one historical CPU response record
to stay below the unchanged 80 MiB evidence limit. The original Git blob and
decoded/encoded hashes preserve every historical endpoint value.
