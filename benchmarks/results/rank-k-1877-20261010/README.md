# #1877 rank-k standalone CUDA qualification

Qualified source commit: `68de92cd32a356c1c05e41c3935cc206da77a7ea`.
This is not a complete SCF/SCC
endpoint, cuBLAS promotion, or #1877's two-production-consumer completion.

## Accepted run

qz Job `i1877-rankk-h100-1010z10` exited 0 at 2026-10-10 08:08:48
Asia/Shanghai on H100 80GB HBM3 (sm90), driver 570.124.06, CUDA 12.9.86,
runtime 12090, cuBLAS 120902 and g++ 11.4.0. All 16 timed cases passed:
density/weighted density, row/column order, `3x5`/`17x9`, two batches and both
providers. A CPU `long double` oracle covers signed/negative-zero weights,
`alpha=1.25`, `beta=-0.5`, odd tails, asymmetric old lower data and two graph
replays. Untimed gates cover separate overwrite/update TensorIR roots,
`beta=+0/-0` without reading NaN old output, signed-zero alpha, alias/order/
resource rejection, overflow and nonfinite all-output preservation, and
mismatched request rejection. The nonzero update additionally binds asymmetric
physical storage through its upper-authoritative symmetric logical input.
The `3x5` and `17x9` rows use distinct fixed-shape scientific identities.

[qualification-summary.csv](qualification-summary.csv) retains every PASS
timing/work/resource row. Times use 20 iterations after four warmups and exclude
host transfers/preparation; the reported device reset bytes are timed. Generated
measured 12.5376--21.4512 us; cuBLAS measured 25.5776--35.7824 us and was slower
in all eight pairs.

## Identities and raw receipts

All 1,584 source rows passed. GPU regeneration from the actual host compiler,
CUDA headers, complete fixed device-tool closure and empty override environment
matched the staged header byte-for-byte.

| Item | SHA-256 |
| --- | --- |
| source / generated manifests | `21c7419f797e4677710473552e68e406cae4d32452e5bf7d59ac91a06c7e08c1` / `89340c640be56047b260ed48d73456c6ed458350ec8c3bd3f92a4737a5cdfb21` |
| raw JSONL / provenance | `c2aef03626a7dd75cabcae439e75d1bf447316dcb953a64fd0906a1b33c12296` / `40df80c3bb8b082b981da3e9bf50375a468de6812a468dc37c502a1cb3d64ece` |
| native header / harness | `87228fbe1b8bb53aaa84d2cdcb351797f61c35f670b847d76ce040a6816111fe` / `fce0f82637e76a881c29740eb13305dd770260cf3be7dfec73ca7101db98fa8f` |
| generated header / object / binary | `eb0dc64a0284446aa12a2331ff4ef5918be862089b0894ef19dcd93c06ae9b3b` / `e9ec4417a4a844848e49784dbdd9f2a24c3082f3dfcdf0aec344c8e18dad6a61` / `bd771f22635c21999de5f8dd39e1e295bedd46dec565700266e0740b5564f5a6` |
| sccache before / after | `5bad79e9df3dfc23bee25fcecb1877afc5ddbfb581694ec044456bb16f065b81` / `56609624c5c496b26a3d0a956256fb20c2781e9983b39830a1080ec4407f6bec` |
| host compiler / Python | `d7122fd9a7a8fe12d12c00c54d3a6fbebcb3e9285cf675709674e751d900fc63` / `6ce8f488520d0f0cb8ccc405ba844279492c3926e18b73f0539e5fecd8033467` |

The CUDA closure includes `nvcc.profile`, device children, libdevice, link stub
and libraries. The staged/actual 3,666-role GCC closure manifest hashes to
`9092a8166628cbd431a48f50fec09fb8484f273dc18fafcf046764a44b0d652c`.

Raw Job `z10`: qz
`/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1877-rankk-1010/results/z10`.
Independent raw-receipt review requires retrieving it.

`sccache 0.16.0` wrapped compile/link: one CUBIN hit, three misses, four
compilations, one link pass-through, and zero errors/unsupported calls.
Receipt-only commits may reuse Job `z10` only while the qualified source blobs and
modes remain identical and latest-head review verifies that boundary.
