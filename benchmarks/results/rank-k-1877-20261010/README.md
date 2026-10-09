# #1877 rank-k standalone CUDA qualification

Qualified source commit: `3715893dc2f15789bf0b8cd52384a6558ace4c6e`.
This is not a complete SCF/SCC
endpoint, cuBLAS promotion, or #1877's two-production-consumer completion.

## Accepted run

qz Job `i1877-rankk-h100-1010z3` exited 0 at 2026-10-10 06:38:51
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
transfers/preparation. Generated measured
13.0992--21.4432 us; cuBLAS measured 25.5040--35.8896 us and was slower in all
eight pairs.

## Identities and raw receipts

All 1,581 source rows passed. GPU regeneration from the actual host compiler,
CUDA headers, complete fixed device-tool closure and empty override environment
matched the staged header byte-for-byte.

| Item | SHA-256 |
| --- | --- |
| source / generated manifests | `2ed74bb1cb0250978a3acffdbf46f02872ef0f0e1cfa6a7058c1b717cfe34db3` / `7019aabb4177e8145c3156487d60a67114be36f21e6c2c01e878e0291f237b89` |
| raw JSONL / provenance | `354b4bd2b0445f13c9ddb59bb6533b8fea449c4b688f23f3ada402a19034456f` / `d2d58cd7f9ff3da53cf0db13c509e0b681c2433ddafab45c02b5aaa48ac163b7` |
| native header / harness | `433c9e0562a64be219c8385e98d835e66da38fdeb02dcd529d078e015aa33ca7` / `cee2011891b65b3c84db308aa6c61135df9cbef71ef922e192d2f1c1d05c9775` |
| generated header / object / binary | `f7ac1c9be830ab63e6a85a03007a1ae732be11c649850da2784496e95d4046bd` / `3c216444330b55bf4b4c195eeb14eeead5c963ba62cb3e9ec2f2433c49db3138` / `5773cf643e469c84e56d12610fad2b172c9153a5a96232b599462a8f5ab58288` |
| sccache before / after | `5bad79e9df3dfc23bee25fcecb1877afc5ddbfb581694ec044456bb16f065b81` / `3510a720d508422b1d62de38707486d5fb7e8a2a924789ffa72c8c06189a1160` |
| host compiler / Python | `d7122fd9a7a8fe12d12c00c54d3a6fbebcb3e9285cf675709674e751d900fc63` / `6ce8f488520d0f0cb8ccc405ba844279492c3926e18b73f0539e5fecd8033467` |

The CUDA closure includes `nvcc.profile`, device children, libdevice, link stub
and libraries. The staged/actual 3,666-role GCC closure manifest hashes to
`9092a8166628cbd431a48f50fec09fb8484f273dc18fafcf046764a44b0d652c`.

Raw Job `z3`: qz
`/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1877-rankk-1010/results/z3`.
Independent raw-receipt review requires retrieving it.

`sccache 0.16.0` wrapped compile/link: one CUBIN hit, three misses, four
compilations, one link pass-through, and zero errors/unsupported calls.
Receipt-only commits may reuse Job `z3` only while the qualified source blobs and
modes remain identical and latest-head review verifies that boundary.
