# #1877 rank-k standalone CUDA qualification

Qualified source commit: `c4b561f9957f696d9c43ed4a9ce4407efc6f5742`.
This is not a complete SCF/SCC
endpoint, cuBLAS promotion, or #1877's two-production-consumer completion.

## Accepted run

qz Job `i1877-rankk-h100-1010z8` exited 0 at 2026-10-10 07:27:15
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
host transfers/preparation; the reported device reset bytes are timed. Generated measured
13.0928--20.7408 us; cuBLAS measured 25.5216--35.7040 us and was slower in all
eight pairs.

## Identities and raw receipts

All 1,581 source rows passed. GPU regeneration from the actual host compiler,
CUDA headers, complete fixed device-tool closure and empty override environment
matched the staged header byte-for-byte.

| Item | SHA-256 |
| --- | --- |
| source / generated manifests | `ade93dbd4c25345560ef55470e5a80396f0436b13d564a692e7cf726b0c73141` / `3569a9d2bd4d2412bf0316d76d3db726a3d92c22d25e08476827bb4dc4d2d77b` |
| raw JSONL / provenance | `1af89222a3f67452371078e179a8a8aaaa5bf136c85aa0536f0644b810045276` / `3ab6cb26b18cde31b27007db17e966052b5a83fa9a9c244b5cee5b674ac0931c` |
| native header / harness | `3265164d7337d6de712fe53b99c5b8a6e29f89e080b0ee0ca99427101ffdac98` / `99e0644f6c55b745460becdc5bd144a6d2cb7e8691022d59a4f91cc3f25bedf5` |
| generated header / object / binary | `6f012a1ba37971b880202e7059fc557811a4e46adbc3abf168ae40e0da74c41c` / `15406216d0ab773eab123567876e1a52fabbe7f00c23a916f107c7194cbcea21` / `cff79722037cdc59f270e1344af27dd7f086cd74d2cbcadc08c5d605278094ae` |
| sccache before / after | `5bad79e9df3dfc23bee25fcecb1877afc5ddbfb581694ec044456bb16f065b81` / `6d264c8275aa530f7de703daf490c3f6a09334a9cee0fc855978a4a83fc335cc` |
| host compiler / Python | `d7122fd9a7a8fe12d12c00c54d3a6fbebcb3e9285cf675709674e751d900fc63` / `6ce8f488520d0f0cb8ccc405ba844279492c3926e18b73f0539e5fecd8033467` |

The CUDA closure includes `nvcc.profile`, device children, libdevice, link stub
and libraries. The staged/actual 3,666-role GCC closure manifest hashes to
`9092a8166628cbd431a48f50fec09fb8484f273dc18fafcf046764a44b0d652c`.

Raw Job `z8`: qz
`/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1877-rankk-1010/results/z8`.
Independent raw-receipt review requires retrieving it.

`sccache 0.16.0` wrapped compile/link: one CUBIN hit, three misses, four
compilations, one link pass-through, and zero errors/unsupported calls.
Receipt-only commits may reuse Job `z8` only while the qualified source blobs and
modes remain identical and latest-head review verifies that boundary.
