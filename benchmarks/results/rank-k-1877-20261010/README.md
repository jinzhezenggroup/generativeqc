# #1877 rank-k standalone CUDA qualification

Qualified source commit: `2268c159cefb7320e75fd822aa55c43d8166728c`.
This is not a complete SCF/SCC
endpoint, cuBLAS promotion, or #1877's two-production-consumer completion.

## Accepted run

qz Job `i1877-rankk-h100-1010z5` exited 0 at 2026-10-10 07:01:17
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
13.0544--21.4256 us; cuBLAS measured 25.8048--36.4608 us and was slower in all
eight pairs.

## Identities and raw receipts

All 1,581 source rows passed. GPU regeneration from the actual host compiler,
CUDA headers, complete fixed device-tool closure and empty override environment
matched the staged header byte-for-byte.

| Item | SHA-256 |
| --- | --- |
| source / generated manifests | `81709a45449b048c8d2e8637077098a8a93abc163695a561ef52049ad25bcdf4` / `e7a48bab5ea1d109b780fd73e93343b3851e947e08def69f3bbee566acdb39e6` |
| raw JSONL / provenance | `c2ae523fce729780871acf3c1a9b5b4bc71de1c04264d54ccad2f3d1b8adbcf8` / `8b8cf9b1b9d0e7f8cc43f60444552e9f90e79c936c4ebe6d798d70fb3bfc2390` |
| native header / harness | `ec8c2c69a80f9f82786165b03c91a2e3cec66a81a2cd793a9f07c075b1cef2b6` / `efb0507e54e07cab50b46c35cd6df7ee50a0388b5eb8a942b80c76073afbf0d4` |
| generated header / object / binary | `2eb783fcd346489e1499867d8c79383e62d6ecec6ef977c9d69b9bfb5a6d5a94` / `b46e61e7fb6e94ad19879f7c7e6c4a474e8356db4fe281196e791f2f4439a30e` / `ade05ceef952013e686a8699553b746421a3fbadca5274faf47384901278265e` |
| sccache before / after | `5bad79e9df3dfc23bee25fcecb1877afc5ddbfb581694ec044456bb16f065b81` / `76b79ff46cf552972fb8362d1c4c4c5ce6cc4d5b447d5a3c0f650938b83b88ab` |
| host compiler / Python | `d7122fd9a7a8fe12d12c00c54d3a6fbebcb3e9285cf675709674e751d900fc63` / `6ce8f488520d0f0cb8ccc405ba844279492c3926e18b73f0539e5fecd8033467` |

The CUDA closure includes `nvcc.profile`, device children, libdevice, link stub
and libraries. The staged/actual 3,666-role GCC closure manifest hashes to
`9092a8166628cbd431a48f50fec09fb8484f273dc18fafcf046764a44b0d652c`.

Raw Job `z5`: qz
`/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1877-rankk-1010/results/z5`.
Independent raw-receipt review requires retrieving it.

`sccache 0.16.0` wrapped compile/link: one CUBIN hit, three misses, four
compilations, one link pass-through, and zero errors/unsupported calls.
Receipt-only commits may reuse Job `z5` only while the qualified source blobs and
modes remain identical and latest-head review verifies that boundary.
