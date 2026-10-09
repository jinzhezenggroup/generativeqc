# #1877 rank-k standalone CUDA qualification

This qualifies the prepared capability at source commit
`451045b18613281a2da61212012478cba9900a05`; it is not a complete SCF/SCC
endpoint, cuBLAS promotion, or #1877's two-production-consumer completion.

## Accepted run

qz Job `i1877-rankk-h100-1010x` exited 0 at 2026-10-10 05:48:19
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

[qualification-summary.csv](qualification-summary.csv) retains all PASS rows
with timing, semantic work and resource counts. Endpoint time is 20 iterations
after four warmups and excludes transfers/preparation. Generated measured
13.9232--20.8352 us; cuBLAS measured 30.7024--41.9648 us and was slower in all
eight pairs.

## Identities and raw receipts

All 1,580 source rows passed. GPU regeneration from the actual host compiler,
CUDA headers, complete fixed device-tool closure and empty override environment
matched the staged header byte-for-byte.

| Item | SHA-256 |
| --- | --- |
| source / generated manifests | `e03f066b1e045afe2d8b170d88cf8d87907e0712d1331453de34254692721762` / `26cb22a5d9fe85fe3b92f1a6dd04cae93644a3364dca89fcda65e03a1c8c2e3c` |
| raw JSONL / provenance | `24fcf58f0c95ae5706e3b74ad6bc8a1cb350ee84b2f2446c88e5e2649e02a50a` / `1b00a4786160e537e302cdf821509a0df4dfbc3d6c0bcfcfd08bfa90117f6ab4` |
| native header / harness | `b479cdfe6cf147adcefc03ad73de4976f01905331ec00fb75dac708ad1c79de9` / `00d21abc1e3417e37095eab86af921d6b1ccf778c7d85a4af2e8339cef940737` |
| generated header / object / binary | `d01355789aca5eb81e942324350eb0f5192e3448061634bef3899f8729c0464d` / `1e1dda32d5e5b87a9aa93610fb03a94ae5ef0413a08b3cadd133000d59e17ca6` / `f8ab4f5261571636ab24ac370aff57ad1a24ab83b1849c8086183d627c6113e2` |
| sccache before / after | `5bad79e9df3dfc23bee25fcecb1877afc5ddbfb581694ec044456bb16f065b81` / `eb8fb5859a7d0738b895fc91ca92cfcbea03ce369999b7a5a7b0466838130506` |
| host compiler / Python | `d7122fd9a7a8fe12d12c00c54d3a6fbebcb3e9285cf675709674e751d900fc63` / `6ce8f488520d0f0cb8ccc405ba844279492c3926e18b73f0539e5fecd8033467` |

The hashed CUDA closure includes `nvcc.profile`, every observed device child,
libdevice, link stub, device runtime, cuBLAS and cudart. Exact hashes are in raw
`provenance.txt`; `nvcc.profile` is `4a5b882b5c16a8912ef84275d860f5b371164f3b354dad830f04fd6f128dbe2e`.

Raw Job `x` is at qz
`/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1877-rankk-1010/results/x`;
earlier negative/superseded jobs remain under that task root. A reviewer must
retrieve Job `x` raw before claiming independent raw-receipt review.

`sccache 0.16.0` wrapped compile/link: one CUBIN hit, three misses, four
compilations, one link pass-through, and zero errors/unsupported calls.
Receipt-only commits may reuse Job `x` only while the qualified source blobs and
modes remain identical and latest-head review verifies that boundary.
