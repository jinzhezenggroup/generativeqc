# #1877 rank-k standalone CUDA qualification

This qualifies the prepared capability at source commit
`cf41294a1b6408e20b6ed721cb402e48bd5b7442`; it is not a complete SCF/SCC
endpoint, cuBLAS promotion, or #1877's two-production-consumer completion.

## Accepted run

qz Job `i1877-rankk-h100-1010v` exited 0 at 2026-10-10 05:32:00
Asia/Shanghai on H100 80GB HBM3 (sm90), driver 570.124.06, CUDA 12.9.86,
runtime 12090, cuBLAS 120902 and g++ 11.4.0. All 16 timed cases passed:
density/weighted density, row/column order, `3x5`/`17x9`, two batches and both
providers. A CPU `long double` oracle covers signed/negative-zero weights,
`alpha=1.25`, `beta=-0.5`, odd tails, asymmetric old lower data and two graph
replays. Untimed gates cover separate overwrite/update TensorIR roots,
`beta=+0/-0` without reading NaN old output, signed-zero alpha, alias/order/
resource rejection, overflow and nonfinite all-output preservation, and
mismatched overwrite/update request rejection.

[qualification-summary.csv](qualification-summary.csv) retains all PASS rows
with timing, semantic work and resource counts. Endpoint time is 20 iterations
after four warmups and excludes transfers/preparation. Generated measured
12.7792--21.4176 us; cuBLAS measured 25.6736--36.4192 us and was slower in all
eight pairs. The CSV retains the distinct work and resource counts.

## Identities and raw receipts

All 1,576 source rows passed. GPU regeneration from the actual host compiler,
CUDA headers, complete fixed device-tool closure and empty override environment
matched the staged header byte-for-byte.

| Item | SHA-256 |
| --- | --- |
| source / generated manifests | `326617e92857cd429940d9c4fbc41b51dee21dde81aef0bb995ab76ea32f44d4` / `4fefaeed87e5c3cd41749e3b20add801d166a7cd99ee3fa49a9b08f3733cbe4d` |
| raw JSONL / provenance | `f402c026de199da9dc7b1eabb855c188736fd90cccfdf3ecc6d9843317645d52` / `e4df35971bb738288f81ddd62ef2cb21f1b2ba67b4962550a1e2033a28d34b6b` |
| native header / harness | `e1edce4b9caf6cc0777bf7206894410ed2d5201f4a1ec9f6bac2dd5d5c26a287` / `1c820e7e32bd10042d5d1f96bff1bfb1b99f39c7d541d62aa1f847be66755e23` |
| generated header / object / binary | `b2e35e9d670acefd8a2fd894f95393b6275f5926886ef3fb633e48573bcab358` / `06864bed4c9335280a04898d8379e335d518331ef0d6cf83d7f73ebd1bac27cf` / `66874b161ba704ec06498f18da06b7164b3e9766d4f9b502f48907f1de35fab6` |
| sccache before / after | `5bad79e9df3dfc23bee25fcecb1877afc5ddbfb581694ec044456bb16f065b81` / `20bf03ef024eb5063a37dc12fc785689d0835489af68527c32d03564e0ca8afb` |
| host compiler / Python | `d7122fd9a7a8fe12d12c00c54d3a6fbebcb3e9285cf675709674e751d900fc63` / `6ce8f488520d0f0cb8ccc405ba844279492c3926e18b73f0539e5fecd8033467` |

The hashed CUDA closure is `nvcc`, `cudafe++`, `fatbinary`, `nvlink`, `ptxas`,
`cicc`, `link.stub`, `libdevice.10.bc`, `libcudadevrt.a`, cuBLAS, cuBLASLt and
cudart. Exact per-file hashes are in raw `provenance.txt`; the `cicc` hash is
`3580897800493d133b192f3873ea85e0e3e616b2da3a2e163752f8c3491bd9ef`.

Raw Job `v` receipts are at qz
`/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1877-rankk-1010/results/v`.
Earlier negative/superseded receipts remain under the same task root; pre-`j`
raw is also recoverable from Git commit `9ec7fc52e408c062802db6e68de0f31eca7eff1f`.
Job `s` was the last complete run before overwrite/update splitting and Job `u`
was a dry-run-only tool-discovery probe; neither qualifies this head. A reviewer
must retrieve Job `v` raw before claiming independent raw-receipt review.

`sccache 0.16.0` wrapped compile/link: one CUBIN hit, three misses, four
compilations, one link pass-through, and zero errors/unsupported calls.
Receipt-only commits may reuse Job `v` only while the qualified source blobs and
modes remain identical and latest-head review verifies that boundary.
