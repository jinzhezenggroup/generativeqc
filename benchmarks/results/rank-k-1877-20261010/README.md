# #1877 rank-k standalone CUDA qualification

Qualified source commit: `40c5773533e3cde3763193017332295b1911f99a`.
This is not a complete SCF/SCC
endpoint, cuBLAS promotion, or #1877's two-production-consumer completion.

## Accepted run

qz Job `i1877-rankk-h100-1010z4` exited 0 at 2026-10-10 06:55:39
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
13.0224--21.1984 us; cuBLAS measured 24.2112--33.8400 us and was slower in all
eight pairs.

## Identities and raw receipts

All 1,581 source rows passed. GPU regeneration from the actual host compiler,
CUDA headers, complete fixed device-tool closure and empty override environment
matched the staged header byte-for-byte.

| Item | SHA-256 |
| --- | --- |
| source / generated manifests | `332c97f280d2531bf8664d9f7436b051eb554d955441965a9822d28e77264db8` / `86513100b3da59791dc56ebe89b10d178b90be6dabbb4c3a94f21557deb80566` |
| raw JSONL / provenance | `14cd643470f1f62e75272eacc0ba3557b0492fc76618e6c25cbfae7ec60348f3` / `ed85b373aedb366085daf032edea84d40a340f0e8fc836bb8fdf53fde49221a3` |
| native header / harness | `542bb6623bd20b03a922fd98481610e9ef47cad4bcd8d4458d22383e80b3f11b` / `badefcb3131e899840aef32985d4037e32715baeb2a1307bd2936624f2f361c3` |
| generated header / object / binary | `aee5813be4949e57dc9547ff69948c28d40359895e2c29cc4f43c4a2689acf98` / `2d4896f6f2c4dad65553d60448733c21ea5a2084ba255a7661a67c6914627591` / `5d8e8bb8e72d5029207db7d40ff562dd9a67aa5d74eea7b95ef18f583fcdd030` |
| sccache before / after | `5bad79e9df3dfc23bee25fcecb1877afc5ddbfb581694ec044456bb16f065b81` / `513a8ae21411f6cc037e320283125ea48155865019139a514e63916b298dc754` |
| host compiler / Python | `d7122fd9a7a8fe12d12c00c54d3a6fbebcb3e9285cf675709674e751d900fc63` / `6ce8f488520d0f0cb8ccc405ba844279492c3926e18b73f0539e5fecd8033467` |

The CUDA closure includes `nvcc.profile`, device children, libdevice, link stub
and libraries. The staged/actual 3,666-role GCC closure manifest hashes to
`9092a8166628cbd431a48f50fec09fb8484f273dc18fafcf046764a44b0d652c`.

Raw Job `z4`: qz
`/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1877-rankk-1010/results/z4`.
Independent raw-receipt review requires retrieving it.

`sccache 0.16.0` wrapped compile/link: one CUBIN hit, three misses, four
compilations, one link pass-through, and zero errors/unsupported calls.
Receipt-only commits may reuse Job `z4` only while the qualified source blobs and
modes remain identical and latest-head review verifies that boundary.
