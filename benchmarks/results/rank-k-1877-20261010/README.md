# #1877 rank-k standalone CUDA qualification

Qualified source commit: `c2ab6a57ed43cb7bd8f7a877fe6f4e10926ef062`.
This is not a complete SCF/SCC
endpoint, cuBLAS promotion, or #1877's two-production-consumer completion.

## Accepted run

qz Job `i1877-rankk-h100-1010z9` exited 0 at 2026-10-10 07:58:27
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
measured 13.0224--21.3216 us; cuBLAS measured 25.9920--35.9440 us and was slower
in all eight pairs.

## Identities and raw receipts

All 1,581 source rows passed. GPU regeneration from the actual host compiler,
CUDA headers, complete fixed device-tool closure and empty override environment
matched the staged header byte-for-byte.

| Item | SHA-256 |
| --- | --- |
| source / generated manifests | `f38924d127eec174d378a6ddcceec758df50e25862b6112bd416fe7e2e318c54` / `8f13311c4a96cc47b66824fb625cd138720c3faa15f3da5e3130c4938d24e88b` |
| raw JSONL / provenance | `042485a3cde6da0f1bdd30b6567cd010f7e84d65c7bc24114187d5ba0430f6c7` / `4137c0608416244d939ea5c20ca81cdd4dca9cd6d5548141b39014f5ee8fd0b3` |
| native header / harness | `87228fbe1b8bb53aaa84d2cdcb351797f61c35f670b847d76ce040a6816111fe` / `fce0f82637e76a881c29740eb13305dd770260cf3be7dfec73ca7101db98fa8f` |
| generated header / object / binary | `7ac57e58a3e0ff789d3373d2269a9d987a71f8b1e601c182b5553ed7e8aa7c7d` / `2559e3eb7dc9fb14a449c980badd6095c47afe44ae674b0c33f22d1127af0717` / `343d93cf9d64a92395dbdd986440b1f14af07d829e24c344fb6c04abbf8214d4` |
| sccache before / after | `5bad79e9df3dfc23bee25fcecb1877afc5ddbfb581694ec044456bb16f065b81` / `33ef98439a85954f180d5bf6c9134035ffcddc041cadd128e3a5d660e0ca0884` |
| host compiler / Python | `d7122fd9a7a8fe12d12c00c54d3a6fbebcb3e9285cf675709674e751d900fc63` / `6ce8f488520d0f0cb8ccc405ba844279492c3926e18b73f0539e5fecd8033467` |

The CUDA closure includes `nvcc.profile`, device children, libdevice, link stub
and libraries. The staged/actual 3,666-role GCC closure manifest hashes to
`9092a8166628cbd431a48f50fec09fb8484f273dc18fafcf046764a44b0d652c`.

Raw Job `z9`: qz
`/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1877-rankk-1010/results/z9`.
Independent raw-receipt review requires retrieving it.

`sccache 0.16.0` wrapped compile/link: one CUBIN hit, three misses, four
compilations, one link pass-through, and zero errors/unsupported calls.
Receipt-only commits may reuse Job `z9` only while the qualified source blobs and
modes remain identical and latest-head review verifies that boundary.
