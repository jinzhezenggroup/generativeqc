# #1877 rank-k standalone CUDA qualification

Qualified source commit: `d8e5eaa9af430a6509ae635e50c6a736a1127817`.
This is not a complete SCF/SCC
endpoint, cuBLAS promotion, or #1877's two-production-consumer completion.

## Accepted run

qz Job `i1877-rankk-h100-1010z11` exited 0 at 2026-10-10 09:07:43
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
measured 13.2624--21.4192 us; cuBLAS measured 25.5056--36.3552 us and was slower
in all eight pairs.

## Identities and raw receipts

All 1,588 source rows passed. GPU regeneration from the actual host compiler,
CUDA headers, complete fixed device-tool closure and empty override environment
matched the staged header byte-for-byte.

| Item | SHA-256 |
| --- | --- |
| source / generated manifests | `4ab192193f9addabe581445bff2df137dcead8bc9e353cabaf185922e6b71602` / `9de1669ac295a015970a904d2696e322e7647a486867f6744d634ce802930c03` |
| raw JSONL / provenance | `ac5a0e968a6c3337c49b5549f6ce6717a9a5dca4dc1fc1e55abd388a0ff1eb81` / `30a77afc77b512b7ad551a33f3153446710e26f458d71fd1e7628733405a47b1` |
| native header / harness | `87228fbe1b8bb53aaa84d2cdcb351797f61c35f670b847d76ce040a6816111fe` / `fce0f82637e76a881c29740eb13305dd770260cf3be7dfec73ca7101db98fa8f` |
| generated header / object / binary | `4d1c3a749538377b7e6d04e1ae2f96c5f76f8c9df297bff672fa203d8ba381b8` / `329e28dd2e93c5784b27d8e78901779d666da8f00212b905a98657eac2ddd7f4` / `783ac03a3a31cc381c556041de115e6bb197638aa7ffa17e544d2244ee00c2ef` |
| sccache before / after | `5bad79e9df3dfc23bee25fcecb1877afc5ddbfb581694ec044456bb16f065b81` / `e3e46057629d8d00b4e262a0173bbc56845ff25dfc691fea7cb5a4c1dc2bf77c` |
| host compiler / Python | `d7122fd9a7a8fe12d12c00c54d3a6fbebcb3e9285cf675709674e751d900fc63` / `6ce8f488520d0f0cb8ccc405ba844279492c3926e18b73f0539e5fecd8033467` |

The CUDA closure includes `nvcc.profile`, device children, libdevice, link stub
and libraries. The staged/actual 3,672-role GCC closure additionally binds the
driver-reported `liblto_plugin.so`, `lto-wrapper` and their dynamic dependencies;
its manifest hashes to
`3854e211cef932df50c38f5ed49d197a6041339589186b66cd3901da75cb3186`.

Raw Job `z11`: qz
`/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1877-rankk-1010/results/z11`.
Independent raw-receipt review requires retrieving it.

`sccache 0.16.0` wrapped compile/link: one CUBIN hit, three misses, four
compilations, one link pass-through, and zero errors/unsupported calls.
Receipt-only commits may reuse Job `z11` only while the qualified source blobs and
modes remain identical and latest-head review verifies that boundary.
