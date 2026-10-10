# #1877 rank-k standalone CUDA qualification

Qualified source commit: `1cd3a783d51ce173d8dc06ad76965b84ef069b86`.
This is not a complete SCF/SCC
endpoint, cuBLAS promotion, or #1877's two-production-consumer completion.

## Accepted run

qz Job `i1877-rankk-h100-1010z12` exited 0 at 2026-10-10 10:04:35
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
measured 13.0576--21.6400 us; cuBLAS measured 25.0624--80.9296 us and was slower
in all eight pairs.

## Identities and raw receipts

All 1,588 source rows passed. GPU regeneration from the actual host compiler,
CUDA headers, complete fixed device-tool closure and empty override environment
matched the staged header byte-for-byte.

| Item | SHA-256 |
| --- | --- |
| source / generated manifests | `a57d5185aac387054f655443b9252b757f39f367d6f727781b3f55dd14c60764` / `00b56bbf28d1df126c8ba6d2209082cc04a6cbaf314b1cbb783120be6881c897` |
| raw JSONL / provenance | `8416da6325bf318008abc20ba3e94845b8cecbc548e94606949db8a17f4585bb` / `3a75c4ac08dd4c5514c2df722f596b078b63cb331e1359d871298afdb7a2c2` |
| native header / harness | `87228fbe1b8bb53aaa84d2cdcb351797f61c35f670b847d76ce040a6816111fe` / `fce0f82637e76a881c29740eb13305dd770260cf3be7dfec73ca7101db98fa8f` |
| generated header / object / binary | `4c522bac9d4cfe64683e2b89eda778bb7196174789a8bb8691ae7c45c26c400a` / `c30c4bfa2edd9cc4a7c885c1ce814966e29b51173c9ed6fca9ea1939d2da494c` / `49bd3306cc1f6363e1ce45746e6956077457c46c9e71b4da3d4331a710637c01` |
| sccache before / after | `5bad79e9df3dfc23bee25fcecb1877afc5ddbfb581694ec044456bb16f065b81` / `dee95fc03c86ddaae7aefd4e480b9ef4c4e0b759f8d0c164f05e44d45c184ec8` |
| host compiler / Python | `d7122fd9a7a8fe12d12c00c54d3a6fbebcb3e9285cf675709674e751d900fc63` / `6ce8f488520d0f0cb8ccc405ba844279492c3926e18b73f0539e5fecd8033467` |

The CUDA closure includes `nvcc.profile`, device children, libdevice, link stub
and libraries. The staged/actual 3,672-role GCC closure additionally binds the
driver-reported `liblto_plugin.so`, `lto-wrapper` and their dynamic dependencies;
its manifest hashes to
`3854e211cef932df50c38f5ed49d197a6041339589186b66cd3901da75cb3186`.

Raw Job `z12`: qz
`/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1877-rankk-1010/results/z12`.
Independent raw-receipt review requires retrieving it.

`sccache 0.16.0` wrapped compile/link: one CUBIN hit, three misses, four
compilations, one link pass-through, and zero errors/unsupported calls.
Receipt-only commits may reuse Job `z12` only while the qualified source blobs and
modes remain identical and latest-head review verifies that boundary.
