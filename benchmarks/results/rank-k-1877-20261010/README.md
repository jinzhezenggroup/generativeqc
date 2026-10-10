# #1877 rank-k standalone CUDA qualification

Qualified source commit: `7ff494c7ecfb9a538f9ea0c77ace788e66c7d1b3`.
This is not a complete SCF/SCC
endpoint, cuBLAS promotion, or #1877's two-production-consumer completion.

## Accepted run

qz Job `i1877-rankk-h100-1010z13` exited 0 at 2026-10-10 15:12:38
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
measured 12.5888--21.2400 us; cuBLAS measured 25.5984--35.8096 us and was slower
in all eight pairs.

## Identities and raw receipts

All 1,590 source rows passed. GPU regeneration from the actual host compiler,
CUDA headers, complete fixed device-tool closure and empty override environment
matched the staged header byte-for-byte.

| Item | SHA-256 |
| --- | --- |
| source / generated manifests | `ddd5a2958d9634879962395c2d659baac480bba92be9f48875580d73fc242f05` / `12c97b85bc46f23ced34be8fbfd5a8b9bd059b031abc675bdb8b8ca8c8d7823b` |
| raw JSONL / provenance | `5648a34b2bcdbfeb5cdaa6712e088a54db52246f0b62525c49cda6efe4192257` / `d597be2d9ac9b26378dc14bd6c8e0242395baf833e3e6ab9be404cc2289d9af5` |
| source/artifact hash receipt | `872f44dac7ca21065b7fa9724fd6a57f6bca39340f46cb6e4d119d3aca0680ef` |
| native header / harness | `87228fbe1b8bb53aaa84d2cdcb351797f61c35f670b847d76ce040a6816111fe` / `fce0f82637e76a881c29740eb13305dd770260cf3be7dfec73ca7101db98fa8f` |
| generated header / object / binary | `9510a964cf8e25a95af53b938bba3da591419f1be531396ba60051313433087c` / `81f0403c5b93f1fd259b947798e1d97783aacf70417d45716273ab0ff1317bd9` / `389c445ffdae18221b0ca2a32e1af8fe31469467176054bd196baffaaac6e98c` |
| sccache before / after | `5bad79e9df3dfc23bee25fcecb1877afc5ddbfb581694ec044456bb16f065b81` / `1a1d5795a35d011d9f9aec118d7bb458c1c212f5feca2c2d688c0e126a0fea4f` |
| host compiler / Python | `d7122fd9a7a8fe12d12c00c54d3a6fbebcb3e9285cf675709674e751d900fc63` / `6ce8f488520d0f0cb8ccc405ba844279492c3926e18b73f0539e5fecd8033467` |

The CUDA closure includes `nvcc.profile`, device children, libdevice, link stub
and libraries. The staged/actual 3,672-role GCC closure additionally binds the
driver-reported `liblto_plugin.so`, `lto-wrapper` and their dynamic dependencies;
its manifest hashes to
`3854e211cef932df50c38f5ed49d197a6041339589186b66cd3901da75cb3186`.

Raw Job `z13`: qz
`/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1877-rankk-1010/results/z13`.
Independent raw-receipt review requires retrieving it.

`sccache 0.16.0` wrapped compile/link: one CUBIN hit, three misses, four
compilations, one link pass-through, and zero errors/unsupported calls.
Receipt-only commits may reuse Job `z13` only while the qualified source blobs and
modes remain identical and latest-head review verifies that boundary.
