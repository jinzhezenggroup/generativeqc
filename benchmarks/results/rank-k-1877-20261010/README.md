# #1877 rank-k standalone CUDA qualification

This qualifies a prepared rank-k capability at source commit
`64422e16e0089a4ee19a54e32791599d9857acb5`. It is not a complete SCF/SCC
endpoint, cuBLAS promotion, or #1877's two-production-consumer completion.

## Accepted run

qz Job `i1877-rankk-h100-1010o` exited 0 on 2026-10-10 04:37:30
Asia/Shanghai: H100 80GB HBM3 (sm90), driver 570.124.06, CUDA 12.9.86,
runtime 12090, cuBLAS 120902, and g++ 11.4.0. Its 16/16 cases span
density/weighted density, both physical orders, `3x5`/`17x9`, two batches and
generated/cuBLAS. The independent `long double` oracle covers signed and
negative-zero weights, `alpha=1.25`, `beta=-0.5`, odd tails, asymmetric old
lower data, two graph replays, alias/order/overflow/scalar/resource rejection,
and nonfinite all-output preservation.

[qualification-summary.csv](qualification-summary.csv) keeps every PASS row:
weighted flag, order, provider, `n/k`, endpoint/prepare microseconds, logical/
executed products, scale/weight/validation/mirror elements, reset bytes,
temporary bytes and retained provider bytes. Batches are 2; endpoint timing is
20 repetitions after four warmups and excludes transfers/preparation.

Generated measured 12.9920--20.7920 us; cuBLAS measured 23.6528--34.1040 us
and was slower in all eight pairs. At `17x9`, generated executes 5,508 products/
scales; cuBLAS executes 5,202 products plus 306 scales. Library scratch is 384
or 7,072 bytes with a 96 MiB allowance and 64--66 MiB retained; generated
retains no provider resource.

## Identities and raw data

All 1,576 source-inventory rows passed. GPU regeneration from the actual host
compiler, CUDA headers/toolchain and fixed empty override environment matched
the staged header byte-for-byte. SHA-256 is integrity, not authentication.

| Item | SHA-256 |
| --- | --- |
| source / generated manifests | `335ec009b7ce79eb2de10caadcdafd16b90fc7642576dfc373a7ff59c9fe05ec` / `8501a30cf02dd070d22e2f32420acd34c5bb41f8902356439d7e3a5547c6fd00` |
| final raw JSONL | `afc58e141420a944551a5dae1413b672c47d8c28153aae7d6725c4b0e29fd516` |
| native header / harness | `172825cafc24fa7e9079fe3faebb715ca391a3a132a671844f96224288435517` / `c60b0b5e0f2b3e526b9ff8fad306dcec7408286d2f218a1556c62fd3bdd93e26` |
| generated header / object / binary | `ac9f4a173bcf8d2cb0c28333abe1561d633a197caf0569c7f364ca4aa0db2f79` / `4ee3ff165c029e481123866b50d0cf7ce965bb47b59fc3e1d82f5d6204aeb41b` / `4db4d013f0eb8a6d02f4b5c5c3ec385769137d04214a11989b59b0db013d18` |
| sccache before / after | `5bad79e9df3dfc23bee25fcecb1877afc5ddbfb581694ec044456bb16f065b81` / `81909b441b65fc162f3c52cadb9df547488b943fd8b6f2fd82f6ae90cab59626` |
| nvcc / ptxas | `df9974db233a0b7a6c6d59c0e5d74e011566104098bcb3553495ff1b95bdeaf6` / `983b0e9283855979f42cebfd80d43f9b6e786eb84f03f7570bf941c4d3a3c461` |
| cuBLAS / cuBLASLt / cudart | `5757ab5839fb4f203ca47ecb336110d10f4a5606b1e097f195fbca89774569e2` / `2c9006a75c74b3bea2dc7ae2ec38ab038b0e45ea02cb4b717a915e8a5796acb1` / `256e6409e4f06f618e1fb53d4844a6b81cdded1013afa8ade40c22f99eb133b7` |
| host compiler / Python | `d7122fd9a7a8fe12d12c00c54d3a6fbebcb3e9285cf675709674e751d900fc63` / `6ce8f488520d0f0cb8ccc405ba844279492c3926e18b73f0539e5fecd8033467` |

Full raw receipts remain at qz
`/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1877-rankk-1010/results/{b,c,d,e,f,g,h,i,j,k,m,n,o}`;
earlier raw is also in Git history at `9ec7fc52e408c062802db6e68de0f31eca7eff1f`.
The source-matched `o` raw is qz-only; a reviewer must retrieve it before
claiming independent raw-receipt review.

## Retained run boundaries

- `a/b` lacked Git/Python; `c` exposed `CUDART_NAN`; no compile in `a/b`.
- `d` passed on unsupported CUDA 12.8; `e` caught a wrong expected exception;
  `f/g/h` precede final precision/lint/format source.
- `i` had a wrong generated-manifest root; `j/k` qualified superseded identities.
- `l` proved portable Python/host bytes; `m` rejected ambient `LIBRARY_PATH`;
  `n` predates the numeric cache-version fix; `o` is final.

`sccache 0.16.0` wrapped compile and link: one CUBIN hit, three misses, four
compilations, one link pass-through, zero errors or unsupported calls.

## Source boundary

Job `o` compiled the exact compiler/native/harness/tool inputs at `64422e16e`.
Later receipt-only commits may reuse it only while those blobs and modes remain
identical and latest-head review verifies that boundary.
