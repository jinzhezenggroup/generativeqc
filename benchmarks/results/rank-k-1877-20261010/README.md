# #1877 rank-k standalone CUDA qualification

This qualifies a prepared rank-k capability at source commit
`c6eebd3555754157dc8ad353243460d6c94f4c68`. It is not a complete SCF/SCC
endpoint, cuBLAS promotion, or #1877's two-production-consumer completion.

## Accepted run

qz Job `i1877-rankk-h100-1010r` exited 0 on 2026-10-10 05:09:23
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

Generated measured 12.6288--21.1200 us; cuBLAS measured 23.5952--34.3520 us
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
| source / generated manifests | `6ae6f0d08cb43f68ee206e5e35b0c3d6291ff47d0f40a411fdba72d1228aa881` / `3c67b6be1a771cb28ad2234f72fe8fc88e38bdf573aac8f6134a65928faf99e8` |
| final raw JSONL | `24c4ddea54c3b1966d3a0923763fb0729b22410416f2138668847952d49d2144` |
| native header / harness | `d0c6cf440817213a40f7cb4acaef5a1db3a44f571e7149da4956f89dacafa1c4` / `c60b0b5e0f2b3e526b9ff8fad306dcec7408286d2f218a1556c62fd3bdd93e26` |
| generated header / object / binary | `62faeea2cca28ca98dd5fdf8d6a1605dddefd6bb6a929826bec192d068b9d9b0` / `e3e1794791d8ea97926e19b542dc25a4e103a21a529601233d27e2c6cc6f25ef` / `02137d3d7579fcb0b39bf48b060cb08eb89547bfe4a95a2dde7e20eafb674cbb` |
| sccache before / after | `5bad79e9df3dfc23bee25fcecb1877afc5ddbfb581694ec044456bb16f065b81` / `8c6e5ddab3a8e21b0791e3b91fe449cef23277038e690f6dabe46da40316d89e` |
| nvcc / ptxas | `df9974db233a0b7a6c6d59c0e5d74e011566104098bcb3553495ff1b95bdeaf6` / `983b0e9283855979f42cebfd80d43f9b6e786eb84f03f7570bf941c4d3a3c461` |
| cuBLAS / cuBLASLt / cudart | `5757ab5839fb4f203ca47ecb336110d10f4a5606b1e097f195fbca89774569e2` / `2c9006a75c74b3bea2dc7ae2ec38ab038b0e45ea02cb4b717a915e8a5796acb1` / `256e6409e4f06f618e1fb53d4844a6b81cdded1013afa8ade40c22f99eb133b7` |
| host compiler / Python | `d7122fd9a7a8fe12d12c00c54d3a6fbebcb3e9285cf675709674e751d900fc63` / `6ce8f488520d0f0cb8ccc405ba844279492c3926e18b73f0539e5fecd8033467` |

Full raw receipts remain at qz
`/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1877-rankk-1010/results/{b,c,d,e,f,g,h,i,j,k,m,n,o,q,r}`;
earlier raw is also in Git history at `9ec7fc52e408c062802db6e68de0f31eca7eff1f`.
The source-matched `r` raw is qz-only; a reviewer must retrieve it before
claiming independent raw-receipt review.

## Retained run boundaries

- `a/b` lacked Git/Python; `c` exposed `CUDART_NAN`; no compile in `a/b`.
- `d` passed on unsupported CUDA 12.8; `e` caught a wrong expected exception;
  `f/g/h` precede final precision/lint/format source.
- `i` had a wrong generated-manifest root; `j/k` qualified superseded identities.
- `l` proved portable Python/host bytes; `m` rejected ambient `LIBRARY_PATH`;
  `n/o/q` predate the cache-version/update/precision identity fixes; `r` is final.

`sccache 0.16.0` wrapped compile and link: one CUBIN hit, three misses, four
compilations, one link pass-through, zero errors or unsupported calls.

## Source boundary

Job `r` compiled the exact compiler/native/harness/tool inputs at `c6eebd355`.
Later receipt-only commits may reuse it only while those blobs and modes remain
identical and latest-head review verifies that boundary.
