# #1877 rank-k standalone CUDA qualification

This qualifies a prepared rank-k capability at source commit
`66208b3e00aa190ad0fc3ac0373b006a9a3bdd52`. It is not a complete SCF/SCC
endpoint, cuBLAS promotion, or #1877's two-production-consumer completion.

## Accepted run

qz Job `i1877-rankk-h100-1010q` exited 0 on 2026-10-10 05:00:33
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

Generated measured 12.9696--21.5328 us; cuBLAS measured 24.2256--34.6304 us
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
| source / generated manifests | `06ba2833045f1eafb9ef964b1689af41b51077605dc753bdc00153e7e86e5e8b` / `93634f1177f38fb2fe4dbd0c713025023d82b9795429a9aa2b614d26a98af1d9` |
| final raw JSONL | `c274f7d08a9e5f5bd66aaf62384fccbd6b7d523383fa6030855014d977bf5bf0` |
| native header / harness | `eb53ea076da52c0b6f20fb27c9cccbd510238e439afe0b91f64d45e609ebf611` / `c60b0b5e0f2b3e526b9ff8fad306dcec7408286d2f218a1556c62fd3bdd93e26` |
| generated header / object / binary | `4c0c774893769512ce5bd840390a2d11d1f7972429312c65ca50e04cce49ba90` / `decec886727bf5e36c9359c09b60409efd3a0e89f1caa5ded860c850d6aecb7c` / `b3172871bbb746dfa1ada20965ba0bedde2d35866b3b106b8fcc75d067ac7606` |
| sccache before / after | `5bad79e9df3dfc23bee25fcecb1877afc5ddbfb581694ec044456bb16f065b81` / `f1aa9c47e03245fd0d83ead73fac642f380fb54e4a84468dcf784a091b2ca05c` |
| nvcc / ptxas | `df9974db233a0b7a6c6d59c0e5d74e011566104098bcb3553495ff1b95bdeaf6` / `983b0e9283855979f42cebfd80d43f9b6e786eb84f03f7570bf941c4d3a3c461` |
| cuBLAS / cuBLASLt / cudart | `5757ab5839fb4f203ca47ecb336110d10f4a5606b1e097f195fbca89774569e2` / `2c9006a75c74b3bea2dc7ae2ec38ab038b0e45ea02cb4b717a915e8a5796acb1` / `256e6409e4f06f618e1fb53d4844a6b81cdded1013afa8ade40c22f99eb133b7` |
| host compiler / Python | `d7122fd9a7a8fe12d12c00c54d3a6fbebcb3e9285cf675709674e751d900fc63` / `6ce8f488520d0f0cb8ccc405ba844279492c3926e18b73f0539e5fecd8033467` |

Full raw receipts remain at qz
`/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1877-rankk-1010/results/{b,c,d,e,f,g,h,i,j,k,m,n,o,q}`;
earlier raw is also in Git history at `9ec7fc52e408c062802db6e68de0f31eca7eff1f`.
The source-matched `q` raw is qz-only; a reviewer must retrieve it before
claiming independent raw-receipt review.

## Retained run boundaries

- `a/b` lacked Git/Python; `c` exposed `CUDART_NAN`; no compile in `a/b`.
- `d` passed on unsupported CUDA 12.8; `e` caught a wrong expected exception;
  `f/g/h` precede final precision/lint/format source.
- `i` had a wrong generated-manifest root; `j/k` qualified superseded identities.
- `l` proved portable Python/host bytes; `m` rejected ambient `LIBRARY_PATH`;
  `n/o` predate the cache-version/full-update identity fixes; `q` is final.

`sccache 0.16.0` wrapped compile and link: zero hits, four misses, five
compilations, one link pass-through, zero errors or unsupported calls.

## Source boundary

Job `q` compiled the exact compiler/native/harness/tool inputs at `66208b3e0`.
Later receipt-only commits may reuse it only while those blobs and modes remain
identical and latest-head review verifies that boundary.
