# #1877 rank-k standalone CUDA qualification

This is a compact record of the qualification at source commit
`f0c02130c179ca72e8d06d459ed0f695897ee2f0`. It is not a complete SCF/SCC
endpoint result, does not promote cuBLAS, and does not satisfy #1877's remaining
two-production-consumer gate.

## Accepted run

qz Job `i1877-rankk-h100-1010k` exited 0 on 2026-10-10 04:08:32
Asia/Shanghai: H100 80GB HBM3 (sm90), driver 570.124.06, CUDA 12.9.86,
runtime 12090 and cuBLAS 120902. It passed 16/16 combinations of density versus
energy-weighted density, row/column physical order, `3x5`/`17x9` panels, two
batches, and generated versus cuBLAS execution. The independent `long double`
oracle covered signed and negative-zero weights, `alpha=1.25`, `beta=-0.5`, odd
tails, asymmetric old lower-triangle data, two graph replays, alias/order/
overflow/scalar rejection, zero-budget generated fallback, and nonfinite
all-output preservation.

[qualification-summary.csv](qualification-summary.csv) retains every accepted
case. Columns are weighted flag, order, provider, `n`, `k`, endpoint and prepare
microseconds, logical/executed products, scale and weight-materialization
elements, validation and mirror elements, output-reset bytes, temporary bytes,
and retained provider bytes. Each row is `PASS`; batches are 2. Endpoint timing
includes reset, weight materialization, scaling, reduction/GEMM, validation,
mirroring and launches after four warmups over 20 repetitions. It excludes
host/device input transfers and preparation.

Generated measured 12.9520--21.5408 us; cuBLAS measured 23.8912--33.6896 us.
cuBLAS was slower in every tested pair, so generated remains the default. At
`n=17,k=9`, generated executed 5,508 products/scales; cuBLAS executed 5,202
products plus 306 scales. Library scratch was 384 or 7,072 bytes with a 96 MiB
allowance and 64--66 MiB retained; generated retained no provider resource.

## Identities and retained raw data

The clean 1,367-file source manifest and generated header passed GPU-side
integrity checks. SHA-256 is an integrity receipt, not source authentication.

| Item | SHA-256 |
| --- | --- |
| source manifest | `3ec5b3c7bf2a49f19cfd3ef9b18362642d81054b27f25824bc0ea0044560d4bd` |
| generated manifest | `f02e4613e479d52f663250325e8c5ea6fdbd50a5b905391b60fc78c75aabe211` |
| final raw JSONL | `7dc052c9ff52468c6e37eb40162dced68da17c311ed25f2839ff3880caa0525f` |
| native header / harness | `172825cafc24fa7e9079fe3faebb715ca391a3a132a671844f96224288435517` / `c60b0b5e0f2b3e526b9ff8fad306dcec7408286d2f218a1556c62fd3bdd93e26` |
| generated header | `bad7d7090b7a65a5351236cf027d542bc81910f30c8ea13400a6969efa2c37de` |
| object | `8a3ae3453bbe303a3c4481431a2fd22033feb9723a05ee05b2ab043a0cbe5797` |
| binary | `3da08c046584c615fa39a75fd67b4a7cf2840592c19b09d3579964a6fad2de0e` |
| sccache before/after | `5bad79e9df3dfc23bee25fcecb1877afc5ddbfb581694ec044456bb16f065b81` / `302d8f97227a9a08dceb789df03a19c39e93a45a99057d37110a03203e9f9441` |
| nvcc / ptxas | `df9974db233a0b7a6c6d59c0e5d74e011566104098bcb3553495ff1b95bdeaf6` / `983b0e9283855979f42cebfd80d43f9b6e786eb84f03f7570bf941c4d3a3c461` |
| cuBLAS / cuBLASLt / cudart | `5757ab5839fb4f203ca47ecb336110d10f4a5606b1e097f195fbca89774569e2` / `2c9006a75c74b3bea2dc7ae2ec38ab038b0e45ea02cb4b717a915e8a5796acb1` / `256e6409e4f06f618e1fb53d4844a6b81cdded1013afa8ade40c22f99eb133b7` |

Full raw receipts remain at qz path
`/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1877-rankk-1010/results/{b,c,d,e,f,g,h,i,j,k}`.
Earlier raw receipts are also recoverable from Git history at commit
`9ec7fc52e408c062802db6e68de0f31eca7eff1f`; the source-matched `k` receipts are
qz-only. A reviewer must retrieve and verify those bytes independently before
claiming independent raw-receipt review.

## Negative and preliminary jobs

- `a` failed before compile: image lacked Git; `b` failed before compile: no
  Python; `c` reached compile and exposed undefined `CUDART_NAN`.
- `d` passed 16 cases with CUDA 12.8.61, below the supported 12.9 floor; `e`
  compiled/linked on 12.9 but the harness expected the wrong overflow exception.
- `f` passed before strict weight-stage precision/negative-zero additions; `g`
  passed before final Python lint/format changes; `h` qualified `41f920b` before
  the pre-commit bot reformatted the native source and harness.
- `i` failed before compile because CPU staging wrote the generated manifest
  relative to the task root rather than the verifier's repo root. `j` qualified
  `c513dccb` before the compiler-identity and semantic-review fixes; `k` is the
  supported-toolchain, source-matched qualification for `f0c02130`.

The `sccache 0.16.0` object compile recorded one request, one CUBIN hit, three
CUDA/PTX misses, four compilations, zero cache errors and zero unsupported calls.

## Source boundary

Job `k` compiled the exact compiler, native header and harness blobs at `f0c02130`.
Receipt-only commits after that source commit are not themselves device runs;
they can reuse `k` only while every qualified implementation blob remains
identical and the latest-head review checks that boundary explicitly.
