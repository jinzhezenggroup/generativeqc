# #1877 rank-k standalone CUDA qualification

This is a compact record of the qualification at source commit
`41f920b512e25dc6da8c08130de9bbda15e71cbe`. It is not a complete SCF/SCC
endpoint result, does not promote cuBLAS, and does not satisfy #1877's remaining
two-production-consumer gate.

## Accepted run

qz Job `i1877-rankk-h100-1010h` exited 0 on 2026-10-10 03:37:14
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

Generated measured 12.7920--21.4976 us; cuBLAS measured 25.5552--50.9840 us.
cuBLAS was slower in every tested pair, so generated remains the default. At
`n=17,k=9`, generated executed 5,508 products/scales; cuBLAS executed 5,202
products plus 306 scales. Library scratch was 384 or 7,072 bytes with a 96 MiB
allowance and 64--66 MiB retained; generated retained no provider resource.

## Identities and retained raw data

The clean 1,367-file source manifest and generated header passed GPU-side
integrity checks. SHA-256 is an integrity receipt, not source authentication.

| Item | SHA-256 |
| --- | --- |
| source manifest | `7ffba5b2314afe369ff071edfb039e8bcf1c2805d91c8a7d68dda83d6a88b4e7` |
| generated manifest | `45a573663fdba315971cef491a310b5ecbc89e6cb95f813abf04e256434f2b79` |
| final raw JSONL | `50b2706e7f2bf6cc5a93bf929d19a9320d25f737d85aad04840a8651f9079c63` |
| qualified native header | `82d57ebb17ad61bcc5c40458d5981c6e6d63436538cf7a611ee02fc4ca4361da` |
| generated header | `937cf8d7b962b9df8a3f191470409205206791caf28c0fe0ff365969b1c3cbee` |
| object | `87ad3f1a40a094e5e7ac73062d18a8d9f7a73a985ae7c79ba77353a811179c2a` |
| binary | `74293a2a44326940c9000ea70af43ed819e8dff465605afc922b942dece5aa07` |
| sccache before/after | `1374c560520b9e4f65401adf1a44e0316fbec36a5d62e9cfd8c125ae0c82487d` / `9bfd13a445dca025aaca5772cb4e08d80f8c36b59816757f85b86047d4a2806a` |
| nvcc / ptxas | `df9974db233a0b7a6c6d59c0e5d74e011566104098bcb3553495ff1b95bdeaf6` / `983b0e9283855979f42cebfd80d43f9b6e786eb84f03f7570bf941c4d3a3c461` |
| cuBLAS / cuBLASLt / cudart | `5757ab5839fb4f203ca47ecb336110d10f4a5606b1e097f195fbca89774569e2` / `2c9006a75c74b3bea2dc7ae2ec38ab038b0e45ea02cb4b717a915e8a5796acb1` / `256e6409e4f06f618e1fb53d4844a6b81cdded1013afa8ade40c22f99eb133b7` |

Full raw receipts remain at qz path
`/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1877-rankk-1010/results/{b,c,d,e,f,g,h}`
and in Git history at commit `9ec7fc52e408c062802db6e68de0f31eca7eff1f` under this directory. A reviewer
must retrieve and verify those bytes independently before claiming independent
raw-receipt review.

## Negative and preliminary jobs

- `a` failed before compile: image lacked Git; `b` failed before compile: no
  Python; `c` reached compile and exposed undefined `CUDART_NAN`.
- `d` passed 16 cases with CUDA 12.8.61, below the supported 12.9 floor; `e`
  compiled/linked on 12.9 but the harness expected the wrong overflow exception.
- `f` passed before strict weight-stage precision/negative-zero additions; `g`
  passed before final Python lint/format changes; only `h` is the final
  supported-toolchain qualification for commit `41f920b`.

The `sccache 0.16.0` object compile recorded one request, one CUBIN hit, three
CUDA/PTX misses, four compilations, zero cache errors and zero unsupported calls.

## Latest-head boundary

Pre-commit bot commit `9ec7fc52e408c062802db6e68de0f31eca7eff1f` reformatted the native header and
harness (including include reordering), so Job `h` is not an exact-byte
qualification of that later head. Its device result remains attached only to
`41f920b`; any later source-matched qualification must carry its own hashes.
