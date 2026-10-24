# Shared response qualification

Frozen-source receipts are not qualification of later trees. Both CUDA Lambda
residency bindings remain default-off; #932 is not complete. Current contracts:
[`response.md`](../../../../docs/developer/response.md).

- [Initial contract](qualification.json).
- [Conventional residency](qualification-resident-lambda.json.xz): `8e8cf9aab`.
- [Master integration](qualification-resident-integration.json.xz): `b9617de02`.
- [DF residency and allocation-fallback integration](qualification-resident-df.json.xz):
  `de67ead96`; actual complete endpoints, sanitizer and matched device traces.
- [Exact previous narrative](README-history.md.xz): lossless historical bytes,
  including earlier failures, source identities and incomplete acceptance gates.

Decode with `xz -dc <path>`. Raw logs, source archives and traces remain ignored
under `build/issue932-response-contract/resident/` and the remote
`/data/jzzeng/qc-932-expanded-20261011/evidence/`. Receipts retain missing prepared,
HF/CPKS/MP2, full memory, broad performance and finite cold-JIT gates explicitly.
