# Native automatic Libxc qualification

Measured source revision: `36085eec7d745ad0e25514468ddfa42aea2c9f86`.
Retained measured source inputs: `ec311fd6a9bc3a7105ab42ff9f7e154400119a91d79b3b6987e62c3e09cfc0b3`.
Current checked source inputs: `2f4a207a11eb2e7cbf05bc6e4c6d4fe98942e1ba1676000ab8d3e7e7bc2e6f78`.
Current checked native input identity: `0d0ba7427e1951632a716d1032e0344fdae4babc5744b39eb2b7eecca87f3569`.
Retained evidence stale for checked inputs: `True`.
Retained executed native library SHA-256: `d16bf758445db7fc0b1b043847979e7a12ef851ee98254be2e9a7f47dace0efa`.
Independent oracle: PySCF 2.14.0 / Libxc 7.0.0.
Structurally selectable non-curated automatic registrations: 197; registrations without a retained native campaign: 210.

This offline qualification inventory does not restrict structural public selection.
The automatic count excludes registrations already owned by curated components.
Counts apply only to the exact retained native contexts; no response, Hessian,
HVP, higher-l, mixed/Fast, ECP or DF scope is inferred.
The `public-method` row counts the optional stronger admission receipt,
not available public selectors. Its absence does not disable a method.

```text
Libxc capability summary (221 functionals)
stage                qualified  ready  blocked
graph-imported             221      0        0
pointwise-validated        221      0        0
compiled-cpu                 0    221       11
compiled-cuda                0    221       11
production-domain            0    213       11
gpu-runtime                  0      0       11
molecular-scf                0      0       11
forces                       0      0       11
response                     0      0        0
public-method                0      0        0
public DFT: 0/221
explicit blockers: 66 across 11 functionals
```

| Family | Backend | Spin | Product | Qualified registrations |
| --- | --- | --- | --- | ---: |

Current-source admitted backend/spin lifecycle records: 0.
Historical measurements remain in the retained campaign; stale receipts are not
counted as current-source admitted contexts. Admitted records retain exact
geometry, grid, precision, binary and derivative scope.
The [manifest](../../manifests/libxc/native-qualification.json) locates the
compressed machine-readable snapshot and independent numerical receipts.

## Qualified registrations

| Registration | Backend | Product | Qualified spin layouts |
| --- | --- | --- | --- |

## Explicit blockers

- `MGGA_X_EDMGGA`: unsupported-ingredients:laplacian
- `MGGA_X_GDME_0`: unsupported-ingredients:laplacian
- `MGGA_X_GDME_KOS`: unsupported-ingredients:laplacian
- `MGGA_X_GDME_NV`: unsupported-ingredients:laplacian
- `MGGA_X_GDME_VT`: unsupported-ingredients:laplacian
- `MGGA_X_JK`: unsupported-ingredients:laplacian
- `MGGA_X_MK00`: unsupported-ingredients:laplacian
- `MGGA_X_RLDA`: unsupported-ingredients:laplacian
- `GGA_C_APBE`: stale native/compiler/Graph/domain/ABI inputs; rerun qualification
- `GGA_X_APBE`: stale native/compiler/Graph/domain/ABI inputs; rerun qualification
- `GGA_X_B86`: stale native/compiler/Graph/domain/ABI inputs; rerun qualification
- `GGA_X_PBE_SOL`: stale native/compiler/Graph/domain/ABI inputs; rerun qualification
- `GGA_X_RPBE`: stale native/compiler/Graph/domain/ABI inputs; rerun qualification
- `LDA_C_CHACHIYO`: stale native/compiler/Graph/domain/ABI inputs; rerun qualification
- `LDA_C_PZ_MOD`: stale native/compiler/Graph/domain/ABI inputs; rerun qualification
- `LDA_C_VWN_4`: stale native/compiler/Graph/domain/ABI inputs; rerun qualification
- `MGGA_C_TPSS`: stale native/compiler/Graph/domain/ABI inputs; rerun qualification
- `MGGA_X_GVT4`: stale native/compiler/Graph/domain/ABI inputs; rerun qualification
- `MGGA_X_LTA`: stale native/compiler/Graph/domain/ABI inputs; rerun qualification

Physical empty-spin particle-birth continuation remains excluded;
a successful molecular force measurement does not qualify that domain.

## Revision diff

```text
Libxc capability changes
added functionals: none
removed functionals: none
identity changes: none
promotions: none
demotions: none
```
