# Native automatic Libxc qualification

Measured source revision: `fbf57c111c7c885efbc0cd1457e6a7732c9e3d95`.
Retained measured source inputs: `ba63873931ccffae7d9b2cea01831028f9bcf04a9903b818da76bd1539f64d64`.
Current checked source inputs: `2941e70ebefe56777983465dece1107925801c63e9d25f5bd8e3ee760aab444d`.
Current checked native input identity: `a305bde4a845e554e5f04ca4ab593b26166b7e31902635dcd5a900194532aea3`.
Retained evidence stale for checked inputs: `True`.
Retained executed native library SHA-256: `c2d1b725fb754e72d35ca5829ba9a067974703c6a98bc4e4cefb44ea6abc256c`.
Independent oracle: PySCF 2.14.0 / Libxc 7.0.0.
Structurally selectable non-curated automatic registrations: 198; registrations without a retained native campaign: 211.

This offline qualification inventory does not restrict structural public selection.
The automatic count excludes registrations already owned by curated components.
Counts apply only to the exact retained native contexts; no response, Hessian,
HVP, higher-l, mixed/Fast, ECP or DF scope is inferred.
The `public-method` row counts the optional stronger admission receipt,
not available public selectors. Its absence does not disable a method.
A successful default check establishes archival consistency only. Use
`python tools/check_libxc_native_catalog.py --require-current` for representative
source-matched paired energy/force acceptance across families, backends and spins.

```text
Libxc capability summary (222 functionals)
stage                qualified  ready  blocked
graph-imported             222      0        0
pointwise-validated        222      0        0
compiled-cpu                 0    222       11
compiled-cuda                0    222       11
production-domain            0    214       11
gpu-runtime                  0      0       11
molecular-scf                0      0       11
forces                       0      0       11
response                     0      0        0
public-method                0      0        0
public DFT: 0/222
explicit blockers: 66 across 11 functionals
```

| Family | Backend | Spin | Product | Qualified registrations |
| --- | --- | --- | --- | ---: |

Current-source observed backend/spin lifecycle records: 0.
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
