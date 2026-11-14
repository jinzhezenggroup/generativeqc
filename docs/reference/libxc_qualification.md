# Native automatic Libxc qualification

Measured source revision: `fbf57c111c7c885efbc0cd1457e6a7732c9e3d95`.
Retained measured source inputs: `ba63873931ccffae7d9b2cea01831028f9bcf04a9903b818da76bd1539f64d64`.
Some retained contexts require targeted revalidation: `True`.
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
dependency-current paired energy/force acceptance across families, backends and spins.

```text
Libxc capability summary (222 functionals)
stage                qualified  ready  blocked
graph-imported             222      0        0
pointwise-validated        222      0        0
compiled-cpu                 0    222       11
compiled-cuda                0    222       11
production-domain           11    203        0
gpu-runtime                  0      0       11
molecular-scf                0      0       11
forces                       0      0       11
response                     0      0        0
public-method                0      0        0
public DFT: 0/222
explicit blockers: 55 across 11 functionals
```

| Family | Backend | Spin | Product | Qualified registrations |
| --- | --- | --- | --- | ---: |

Dependency-current backend/spin lifecycle records: 0.
Historical measurements remain in the retained campaign; stale receipts are not
counted as dependency-current admitted contexts. Unchanged dependencies can
inherit evidence; this does not claim a new binary was measured. Records retain exact
geometry, grid, precision, binary and derivative scope.
The [manifest](../../manifests/libxc/native-qualification.json) locates the
compressed machine-readable snapshot and independent numerical receipts.

## Retained executions

Incremental batches supplement, rather than relabel, earlier measurements.
Each context retains its own executed library identity in the snapshot.

| Measured revision/base | Measured source inputs | Executed ELF SHA-256 |
| --- | --- | --- |
| `fbf57c111c7c885efbc0cd1457e6a7732c9e3d95` | `ba63873931ccffae7d9b2cea01831028f9bcf04a9903b818da76bd1539f64d64` | `c2d1b725fb754e72d35ca5829ba9a067974703c6a98bc4e4cefb44ea6abc256c` |
| `fb8cbabd90fe77c07609987bfeea06162177ed21` | `9c225b7f02856f3a472786fcbe4b135ada91ff634c3809e6669f02e2212002ff` | `de2c01a15769c7942338ec9158e0fc0a517f5f447ae23c84475b0a6f5289121e` |
| `fb8cbabd90fe77c07609987bfeea06162177ed21` | `9c225b7f02856f3a472786fcbe4b135ada91ff634c3809e6669f02e2212002ff` | `de2c01a15769c7942338ec9158e0fc0a517f5f447ae23c84475b0a6f5289121e` |

## Targeted revalidation

Whole-tree/ELF hashes authenticate historical execution, not a global expiry switch.
The machine-readable snapshot lists changed inputs for each affected backend/product.
Contexts with unchanged dependencies retain acceptance; missing measurements are not failures.
ABI, build and unknown shared inputs require compatibility review and targeted regression.

| Registration | Backend | Products to revalidate | Reason |
| --- | --- | --- | --- |
| `GGA_C_APBE` | cpu | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `GGA_C_APBE` | cuda | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `GGA_X_APBE` | cpu | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `GGA_X_APBE` | cuda | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `GGA_X_B86` | cpu | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `GGA_X_B86` | cuda | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `GGA_X_PBE_SOL` | cpu | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `GGA_X_PBE_SOL` | cuda | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `GGA_X_RPBE` | cpu | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `GGA_X_RPBE` | cuda | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `LDA_C_CHACHIYO` | cpu | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `LDA_C_CHACHIYO` | cuda | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `LDA_C_PZ_MOD` | cpu | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `LDA_C_PZ_MOD` | cuda | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `LDA_C_VWN_4` | cpu | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `LDA_C_VWN_4` | cuda | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `MGGA_C_TPSS` | cpu | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `MGGA_C_TPSS` | cuda | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `MGGA_X_GVT4` | cpu | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `MGGA_X_GVT4` | cuda | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `MGGA_X_LTA` | cpu | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `MGGA_X_LTA` | cuda | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `LDA_C_VWN_4` | cpu | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `LDA_C_VWN_4` | cuda | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `GGA_X_APBE` | cpu | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `GGA_X_APBE` | cuda | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `MGGA_X_GVT4` | cpu | energy, forces | changed endpoint dependencies; targeted numerical regression required |
| `MGGA_X_GVT4` | cuda | energy, forces | changed endpoint dependencies; targeted numerical regression required |

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
- `GGA_C_APBE`: changed endpoint dependencies; targeted numerical regression required
- `GGA_X_APBE`: changed endpoint dependencies; targeted numerical regression required
- `GGA_X_B86`: changed endpoint dependencies; targeted numerical regression required
- `GGA_X_PBE_SOL`: changed endpoint dependencies; targeted numerical regression required
- `GGA_X_RPBE`: changed endpoint dependencies; targeted numerical regression required
- `LDA_C_CHACHIYO`: changed endpoint dependencies; targeted numerical regression required
- `LDA_C_PZ_MOD`: changed endpoint dependencies; targeted numerical regression required
- `LDA_C_VWN_4`: changed endpoint dependencies; targeted numerical regression required
- `MGGA_C_TPSS`: changed endpoint dependencies; targeted numerical regression required
- `MGGA_X_GVT4`: changed endpoint dependencies; targeted numerical regression required
- `MGGA_X_LTA`: changed endpoint dependencies; targeted numerical regression required
- `GGA_X_APBE`: missing native endpoint receipts

Physical empty-spin particle-birth continuation remains excluded;
a successful molecular force measurement does not qualify that domain.

## Revision diff

```text
Libxc capability changes
added functionals: none
removed functionals: none
identity changes: none
promotions: none
demotions:
  compiled-cpu: GGA_X_APBE, LDA_C_VWN_4, MGGA_X_GVT4
  compiled-cuda: GGA_X_APBE, LDA_C_VWN_4, MGGA_X_GVT4
  gpu-runtime: GGA_X_APBE, LDA_C_VWN_4, MGGA_X_GVT4
  molecular-scf: GGA_X_APBE, LDA_C_VWN_4, MGGA_X_GVT4
  forces: GGA_X_APBE, LDA_C_VWN_4, MGGA_X_GVT4
```
