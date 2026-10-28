# Native automatic Libxc qualification

Measured source revision: `36085eec7d745ad0e25514468ddfa42aea2c9f86`.
Source inputs: `ec311fd6a9bc3a7105ab42ff9f7e154400119a91d79b3b6987e62c3e09cfc0b3`.
Embedded native source identity: `a47263f92b809a0d6f52165a9eb42541017b879b437a738d608be93cd548db6f`.
Executed native library SHA-256: `d16bf758445db7fc0b1b043847979e7a12ef851ee98254be2e9a7f47dace0efa`.
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
compiled-cpu                11    210        0
compiled-cuda               11    210        0
production-domain           11    202        0
gpu-runtime                 11      0        0
molecular-scf               11      0        0
forces                      11      0        0
response                     0     11        0
public-method                0     11        0
public DFT: 0/221
explicit blockers: 0 across 0 functionals
```

| Family | Backend | Spin | Product | Qualified registrations |
| --- | --- | --- | --- | ---: |
| gga | cpu | polarized | energy | 5 |
| gga | cpu | polarized | forces | 5 |
| gga | cpu | unpolarized | energy | 5 |
| gga | cpu | unpolarized | forces | 5 |
| gga | cuda | polarized | energy | 5 |
| gga | cuda | polarized | forces | 5 |
| gga | cuda | unpolarized | energy | 5 |
| gga | cuda | unpolarized | forces | 5 |
| lda | cpu | polarized | energy | 3 |
| lda | cpu | polarized | forces | 3 |
| lda | cpu | unpolarized | energy | 3 |
| lda | cpu | unpolarized | forces | 3 |
| lda | cuda | polarized | energy | 3 |
| lda | cuda | polarized | forces | 3 |
| lda | cuda | unpolarized | energy | 3 |
| lda | cuda | unpolarized | forces | 3 |
| mgga | cpu | polarized | energy | 3 |
| mgga | cpu | polarized | forces | 3 |
| mgga | cpu | unpolarized | energy | 3 |
| mgga | cpu | unpolarized | forces | 3 |
| mgga | cuda | polarized | energy | 3 |
| mgga | cuda | polarized | forces | 3 |
| mgga | cuda | unpolarized | energy | 3 |
| mgga | cuda | unpolarized | forces | 3 |

Measured backend/spin lifecycle records: 44.
All records retain exact geometry, grid, precision, binary and derivative scope.
The [manifest](../../manifests/libxc/native-qualification.json) locates the
compressed machine-readable snapshot and independent numerical receipts.

## Qualified registrations

| Registration | Backend | Product | Qualified spin layouts |
| --- | --- | --- | --- |
| `GGA_C_APBE` | cpu | energy | polarized, unpolarized |
| `GGA_C_APBE` | cpu | forces | polarized, unpolarized |
| `GGA_C_APBE` | cuda | energy | polarized, unpolarized |
| `GGA_C_APBE` | cuda | forces | polarized, unpolarized |
| `GGA_X_APBE` | cpu | energy | polarized, unpolarized |
| `GGA_X_APBE` | cpu | forces | polarized, unpolarized |
| `GGA_X_APBE` | cuda | energy | polarized, unpolarized |
| `GGA_X_APBE` | cuda | forces | polarized, unpolarized |
| `GGA_X_B86` | cpu | energy | polarized, unpolarized |
| `GGA_X_B86` | cpu | forces | polarized, unpolarized |
| `GGA_X_B86` | cuda | energy | polarized, unpolarized |
| `GGA_X_B86` | cuda | forces | polarized, unpolarized |
| `GGA_X_PBE_SOL` | cpu | energy | polarized, unpolarized |
| `GGA_X_PBE_SOL` | cpu | forces | polarized, unpolarized |
| `GGA_X_PBE_SOL` | cuda | energy | polarized, unpolarized |
| `GGA_X_PBE_SOL` | cuda | forces | polarized, unpolarized |
| `GGA_X_RPBE` | cpu | energy | polarized, unpolarized |
| `GGA_X_RPBE` | cpu | forces | polarized, unpolarized |
| `GGA_X_RPBE` | cuda | energy | polarized, unpolarized |
| `GGA_X_RPBE` | cuda | forces | polarized, unpolarized |
| `LDA_C_CHACHIYO` | cpu | energy | polarized, unpolarized |
| `LDA_C_CHACHIYO` | cpu | forces | polarized, unpolarized |
| `LDA_C_CHACHIYO` | cuda | energy | polarized, unpolarized |
| `LDA_C_CHACHIYO` | cuda | forces | polarized, unpolarized |
| `LDA_C_PZ_MOD` | cpu | energy | polarized, unpolarized |
| `LDA_C_PZ_MOD` | cpu | forces | polarized, unpolarized |
| `LDA_C_PZ_MOD` | cuda | energy | polarized, unpolarized |
| `LDA_C_PZ_MOD` | cuda | forces | polarized, unpolarized |
| `LDA_C_VWN_4` | cpu | energy | polarized, unpolarized |
| `LDA_C_VWN_4` | cpu | forces | polarized, unpolarized |
| `LDA_C_VWN_4` | cuda | energy | polarized, unpolarized |
| `LDA_C_VWN_4` | cuda | forces | polarized, unpolarized |
| `MGGA_C_TPSS` | cpu | energy | polarized, unpolarized |
| `MGGA_C_TPSS` | cpu | forces | polarized, unpolarized |
| `MGGA_C_TPSS` | cuda | energy | polarized, unpolarized |
| `MGGA_C_TPSS` | cuda | forces | polarized, unpolarized |
| `MGGA_X_GVT4` | cpu | energy | polarized, unpolarized |
| `MGGA_X_GVT4` | cpu | forces | polarized, unpolarized |
| `MGGA_X_GVT4` | cuda | energy | polarized, unpolarized |
| `MGGA_X_GVT4` | cuda | forces | polarized, unpolarized |
| `MGGA_X_LTA` | cpu | energy | polarized, unpolarized |
| `MGGA_X_LTA` | cpu | forces | polarized, unpolarized |
| `MGGA_X_LTA` | cuda | energy | polarized, unpolarized |
| `MGGA_X_LTA` | cuda | forces | polarized, unpolarized |

## Explicit blockers

- `MGGA_X_EDMGGA`: unsupported-ingredients:laplacian
- `MGGA_X_GDME_0`: unsupported-ingredients:laplacian
- `MGGA_X_GDME_KOS`: unsupported-ingredients:laplacian
- `MGGA_X_GDME_NV`: unsupported-ingredients:laplacian
- `MGGA_X_GDME_VT`: unsupported-ingredients:laplacian
- `MGGA_X_JK`: unsupported-ingredients:laplacian
- `MGGA_X_MK00`: unsupported-ingredients:laplacian
- `MGGA_X_RLDA`: unsupported-ingredients:laplacian

Physical empty-spin particle-birth continuation remains excluded;
a successful molecular force measurement does not qualify that domain.

## Revision diff

```text
Libxc capability changes
added functionals: none
removed functionals: none
identity changes: none
promotions:
  compiled-cpu: GGA_C_APBE, GGA_X_APBE, GGA_X_B86, GGA_X_PBE_SOL, GGA_X_RPBE, LDA_C_CHACHIYO, LDA_C_PZ_MOD, LDA_C_VWN_4, MGGA_C_TPSS, MGGA_X_GVT4, MGGA_X_LTA
  compiled-cuda: GGA_C_APBE, GGA_X_APBE, GGA_X_B86, GGA_X_PBE_SOL, GGA_X_RPBE, LDA_C_CHACHIYO, LDA_C_PZ_MOD, LDA_C_VWN_4, MGGA_C_TPSS, MGGA_X_GVT4, MGGA_X_LTA
  production-domain: GGA_C_APBE, GGA_X_APBE, GGA_X_B86, GGA_X_PBE_SOL, GGA_X_RPBE, LDA_C_CHACHIYO, LDA_C_PZ_MOD, LDA_C_VWN_4, MGGA_C_TPSS, MGGA_X_GVT4, MGGA_X_LTA
  gpu-runtime: GGA_C_APBE, GGA_X_APBE, GGA_X_B86, GGA_X_PBE_SOL, GGA_X_RPBE, LDA_C_CHACHIYO, LDA_C_PZ_MOD, LDA_C_VWN_4, MGGA_C_TPSS, MGGA_X_GVT4, MGGA_X_LTA
  molecular-scf: GGA_C_APBE, GGA_X_APBE, GGA_X_B86, GGA_X_PBE_SOL, GGA_X_RPBE, LDA_C_CHACHIYO, LDA_C_PZ_MOD, LDA_C_VWN_4, MGGA_C_TPSS, MGGA_X_GVT4, MGGA_X_LTA
  forces: GGA_C_APBE, GGA_X_APBE, GGA_X_B86, GGA_X_PBE_SOL, GGA_X_RPBE, LDA_C_CHACHIYO, LDA_C_PZ_MOD, LDA_C_VWN_4, MGGA_C_TPSS, MGGA_X_GVT4, MGGA_X_LTA
demotions: none
```
