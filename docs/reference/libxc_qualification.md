# Native automatic Libxc qualification

Measured source revision: `36085eec7d745ad0e25514468ddfa42aea2c9f86`.
Retained measured source inputs: `ec311fd6a9bc3a7105ab42ff9f7e154400119a91d79b3b6987e62c3e09cfc0b3`.
Current checked source inputs: `ed1c36cacd1cf6d3e461e9db06f1b6ce77fab6ae6cd2bc248a1d21b8b55c7522`.
Current checked native input identity: `63c424c9183f3f9b06287c911c4d9f68cbc61cc3cbd31a9fdfe2006b4f0fac1c`.
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
identity changes: GGA_C_ACGGA, GGA_C_ACGGAP, GGA_C_AM05, GGA_C_APBE, GGA_C_BMK, GGA_C_CCDF, GGA_C_CS1, GGA_C_GAM, GGA_C_HCTH_A, GGA_C_HYB_TAU_HCTH, GGA_C_LM, GGA_C_LYP, GGA_C_MGGAC, GGA_C_N12, GGA_C_N12_SX, GGA_C_OPTC, GGA_C_P86, GGA_C_P86VWN, GGA_C_P86VWN_FT, GGA_C_P86_FT, GGA_C_PBE, GGA_C_PBEFE, GGA_C_PBEINT, GGA_C_PBELOC, GGA_C_PBE_GAUSSIAN, GGA_C_PBE_JRGX, GGA_C_PBE_MOL, GGA_C_PBE_SOL, GGA_C_PBE_VWN, GGA_C_PW91, GGA_C_REGTPSS, GGA_C_RGE2, GGA_C_SCAN_E0, GGA_C_SOGGA11, GGA_C_SOGGA11_X, GGA_C_SPBE, GGA_C_TAU_HCTH, GGA_C_TCA, GGA_C_TM_LYP, GGA_C_TM_PBE, GGA_C_WI, GGA_C_WI0, GGA_C_WL, GGA_C_XPBE, GGA_C_ZVPBELOC, GGA_XC_TH1, GGA_XC_TH2, GGA_XC_TH_FC, GGA_XC_TH_FCFO, GGA_XC_TH_FCO, GGA_XC_TH_FL, GGA_X_AIRY, GGA_X_AK13, GGA_X_APBE, GGA_X_B86, GGA_X_B86_MGC, GGA_X_B86_R, GGA_X_B88, GGA_X_B88M, GGA_X_B88_6311G, GGA_X_BAYESIAN, GGA_X_BCGP, GGA_X_BKL1, GGA_X_BKL2, GGA_X_BPCCAC, GGA_X_C09X, GGA_X_CAP, GGA_X_CHACHIYO, GGA_X_DK87_R1, GGA_X_DK87_R2, GGA_X_EB88, GGA_X_ECMV92, GGA_X_EV93, GGA_X_G96, GGA_X_KT1, GGA_X_LAG, GGA_X_LG93, GGA_X_LV_RPW86, GGA_X_MB88, GGA_X_OL2, GGA_X_OPTB86B_VDW, GGA_X_OPTB88_VDW, GGA_X_OPTX, GGA_X_PBE, GGA_X_PBEFE, GGA_X_PBEINT, GGA_X_PBEK1_VDW, GGA_X_PBETRANS, GGA_X_PBE_GAUSSIAN, GGA_X_PBE_JSJR, GGA_X_PBE_MOD, GGA_X_PBE_MOL, GGA_X_PBE_R, GGA_X_PBE_SOL, GGA_X_PBE_TCA, GGA_X_PW86, GGA_X_PW91, GGA_X_Q1D, GGA_X_Q2D, GGA_X_RGE2, GGA_X_RPBE, GGA_X_RPW86, GGA_X_SG4, GGA_X_SOGGA11, GGA_X_SSB_SW, GGA_X_XPBE, LDA_C_BR78, LDA_C_CHACHIYO, LDA_C_CHACHIYO_MOD, LDA_C_GK72, LDA_C_GL, LDA_C_GOMBAS, LDA_C_HL, LDA_C_KARASIEV, LDA_C_KARASIEV_MOD, LDA_C_LP96, LDA_C_MCWEENY, LDA_C_ML1, LDA_C_ML2, LDA_C_OB_PW, LDA_C_OB_PZ, LDA_C_OW, LDA_C_OW_LYP, LDA_C_PW, LDA_C_PW_MOD, LDA_C_PW_RPA, LDA_C_PZ, LDA_C_PZ_MOD, LDA_C_RC04, LDA_C_RPA, LDA_C_RPW92, LDA_C_UPW92, LDA_C_VBH, LDA_C_VWN, LDA_C_VWN_1, LDA_C_VWN_2, LDA_C_VWN_3, LDA_C_VWN_4, LDA_C_VWN_RPA, LDA_C_WIGNER, LDA_XC_LP_A, LDA_XC_LP_B, LDA_XC_TETER93, LDA_XC_ZLP, LDA_X_SLOC, MGGA_C_CC, MGGA_C_CCALDA, MGGA_C_R2SCAN, MGGA_C_R2SCAN01, MGGA_C_REVTM, MGGA_C_REVTPSS, MGGA_C_RMGGAC, MGGA_C_RPPSCAN, MGGA_C_RREGTM, MGGA_C_RSCAN, MGGA_C_SCAN, MGGA_C_TM, MGGA_C_TPSS, MGGA_C_TPSSLOC, MGGA_C_TPSS_GAUSSIAN, MGGA_X_EDMGGA, MGGA_X_GDME_0, MGGA_X_GDME_KOS, MGGA_X_GDME_NV, MGGA_X_GDME_VT, MGGA_X_GVT4, MGGA_X_HLTA, MGGA_X_JK, MGGA_X_KTBM_0, MGGA_X_KTBM_1, MGGA_X_KTBM_10, MGGA_X_KTBM_11, MGGA_X_KTBM_12, MGGA_X_KTBM_13, MGGA_X_KTBM_14, MGGA_X_KTBM_15, MGGA_X_KTBM_16, MGGA_X_KTBM_17, MGGA_X_KTBM_18, MGGA_X_KTBM_19, MGGA_X_KTBM_2, MGGA_X_KTBM_20, MGGA_X_KTBM_21, MGGA_X_KTBM_22, MGGA_X_KTBM_23, MGGA_X_KTBM_24, MGGA_X_KTBM_3, MGGA_X_KTBM_4, MGGA_X_KTBM_5, MGGA_X_KTBM_6, MGGA_X_KTBM_7, MGGA_X_KTBM_8, MGGA_X_KTBM_9, MGGA_X_KTBM_GAP, MGGA_X_LTA, MGGA_X_MK00, MGGA_X_MS0, MGGA_X_MS1, MGGA_X_MS2, MGGA_X_MS2_REV, MGGA_X_MSB86BL, MGGA_X_MSPBEL, MGGA_X_MSRPBEL, MGGA_X_MVS, MGGA_X_MVSB, MGGA_X_MVSBS, MGGA_X_PKZB, MGGA_X_R2SCAN, MGGA_X_R2SCAN01, MGGA_X_R4SCAN, MGGA_X_REGTM, MGGA_X_REVSCAN, MGGA_X_REVTM, MGGA_X_RLDA, MGGA_X_RMSB86BL, MGGA_X_RMSPBEL, MGGA_X_RMSRPBEL, MGGA_X_SCAN, MGGA_X_TH, MGGA_X_TLDA, MGGA_X_TM
promotions: none
demotions:
  compiled-cpu: GGA_C_APBE, GGA_X_APBE, GGA_X_B86, GGA_X_PBE_SOL, GGA_X_RPBE, LDA_C_CHACHIYO, LDA_C_PZ_MOD, LDA_C_VWN_4, MGGA_C_TPSS, MGGA_X_GVT4, MGGA_X_LTA
  compiled-cuda: GGA_C_APBE, GGA_X_APBE, GGA_X_B86, GGA_X_PBE_SOL, GGA_X_RPBE, LDA_C_CHACHIYO, LDA_C_PZ_MOD, LDA_C_VWN_4, MGGA_C_TPSS, MGGA_X_GVT4, MGGA_X_LTA
  production-domain: GGA_C_APBE, GGA_X_APBE, GGA_X_B86, GGA_X_PBE_SOL, GGA_X_RPBE, LDA_C_CHACHIYO, LDA_C_PZ_MOD, LDA_C_VWN_4, MGGA_C_TPSS, MGGA_X_GVT4, MGGA_X_LTA
  gpu-runtime: GGA_C_APBE, GGA_X_APBE, GGA_X_B86, GGA_X_PBE_SOL, GGA_X_RPBE, LDA_C_CHACHIYO, LDA_C_PZ_MOD, LDA_C_VWN_4, MGGA_C_TPSS, MGGA_X_GVT4, MGGA_X_LTA
  molecular-scf: GGA_C_APBE, GGA_X_APBE, GGA_X_B86, GGA_X_PBE_SOL, GGA_X_RPBE, LDA_C_CHACHIYO, LDA_C_PZ_MOD, LDA_C_VWN_4, MGGA_C_TPSS, MGGA_X_GVT4, MGGA_X_LTA
  forces: GGA_C_APBE, GGA_X_APBE, GGA_X_B86, GGA_X_PBE_SOL, GGA_X_RPBE, LDA_C_CHACHIYO, LDA_C_PZ_MOD, LDA_C_VWN_4, MGGA_C_TPSS, MGGA_X_GVT4, MGGA_X_LTA
```
