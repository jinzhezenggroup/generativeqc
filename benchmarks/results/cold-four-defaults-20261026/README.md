# Four default cold endpoints: DF whitening and WB97M-V AOT

Three alternating native/GPU4PySCF rounds use fresh processes on one RTX 5090
under Slurm job 14379. The #2786 protocol measures constructor, preparation,
first energy and analytic forces, and host export, followed by three calls from
the engine's frozen cold density. Both engines keep their default grids and
ordinary convergence settings. The native controls are the campaign's existing
`GENERATIVEQC_CUDA_XC_SPATIAL_ORDER=2` and `GENERATIVEQC_SCF_VALIDATION_GEMM=1`;
no experimental whitening, screening or Schwarz switch is enabled.

| Endpoint | Native cold median [range], s | GPU4PySCF cold median [range], s | Native / GPU4 warm median, s |
| --- | ---: | ---: | ---: |
| Direct WB97M-V48 | 82.4468 [82.1620, 82.5628] | 96.5556 [96.1257, 96.5701] | 6.5644 / 32.3572 |
| DF WB97M-V48 | 67.7175 [67.6460, 67.7204] | 90.5603 [90.4831, 91.0296] | 5.6707 / 32.1400 |
| DF HF96 | 19.3003 [19.2690, 19.3337] | 20.0240 [19.9389, 20.0610] | 4.0917 / 9.9844 |
| DF PBE0-96 | 24.6200 [24.6190, 24.6330] | 26.1959 [26.1075, 26.2063] | 8.1397 / 13.3643 |

All 24 populations and 96 complete calls are retained. Each native cold range
is below its GPU4PySCF range; the median advantages are 14.6%, 25.2%, 3.6% and
6.0%, respectively. This is a static default-grid latency comparison, not a
claim that every method/size or common-accuracy configuration has caught up.
Direct HF96 and Direct r2SCAN-3c96 remain separate unfinished work.

## What changed

Full-rank large retained DF factors use `M=L L^T`, `F=A L^-T` and panel TRSM
instead of spectral panel GEMM. The force owner carries the explicit reverse
map `L^-1`; rank, scratch and capacity refusals keep spectral whitening. The
existing eigensystem and cutoff remain authoritative. Research same-binary
three-round baseline/candidate medians were 20.9577/19.2429 seconds for DF HF96
and 26.3150/24.5842 for DF PBE0-96. These ablations explain the mechanism;
the table uses the separate production-default image.

WB97M-V's exact default composite force wrapper is now packaged AOT for both
spins and ordinary/SPD providers. Cache-miss startup previously invoked NVCC
inside the first force endpoint. Intrusive startup tracing measured 80.44 s in
that first compilation and none on replay, with unchanged SCF work. The
82/68-second cached baseline endpoints are **not** claimed as a newly faster SCF
algorithm. AOT removes this runtime compiler/cache dependence. Custom domains
keep bounded JIT; missing/corrupt declared packages fail closed.

The ordinary native work remains visible: cold SCF/Fock counts are WB 28/29,
HF 11/12, and PBE0 9/9, plus seven initial-guess Focks. Warm counts are WB/PBE0
1/1 and HF 1/2. WB cold retains four VV10 pair sweeps. GPU4PySCF's SCF counts
are 9, 10 and 9, respectively; the driver does not export its Fock counts.
The summary preserves the complete available work and grid diagnostics.

## Numerical qualification and failures

Every native cold/warm sample agrees with the preserved original native
endpoint within 7.28e-12 Eh and 3.03e-12 Eh/Bohr. Dense-reference qualification
reuses the receipt-bound, grid-stability/CPU-spot-checked references from
[`readme-final-20261022`](../readme-final-20261022/README.md). The WB and PBE0
populations pass the unchanged common gates. **DF HF96 fails the energy gate
for both engines**: the fitting error against direct HF is approximately
2.105e-5 Eh/atom, above 1e-5. Its force gates pass. No failed population is hidden.

The production library passes 108 independent long-double metric-adjoint
cases, including rotated/Cholesky gauges, condition numbers through 1e10 and
ragged panels. Native raw-integral oracles cover RHF/UHF, truncated metrics,
capacity refusal, one-Q occupied response, final leases and general density;
the occupied-response suite also passes Compute Sanitizer memcheck.

WB cold/warm, energy-only, finite-difference and moved-force checks pass for
RKS H2/STO-3G, UKS H3/STO-3G and RKS water/def2-SVP while compiler discovery,
source emission and reduction-IR generation are forbidden. All 28 stationary
packages pass native-cubin auditing. Three larger-basis WB energy checks failed
the original run at the unchanged 1e-8 Eh threshold: water/def2-TZVP (2.50e-8), local
def2-TZVPD (2.56e-8), and NH2 UKS/def2-TZVP (8.05e-8). Each was reproduced on
the frozen baseline. These are not claimed as passing checks.

### VV10 reference-domain follow-up

The three failures came from an outdated reference configuration. PR #2652
changed the native molecular VV10 density domain to rho >= 1e-10, but this live
test still forced GPU4PySCF's SCF and gradient masks to rho >= 1e-8. The test now
uses the declared molecular density threshold for both cold and displaced
reference calls and records it with the numerical results. The independent
engine, native production code, grid and acceptance thresholds are unchanged.

Slurm job 14542 on n1/RTX 5090 replays the original frozen `wb-aot2` image with
the PR-head test: the old reference mask reproduces **3 failures**, and the
matched mask gives **3 passes**. Both runs use the identical native library.

| Case | Original energy error, Eh | Matched energy error, Eh | Matched max force error, Eh/Bohr |
| --- | ---: | ---: | ---: |
| RKS water/def2-TZVP | 2.50e-8 | 1.81e-10 | 4.83e-10 |
| RKS water/local def2-TZVPD | 2.56e-8 | 4.47e-10 | 1.01e-9 |
| UKS NH2/def2-TZVP | 8.05e-8 | 4.88e-11 | 4.15e-9 |

All cold/warm, energy-only, translation, three-step finite-difference and moved
force checks pass with compiler discovery/emission still forbidden. Across
moved geometries the maximum energy/force errors are 4.68e-10 Eh and 4.72e-9
Eh/Bohr. The unchanged gates are 1e-8 Eh and 1e-7 Eh/Bohr. Nine host checks cover
the density boundary, matched response grids and failure-safe reference-mask
restoration; Ruff lint/format and diff whitespace checks pass.

[`wb-domain-followup.json.xz`](wb-domain-followup.json.xz) retains both outcomes,
complete numerical receipts, allocation, package versions, binary/test hashes
and the runner. Raw logs remain at `n1:/data/jzzeng/wb-domain-2795-20261026/`.
This is a controlled replay of the original acceptance image, not a new
exact-head native rebuild or performance campaign. The original failure
receipts and `summary.json.xz` are retained unchanged; this follow-up explains
and resolves those three reference-domain failures without claiming broader
WB domain qualification or changing the separate DF HF96 fitting limitation.

## Provenance and reproduction

The timing image is `wb-aot2`, based on
`82ce9e56a352fcc1d86e7d64eeaae08c6f1eb5c8` plus patch SHA-256
`c5007949a252204e6e49d6e689ce9715eda45ad0e79a49bdf0a3d97e123a498f`;
the archive digest, complete binary hashes, allocation, compiler commands hash,
verified sccache 0.16.0 before/after statistics and exact run commands are in
`summary.json.xz`. CUDA 12.9.1/GCC 13.3, Release/sm120, one BLAS/OpenMP thread
and scheduler-assigned GPU visibility were used. No cache was cleared.

The PR follow-up moves unchanged potrf query/submission into the shared solver
provider and adds official SDK-derived host ABI declarations. It also inherits
master's automatic KS resource fix. This source is **not byte-identical** to the
timing image above. [`provider-followup.json`](provider-followup.json) records
the changed/added production inputs and the separately frozen
`phase-four-provider-v2` rebuild. Slurm job 14415 passes the native DF/occupied
oracles, 48 weighted finite differences and 108 independent metric adjoints.
Fresh HF/PBE0 DF96 cold-plus-three-warm calls retain identical energies and
forces within 5.91e-13 Eh/Bohr of the original campaign. Their cold times are
19.1617/24.7602 seconds; these single follow-up rounds do not replace the
three-round comparison. The host ABI tests pass 22 cases with one optional skip.

Full archives, patches, builds, profiles and receipts remain in
`n1:/data/jzzeng/cold-opt-20261025/`. `default-four-paired.py` and `build-wb.sh`
there reproduce the campaign with a **new variant/output directory** under a
finite `srun --partition=main --nodelist=node1 --gres=gpu:5090:1` allocation.
Do not overwrite retained measurements or alter `CUDA_VISIBLE_DEVICES`.

For host-only receipt-integrity and independent requalification, copy
`default-four-paired/` from that root and run:

```bash
PYTHONPATH=python:. python benchmarks/results/cold-four-defaults-20261026/recheck.py \
  --raw-directory /path/to/copied/campaign
```

This opt-in historical check retains the known failed HF populations; it does
not pin current implementation spelling. Design rationale is in the
[DF note](../../../.agents/notes/implemented/performance/2026-10-26-df-cholesky-whitening.md)
and [WB note](../../../.agents/notes/implemented/performance/2026-10-26-wb97mv-stationary-aot.md).
