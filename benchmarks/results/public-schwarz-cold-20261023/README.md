# Direct DFT cold: public-AO Schwarz specialization

Measured on 2026-10-23 against master
`1e77b239e551856741abedc0b45a6e8461f261b9`. The candidate is the local patch
on that base, identified by the input-manifest hashes and changed input hashes
in `summary.json.xz`. The complete manifests remain with the raw campaign.
The benchmark driver's `source_revision` field names the common base; it does
not imply that the candidate is an unmodified checkout of that commit.

## Result and scope

Public-AO Schwarz preparation was evaluating known s/p/d shell pairs through
the generic primitive recurrence. The candidate reuses existing specialized
shell recurrences and the original primitive/term contraction order. Both AO
orientations, signed spherical coefficients, FP64, screening thresholds and
capacity charges are preserved. f shells keep the generic fallback. No new
resident bytes are needed. `GENERATIVEQC_DIRECT_SPECIALIZED_SCHWARZ=0` selects
the generic preparation schedule.

Complete cold energy-and-force endpoint medians on 96-atom water clusters:

| Method | Baseline (s) | Candidate (s) | Reduction | Cold SCF / Fock builds |
| --- | ---: | ---: | ---: | ---: |
| PBE | 21.861572 | 21.035664 | 3.78% | 11 / 11 |
| PBE0 | 28.665403 | 27.615102 | 3.66% | 9 / 9 |
| r2SCAN-3c | 49.488149 | 46.486468 | 6.07% | 10 / 10 |

Three alternating baseline/candidate rounds share a single Slurm allocation
(job 13514) on n1/node1, RTX 5090. Every population starts a fresh process and
measures constructor, preparation, first SCF, analytic forces and host export,
followed by three warm calls initialized from that process's frozen cold
density. Imports and AOT compilation are outside the endpoint. No oracle
density enters native execution. All samples are retained, including the first
PBE0 baseline cold sample of 30.713383 seconds.

All three cold gains exceed the existing robustness floor
`max(2%, 2 * (baseline MAD / median + candidate MAD / median))`.
`summary.json.xz` retains all medians, MADs, ranges and individual times.
Warm medians (baseline/candidate) are PBE 5.453777/5.483969, PBE0
7.190364/6.958069 and r2SCAN-3c 16.074791/16.165229 seconds. Every warm
sample uses one SCF iteration/Fock build. No warm gain is attributed to this
preparation change; PBE and r2SCAN-3c increases are below 0.6%.

A separate intrusive Nsight trace of PBE0/96 attributes 1.130919799 GPU
seconds to baseline public Schwarz kernels and 0.122300348 to candidate
(89.2% reduction). Both evaluate 589,824 oriented AO-pair diagonals in five
launches. Profile times are mechanism evidence and are excluded from endpoint
populations. This patch reduces one cold bottleneck; it does not establish
that Direct has caught GPU4PySCF across methods or sizes.

## Scientific qualification and limitations

The clean cohort has 72 complete calls. Standard 3/12/48-atom and Fast
96-atom PBE0/r2SCAN-3c holdouts add 64 calls. PBE0/12/96 and r2SCAN-3c/96
cold/warm/changed-geometry checks add 18 calls. Two supplemental Fast
r2SCAN-3c/96 pairs add 16 calls: **170 independently qualified endpoints**.
Holdout/moved single observations are regression checks, not speedup claims.

The README default grids, basis sets and convergence profiles are unchanged;
full force-grid response is enabled in both arms. Static qualification reuses
the explicitly audited dense references in
[`readme-final-20261022`](../readme-final-20261022/README.md), including grid
stability and CPU/GPU spot checks. Gates remain 1e-5 Eh/atom, 1e-4 Eh/Bohr
maximum force error and 3e-5 Eh/Bohr RMS force error. Moved atom 1 has
z displaced by 0.02 Bohr; freshly converged, tighter full-density GPU4PySCF
references validate both versions against 2.5e-6, 2.5e-5 and 7.5e-6 gates.
Reference work occurs after native timings. Exact settings, grids, operator
inventories, packages, geometries, forces and references remain in the receipts.

All static Standard baseline/candidate pairs pass the additional 1e-9 Eh and 1e-8
Eh/Bohr diagnostic (maxima 1.365e-12 and 1.662e-9). The first Fast 3c/96
pair differs by 4.372e-8 Eh/Bohr and fails that stricter diagnostic while
passing its independent scientific gates. The two supplemental pairs differ
by 3.129e-8 and 3.176e-8. Comparing fresh processes of the same binary gives
maximum variation 4.042e-8 for unchanged baseline and 3.015e-8 for candidate.
The cause of that variation is not established. Failures remain visible;
the diagnostic threshold was not relaxed and Fast bitwise equivalence is
not claimed.

## Validation and build provenance

- Native GPU bounds pass independent CPU-integral and generic-GPU checks for
  s/p/d/f, spherical/Cartesian AOs, signed contractions, two distinct systems,
  exact mandatory capacity and device nonfinite failure.
- Compute Sanitizer memcheck reports zero errors. Six high-angular GPU tests
  pass, including RHF/UHF energies and forces against Libcint/PySCF.
- 112 host tests pass; 48 CUDA cases intentionally skip in the host selection.
  Compiler dependency, ownership, structural lint and formatting checks pass.
- The existing `--canonical-values-only` suite fails with `range fallback
  displaced an independently generated J/K channel` on both the preserved
  original baseline executable/library and candidate. It is not a passing test.
- The first baseline endpoint attempt lacked `pbe0_rks_spd`. Both source-matched
  complete AOT packages were built before the clean campaign. Failed and pilot
  attempts are excluded and retained separately.

CUDA 12.9.1, GCC 13.3, Release/sm120 and verified sccache 0.16.0 were used.
Caches were reused; compiler commands and before/after cache statistics were
retained. All 5,026 baseline and 5,027 candidate source/build/runtime/test
inputs matched their manifests. Core and ordinary/SPD PBE/PBE0/r2SCAN/M06-2X
plugin hashes are in the summary. Benchmark processes use one OpenMP/OpenBLAS/
MKL thread and preserve Slurm GPU visibility.

## Reading and reproducing the evidence

- `summary.json.xz`: qualifications, all sample times/work counts, paired
  diagnostics, profile CSVs, relevant logs/cache statistics, binary hashes and
  exact build/run scripts.
- `recheck.py`: opt-in host-only artifact-integrity and independent-reference
  requalification using the retained raw campaign, including preserved paired
  diagnostic failures. This is historical evidence checking, not a required
  source-fingerprint CI test.

From the candidate worktree root, which already has the raw campaign in
`.artifacts/remote`:

```bash
PYTHONPATH=python:. python benchmarks/results/public-schwarz-cold-20261023/recheck.py
```

Full builds, source trees, command inventories and Nsight reports remain at
`n1:/data/jzzeng/direct-dft-cold-fix-20261023/`; local copies of compact raw
artifacts are under the candidate worktree's ignored `.artifacts/`. All 44 raw
endpoint populations and both complete source manifests are retained there.
For another checkout, copy the `clean/`, `holdout/`, `fast-repeat/`, `moved/`
JSON receipts and `baseline-source.json`/`candidate-source.json` from that root
and pass their containing directory with `--raw-directory`. Only the compact
summary is checked in, respecting the repository's aggregate evidence budget.
For a fresh build, use separate baseline and candidate source trees, verify
their inputs, and adapt the summary's recorded script paths. `build.sh` was
run sequentially as baseline then candidate in `source/`; the baseline AOT
completion script instead uses the immutable `baseline-source/` archive.
Do not run its baseline command against a candidate tree and label it baseline.
Package both ordinary and SPD stationary artifacts before timing.

Recorded run scripts require an existing finite Slurm allocation. For example,
on n1, with the prepared immutable binaries and a **new output directory**:

```bash
srun --partition=main --nodelist=node1 --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:45:00 bash campaign.sh
```

The script must point to the new output directory to preserve these receipts.
Do not alter `CUDA_VISIBLE_DEVICES` or rerun into the retained campaign paths.
The durable design rationale is in the
[Agent Note](../../../.agents/notes/implemented/performance/2026-10-23-public-schwarz-shell-pairs.md).
