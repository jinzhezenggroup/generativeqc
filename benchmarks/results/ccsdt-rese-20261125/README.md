# CCSD(T) equilibrium geometries against semi-experimental structures

This campaign supports [#3166](https://github.com/jinzhezenggroup/generativeqc/issues/3166)
for the Chem Sci revision. It uses the native CUDA optimizer at
[`22f72d0692a760c2892cd7791022584d9ae1c3e5`](https://github.com/jinzhezenggroup/generativeqc/commit/22f72d0692a760c2892cd7791022584d9ae1c3e5).
The 22 gas-phase neutral singlets span diatomics, linear polyatomics,
bent molecules and small organic molecules. Selection used published
semi-experimental equilibrium parameters, closed-shell ground-state
references, H/C/N/O/F basis availability and tractable sizes. It preceded
production measurements. The complete reference inventory has 46 bond lengths
and 28 angles.

CUDA DF-CCSD(T)/cc-pCVTZ converged for **20/22 molecules**.
Against the semi-experimental structures, bond MAE was
**0.00273 Å** and angle MAE was **0.4255°**.
Two conventional controls reproduce matched cc-pCVTZ literature lengths
within 0.000029 Å. Independent PySCF checks confirmed four local minima,
including three DF geometries.

## Protocol

Each molecule starts at the rough Cartesian coordinates in [cohort.json](cohort.json).
A CUDA RHF/cc-pVDZ preoptimization supplies the initial coordinates for a fresh
native CCSD(T) process. The primary 22-molecule cohort uses all-electron
DF-CCSD(T)/cc-pCVTZ on CUDA, with exact RHF and density fitting of the correlation
integrals. The conventional controls use the same orbital basis and the
four-center integrals. Each optimizer invocation requests a single explicit
stage and includes a fresh final energy and analytic-force reevaluation.

The first allocation at the measured revision used CPU RHF. Its HF
preoptimization reached the 180-second limit and the allocation was stopped
during the next molecule. A second allocation reused the checksummed CLI and
shared library with the RHF backend changed to CUDA. The HF conventional
control converged in 298 seconds at the target stage. The separate DF pilot
tested HF, H₂O and CO on the same binaries and numerical thresholds. The 19
other cases were assigned to two disjoint DF allocations, balanced using
the estimated number of occupied orbitals and spherical AOs. Cases reaching the initial H100 time limit were assigned one longer
retry with the same binaries and numerical settings. The trans-HONO
H200 request remained unscheduled, so the retries used H100. The conventional and DF calculations
retain distinct statistics.

The spherical orbital basis uses cc-pCVTZ on C, N, O and F and cc-pVTZ on H.
The conventional force route admits all electrons, motivating the core-valence
choice. The DF fit uses cc-pwCVTZ-RIFIT on C, N, O and F and cc-pVTZ-RIFIT on
H. Its tight core-valence fitting functions accompany the all-electron
orbital treatment. [Woon and Dunning](https://doi.org/10.1063/1.470645),
[Hättig](https://doi.org/10.1039/b415208e),
[Weigend et al.](https://doi.org/10.1063/1.1445115),
[Basis Set Exchange](https://doi.org/10.1021/acs.jcim.9b00725).

- FP64, spherical representation and closed-shell RHF apply to all stages. The SCF limits are 220 cycles, `1e-12 Eh` energy change
  and `1e-10` density tolerance. The final stages additionally specify 160
  CCSD cycles, `1e-12 Eh` CCSD energy tolerance, `1e-9` CCSD residual
  tolerance and a 32 GiB correlation numeric memory budget.
- Maximum atomic force norms are `3e-4 Eh/bohr` for RHF and `2e-5 Eh/bohr`
  at the target method. Each stage is limited to 80 accepted geometry steps,
  with a 0.25 bohr maximum per-atom displacement setting. The driver allows
  180 seconds for a full RHF process. The correlated stage limit is 1,200
  seconds for initial runs and 2,400 seconds for retries.
- Stage wall time includes starting the executable, basis/system preparation,
  all analytic energy-plus-force evaluations, final verification, output
  and receipt validation. Setup/build and the independent comparison are
  outside the stage timings. The benchmark documents complete optimization
  costs on the allocated system.
- A converged receipt must have the requested basis, executed backend,
  final method, finite coordinates and energy, matching final stage energy
  and force, and a force norm within the declared threshold. The raw
  records retain nonzero exits, timeouts and flag or result failures.

## Reference data and acceptance

The [reference audit](reference-audit.md) gives the page, table, column and
value for every reference in the 22-molecule cohort. It verified all 129
original numerical entries. The audit corrects an omission in the first
report by adding the independent H₂O cc-pCVTZ angle and clarifying the two
NH₃ reference columns. The original measured geometries were replayed offline.

- Bak et al. calculated empirical structures from observed rotational
  constants and computed vibrational corrections. For 16 molecules, we use
  their empirical values in Tables II and III. Same-Hamiltonian lengths at
  cc-pCVTZ are the sum of the empirical value and the signed
  `cc-pCVTZ − Emp.` entry in Table VII. For acetylene and HCN, the empirical
  values differ between tables, so each comparison uses the baseline from
  its own table. The cc-pCVQZ controls use the absolute theoretical lengths
  from Table II. [Bak et al., J. Chem. Phys. (2001)](https://doi.org/10.1063/1.1357225).
- CH₂F₂, syn-ethenol, trans-formic acid and aziridine use the authors'
  B3LYP/SNSD-corrected semi-experimental structures in Table 4. Its dagger
  indicates inclusion of an electronic contribution in the rotational
  model. [Piccardo et al., J. Phys. Chem. A (2015)](https://doi.org/10.1021/jp511432m).
- Trans-HONO and FNO use the semi-experimental fits in Tables 1 and 4.
  For trans-HONO the chosen fit uses vibration–rotation constants from
  an all-electron CCSD(T)/cc-pVTZ cubic force field.
  [Demaison et al., J. Phys. Chem. A (2006)](https://doi.org/10.1021/jp064769v).

Bak Table III gives NH₃ angles of 107.25° in `Emp.`, 106.7° in `Expt.` and
106.36° at all-electron CCSD(T)/cc-pCVQZ. The QZ value is 0.89° below `Emp.`
and 0.34° below `Expt.`. We keep the specified empirical column in the
primary comparison. The GQC DF angle is 105.79607°. Using 106.7° only for
this parameter gives an NH₃ absolute error of 0.9039° and a 16-angle MAE of
0.3911° with a maximum of 0.9039°. [Bak et al., Table III, p. 6550](https://doi.org/10.1063/1.1357225).

Each reference parameter maps to explicit zero-based atom indices. The
summaries pool absolute deviations once per tabulated parameter, with
equivalent copies omitted and symmetry-implied constants such as 180° for
a linear chain absent from the denominator. The independent theoretical
check uses absolute gates of 0.0003 Å for matched bond lengths and 0.05°
where a matched angle is available. These gates account for printed
rounding and finite optimization tolerances. The spectroscopic comparison
reports deviations from the semi-experimental coordinates without an accuracy pass threshold.
Finite basis effects and the model used to correct rotational constants
both contribute to that comparison.

For conventional controls, PySCF 2.14.0 independently evaluates all-electron
CCSD(T) analytic gradients at the final geometry and at two central-difference
steps of 0.002 and 0.001 bohr along the orthogonal complement of rigid motions.
For DF controls, independent PySCF energies use exact RHF, density fitting
only in CC correlation, the same orbital and auxiliary shells and steps
of 0.004 and 0.002 bohr. The DF check obtains diagonal and mixed projected
second derivatives from central energy differences and a Richardson-extrapolated
projected first derivative. The acceptance requires a center energy difference
no greater than `3e-7 Eh`, a maximum atomic force no greater than
`2.5e-5 Eh/bohr`, projected eigenvalues above `1e-4 Eh/bohr²` at both steps,
and eigenvalues stable to `1e-4 Eh/bohr²`. The gradient-difference route also
requires raw Hessian antisymmetry within that threshold. These are unweighted
Cartesian curvatures.

The four checks passed. HF, H₂O and CO were checked at the DF optimized
geometries. HF was also checked at the conventional optimized geometry.
The lowest active curvature at each step is reported below. The largest
DF center energy difference from PySCF was `6.96e-13 Eh`.

| Geometry | Larger step curvature (Eh/bohr²) | Smaller step curvature (Eh/bohr²) |
| --- | ---: | ---: |
| HF, conventional | 1.264494 | 1.264477 |
| HF, DF | 1.264431 | 1.264398 |
| H₂O, DF | 0.170238 | 0.170265 |
| CO, DF | 2.428173 | 2.428116 |

The remaining optimized geometries have the native stationarity check.
The independent minimum evidence is limited to these four geometries.
The [minima archive](minima.json.xz) retains each displacement, energy or
gradient, projected curvature matrix and the original GPU geometry receipt
checksum.

## DF geometry results

The 20 converged DF cases supply 35 bond lengths and
16 angles from the fixed 22-case cohort. Their bond MAE and
maximum are 0.00273 Å and 0.02118 Å.
Their angle MAE and maximum are 0.4255° and
1.4539°. The largest bond deviation is
the FNO N–F bond (−0.02118 Å). The largest angle deviation is
the NH₃ H–N–H angle (−1.4539°).

The successful CCSD(T) stages on H100 used 2–11
accepted optimization steps and 51.83–2,180.44 s
(median 351.48 s). The largest accepted final force norm
was 1.954e-05 Eh/bohr. The table gives one successful run per molecule
where available. The separate stage receipts retain the initial timeout
even when the retry converged.

| Molecule | RHF steps/evals, s | DF-CCSD(T) steps/evals, s | Bond MAE/max Å (n) | Angle MAE/max ° (n) |
| --- | ---: | ---: | ---: | ---: |
| hf | 3/4, 3.17 | 3/4, 68.75 | 0.00020/0.00020 (1) | — |
| h2o | 2/3, 2.87 | 3/4, 62.84 | 0.00094/0.00094 (1) | 0.8350/0.8350 (1) |
| co | 3/5, 5.67 | 4/6, 226.59 | 0.00417/0.00417 (1) | — |
| n2 | 4/6, 4.82 | 4/6, 204.98 | 0.00291/0.00291 (1) | — |
| f2 | 3/4, 3.57 | 4/5, 173.63 | 0.00078/0.00078 (1) | — |
| hcn | 5/7, 8.11 | 5/7, 352.60 | 0.00251/0.00353 (2) | — |
| hnc | 5/7, 7.18 | 5/7, 352.07 | 0.00224/0.00364 (2) | — |
| ch4 | 2/3, 2.52 | 2/3, 51.83 | 0.00163/0.00163 (1) | — |
| nh3 | 6/7, 5.12 | 5/6, 116.01 | 0.00101/0.00101 (1) | 1.4539/1.4539 (1) |
| hno | 8/9, 12.38 | 7/8, 443.09 | 0.00289/0.00325 (2) | 0.3384/0.3384 (1) |
| hof | 7/8, 9.61 | 5/6, 350.88 | 0.00101/0.00118 (2) | 0.2764/0.2764 (1) |
| co2 | 3/4, 4.67 | 3/4, 362.17 | 0.00360/0.00360 (1) | — |
| c2h2 | 5/7, 5.62 | 6/8, 314.11 | 0.00252/0.00301 (2) | — |
| ch2o | 9/10, 9.18 | 8/9, 392.86 | 0.00198/0.00239 (2) | 0.5506/0.5506 (1) |
| trans-n2h2 | 7/8, 6.12 | 7/8, 325.06 | 0.00302/0.00484 (2) | 0.5122/0.5122 (1) |
| c2h4 | 5/6, 4.79 | 7/8, 535.22 | 0.00227/0.00319 (2) | 0.0829/0.0829 (1) |
| fno | 6/8, 12.23 | 8/9, 885.26 | 0.01404/0.02118 (2) | 0.2172/0.2172 (1) |
| trans-hono (retry) | 9/10, 12.98 | 11/13, 1687.81 | 0.00123/0.00287 (3) | 0.2331/0.4612 (2) |
| ch2f2 | 5/6, 6.18 | 6/7, 1134.25 | 0.00123/0.00166 (2) | 0.3534/0.6460 (3) |
| syn-ethenol (retry) | 14/15, 18.90 | timeout, 2400.32 | — | — |
| trans-formic-acid (retry) | 10/11, 13.28 | 11/13, 2180.44 | 0.00226/0.00287 (4) | 0.3382/0.5524 (3) |
| aziridine (retry) | 13/15, 9.88 | timeout, 2400.24 | — | — |

Against the 24 available conventional Table VII bond
lengths at the same orbital basis, the DF geometries have MAE
0.00003134 Å and maximum 0.00011543 Å.
The largest absolute difference is for the F₂ bond. This comparison includes the effect of
density fitting along with convergence tolerances and rounding of the
literature values. The exact-method tolerance applies to the conventional
controls below. [Bak et al., Table VII](https://doi.org/10.1063/1.1357225).

Demaison et al. also found that conventional all-electron CCSD(T) gave
a short N–F bond for FNO at aug-cc-pCVQZ (1.5120 Å versus their
1.51681 Å semi-experimental result). Their different orbital basis and
correlation treatment leave the size of the discrepancy in this campaign
unresolved. [Demaison et al., Table 4 and Sec. 4](https://doi.org/10.1021/jp064769v).

The retained DF failures are listed below.

- `trans-hono` in batch `g` recorded `timeout` after 1,200.28 s. Final cohort status is converged.
- `syn-ethenol` in batch `g` recorded `timeout` after 1,200.26 s. Final cohort status is unresolved.
- `syn-ethenol` in batch `k` recorded `timeout` after 2,400.32 s. Final cohort status is unresolved.
- `trans-formic-acid` in batch `f` recorded `timeout` after 1,200.30 s. Final cohort status is converged.
- `aziridine` in batch `f` recorded `timeout` after 1,200.41 s. Final cohort status is unresolved.
- `aziridine` in batch `l` recorded `timeout` after 2,400.24 s. Final cohort status is unresolved.

## Conventional implementation controls

The four-center CCSD(T) control passed the declared literature tolerance for
both available optimized bond lengths and the H₂O angle. The HF result was 0.91668932 Å
against 0.9167 Å from Table VII. The H₂O O–H result was 0.95842820 Å against
0.9584 Å. The two errors have MAE 0.00001944 Å and maximum 0.00002820 Å.
[Bak et al., J. Chem. Phys. (2001)](https://doi.org/10.1063/1.1357225).

Bak Table III lists the H₂O `Emp.` angle at 104.51° and the CCSD(T)/cc-pCVQZ
angle at 104.22°. Kahn et al. provide a same-table all-electron CCSD(T)
basis comparison, with 103.6839° at cc-pCVTZ and 104.2253° at cc-pCVQZ.
The archived GQC conventional cc-pCVTZ angle is 103.675830°, or −0.008070°
relative to the same-method TZ literature value. The GQC DF result is
103.675017°, or −0.008883° relative to that conventional TZ value.
The Kahn TZ-to-QZ shift of +0.5414° accounts for about 65% of the 0.8350° DF
deviation from 104.51°. At QZ the literature angle still lies 0.2847° below
the empirical value. This supports basis incompleteness as a major contributor
to the TZ deviation. [Bak et al., Table III, p. 6550](https://doi.org/10.1063/1.1357225),
[Kahn et al., Table I, p. 074106-5](https://escholarship.org/uc/item/4g45z2kz).

The corresponding DF and conventional optimized geometries differ by
0.00000869 Å for HF and 0.00001229 Å for the H₂O bond. The H₂O angle
differs by 0.00081°. These paired results concern the two shared
geometries and their specific basis and thresholds.

| H100 case | CUDA RHF steps/evals, s | Conventional CCSD(T) steps/evals, s |
| --- | ---: | ---: |
| HF | 3/4, 3.47 | 3/4, 298.26 |
| H₂O | 2/3, 3.26 | 3/4, 582.71 |
| CO | 3/5, 5.67 | Timeout, 1,200.40 |
| N₂ | 4/6, 4.67 | Interrupted when allocation was stopped |

The two successful conventional geometries have semi-experimental bond
MAE 0.00056944 Å and maximum 0.00092820 Å. The H₂O angle differs by
0.83417°. The allocation stopped after the HF and water controls and CO
timeout to shift compute to the DF cohort. This left the other
conventional cc-pCVTZ cases and the planned HF and CO cc-pCVQZ controls
unattempted.

## Compute and withdrawn attempts

All jobs used project `CQ-NexChem`. GPU jobs requested the highest
non-preemptible setting, priority 4, and qz returned `NORMAL 20`.
The two CPU jobs requested priority 10 and qz returned `HIGH 35`.
The GPU rate was 1 point per allocated instance-hour. The table uses
scheduler allocation events to estimate GPU card-hours and points.
Where the pod-deletion event was unavailable, it uses the last exit event
or the entrypoint duration. The CPU rows use the four-core allocation times the entrypoint duration.
The estimated total is 7.146 GPU card-hours and 7.146 points, including
0.093 card-hours and points from the withdrawn source. The two CPU jobs
used 1.403 allocated CPU-hours and zero points. Per-job invoice totals
were unavailable.

| qz job | Resource | Card-hours | CPU-hours | Estimated points |
| --- | --- | ---: | ---: | ---: |
| `gqc-3166-rese-1125-a` | H100 | 0.031 | — | 0.031 |
| `gqc-3166-rese-1125-b` | H100 | 0.062 | — | 0.062 |
| `gqc-3166-rese-1125-22f-c` | H100 | 0.829 | — | 0.829 |
| `gqc-3166-rese-1125-22f-d` | H100 | 0.590 | — | 0.590 |
| `gqc-3166-rese-1125-22f-e` | H100 | 0.110 | — | 0.110 |
| `gqc-3166-rese-1125-22f-f` | H100 | 1.569 | — | 1.569 |
| `gqc-3166-rese-1125-22f-g` | H100 | 1.484 | — | 1.484 |
| `gqc-3166-rese-1125-22f-h` | H200 | 0.000 | — | 0.000 |
| `gqc-3166-rese-1125-22f-i` | H100 | 0.480 | — | 0.480 |
| `gqc-3166-rese-1125-22f-j` | H100 | 0.637 | — | 0.637 |
| `gqc-3166-rese-1125-22f-k` | H100 | 0.678 | — | 0.678 |
| `gqc-3166-rese-1125-22f-l` | H100 | 0.675 | — | 0.675 |
| `gqc-3166-rese-1125-22f-min-hf` | Xeon Max 9462 | 0 | 0.131 | 0 |
| `gqc-3166-rese-1125-22f-min-df` | Xeon Max 9462 | 0 | 1.272 | 0 |

The withdrawn `d154546d2` jobs were included in the compute accounting.
The first failed before launching a molecule because a CLI build
dependency was missing. The second rejected the unbundled cc-pVDZ name in
all 22 preliminary requests. No old-revision geometry reached CCSD(T).
The first `22f72d069` allocation built the production CLI and then
timed out on the HF CPU RHF stage. It was stopped during the next CPU
preliminary request. The H200 retry request could not be scheduled and
was withdrawn without an allocation. The task's GPU jobs and CPU HPC
entries were cleaned up after the receipts were collected.

## Reproduction

Run the offline replay with Python 3.11 or later.

```bash
python benchmarks/results/ccsdt-rese-20261125/replay.py --output /tmp/3166-summary.json
```

The native acquisition uses the executable and its shared library from one
clean CUDA build of the measured revision. Stage a clean archive of that
commit. For an archive created with macOS tar, use `COPYFILE_DISABLE=1`
and ensure there are no `._*` files before checking the native source
identity. The [H100 build recipe](build-qz.sh) takes the staged source directory, the frozen science environment root and
the staged Ubuntu CLI dependency root. The [qz entrypoint](launch-qz.sh) then
checks the payload checksums and launches the acquisition.
The production allocation uses the [verified-binary entrypoint](launch-reuse-qz.sh). For an existing
source-verified CLI, the equivalent driver is:

```bash
python benchmarks/results/ccsdt-rese-20261125/run.py \
  benchmarks/results/ccsdt-rese-20261125 \
  /path/to/build/generativeqc /path/to/new/output \
  --df-auxiliary-basis benchmarks/results/ccsdt-rese-20261125/cc-pwcvtz-rifit.json
```

Run the same-basis curvature check with [check_minima.py](check_minima.py)
or [check_df_minima.py](check_df_minima.py), passing the payload directory, the native output directory and a new output
directory. Supply `--auxiliary` with the retained RI basis for DF. The [CPU qz entrypoint](launch-minima-qz.sh) verifies the allocated
CPU model before invoking the Python driver. Its interpreter and package
versions are in the recorded provenance.

[records.json.xz](records.json.xz) retains the native UTF-8 stage receipts,
input coordinates, requests, stdout, stderr and available optimized XYZ
files. [summary.json](summary.json) gives signed errors, per-stage wall
times, and the pooled deviations. [provenance.json](provenance.json) binds
the native source, built and loaded binaries, payloads, hardware, compute
allocations and retained file hashes.
The personal shared-mount prefix is replaced by `${GQC_SHARED}` in the
published text. Both the original and publication hashes are recorded, and
the original numerical tokens are preserved. The withdrawn-build records
are replayable on their own with `--records withdrawn-d154546d2.json.xz`
from this directory.
