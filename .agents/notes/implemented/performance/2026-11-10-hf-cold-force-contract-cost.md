# Diagnosis: the README HF cold gap follows strict force acceptance

Status: investigated; retain numerical contracts, no production-code change
Date: 2026-11-10

## Question and boundary

The Oct26 README cohort won all cold comparisons; the fresh master805de425a
cohort lost Direct HF48/HF96. Investigate an unintended implementation regression
before proposing a fix. Do not restore the former stopping rule just to recover
the timing rank, relax independent force gates, or replace individual homepage
points with selected reruns.

## Source-level diagnosis

The public Standard controls remain energy1e-9, density1e-5, screening1e-12.
The old force owner admitted the relaxed Standard AO commutator ceiling1e-5.
The Nov9 force-stationarity repair refuses that energy-only relaxation for
analytic forces and retains the strict1e-8 ceiling. On HF48 this changes ten
ordinary cold iterations/Focks to fourteen; the seven initial-guess Focks remain
separate. The independently isolated repair and its failed original control are
in `benchmarks/results/hf-force-stationarity-20261109/`.

HF96's screened cold proposal still executes eight cheap FP64 applications.
The qualified strict physical target tail needs seven, not the historical three,
so total target/provisional Focks rise eleven to fifteen. See the Nov7 strict-tail
qualification note. Neither all FP64 Focks nor just SCF iteration counts have
equal cost: target exchange is much more expensive than screened proposals.

Historical Oct26 arrays independently re-subtracted against the authenticated
tight oracles give worst cold/warm maximum force errors1.818375e-7(HF48) and
3.019337e-7(HF96). Both fail the additional unchanged1e-7 diagnostic, although
both pass the looser README common gate1e-4. Do not mislabel these as failures of
the README common gate, or claim their faster times qualify today's force owner.

## Fresh same-card checks

Slurm17059 uses one RTX5090 on n1, finite25min,4CPUs/24GiB and assigned visibility0.
GPU UUID: `GPU-45f177f9-33ac-0e4e-f473-2beeaf6d413d`.
The untouched source/ELF are exactly the Nov10 homepage image:

- source identity `a305bde4a845e554e5f04ca4ab593b26166b7e31902635dcd5a900194532aea3`;
- ELF SHA256 `21838a057f9b6f3ce8b025ec94e289ce29ff30827b1afd3da23f95f1fa8b2741`.

No new native compilation is needed. Math threads1, native reference imports
rejected, same geometry/basis/electronic state and default public controls.
GPU4PySCF1.9.0 is fixed within this acquisition. Each fresh process measures the
complete constructor/preparation/SCF/analytic-force/host-export cold endpoint
and three own-cold-density frozen warms. Three engine-pair rounds alternate
first-engine order; a separate same-image default/ordinary/ordinary/default
sequence tests the existing explicit cold-screening0 rollback. That switch also
changes bounded-page admission, so it does not isolate screening arithmetic alone.

| Population | Processes | Cold median [min,max], s | Native cold Focks |
| --- | ---: | --- | ---: |
| HF48 default | 3 | 7.916147 [7.904282,7.924686] | 14 |
| HF48 GPU4PySCF default | 3 | 8.119237 [8.093592,8.312828] | — |
| HF96 default | 5 | 16.874977 [16.821903,16.912032] | 15 |
| HF96 ordinary rollback | 2 | 22.267688 [22.230513,22.304862] | 14 |
| HF96 GPU4PySCF default | 3 | 11.774789 [11.770840,11.854919] | — |

The default HF96 median within the four-process ABBA alone is16.908515s,
24.1% below its ordinary rollback. Automatic admission has not stopped working.
All native receipts retain seven initial-guess Focks, zero post-SCF Focks and
one skipped final rebuild: no duplicate final physical Fock explains the gap.
Default HF96 remains8+7; every frozen warm remains one Fock.

The default HF48 and HF96 times match the previously qualified corrected-owner
7.948143s and strict-tail16.878195s scale. No new extra-cost regression is isolated.
HF48's original single-observation0.6% loss is not a stable ranking: all three
fresh pairs favor native here. HF96 remains consistently slower than the default
comparison engine; retaining that result is mandatory.

There are17 processes/68 endpoints, including one explicitly separate traced
HF96 process. Every native endpoint(44) passes independent absolute E<1e-8 and
Fmax<1e-7; worst force error1.651594e-9. The tighter targets are authenticated
from the existing A100 shared-XY capsule; HF48 is CPU full-Fock and HF96 retains
its audited tighter-producer binding. The GPU4PySCF default comparison processes
are not those independent tight oracles: their worst Fmax1.376761e-7/1.534040e-7
fails this stronger diagnostic but passes the README common gate.

The separate valid diagnostic observes seven uncaptured target Fock owner calls
with total host submission/completion span9.019s, median1.289s. Four additional
such calls have a roughly5.16s cost scale, consistent with the historical HF96
increase. This is not an exact cross-cohort latency decomposition or GPU-exclusive
time. Captured component CUDA-event values must not be treated as complete replay
times. Intrusive tracing is excluded from every timing median above.

## Decision and evidence

No speculative numerical or dispatch patch is promoted. Existing screened-tail
and force-stationarity correctness repairs stay intact. A future optimization
must reduce complete strict J/K cost or improve convergence while retaining
physical target acceptance, independent strong observables, bounded fallbacks,
finite iteration allowance and every warm/lifetime guard. The previously rejected
full DIIS-ring transfer, four-step handoff and384-AO proposal expansion are not
new bug fixes and should not simply be repeated.

All drivers, original vectors/receipts/logs, reviewed references, SHA bindings,
full-vector re-subtraction and ordinary/default results are retained in ignored
`.artifacts/hf-cold-diagnosis-20261110/`. Raw acquisition also remains at
`n1:/data/jzzeng/readme-rtx5090-20261110-805de425a/.artifacts/benchmarks/hf-cold-diagnosis-20261110/`.
The local `evidence.tar.xz` SHA256 is
`e3d61a6e6eafe37f4e909d917e522822d4d07aeee65cd19ee5ab83ae2a490261`.
This is local evidence, not an externally published artifact or full-matrix rerun.
No tracked benchmark evidence is removed, compacted or added; its cap is unchanged.
Both normal and `python -O` historical force-capsule replays agree, and the new
array reducer authenticates all17 raw process receipts before scientific scoring.

## References

- `src/scf/convergence_limits.hpp` and `src/scf/cuda_rhf.cpp`.
- `2026-11-07-cold-hf-strict-screened-tail.md`.
- `../numerics/2026-11-09-hf-force-stationarity.md`.
- `benchmarks/results/a100-shared-rys-xy-20261108/replay.py`.
