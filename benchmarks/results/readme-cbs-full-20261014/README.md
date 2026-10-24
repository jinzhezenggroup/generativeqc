# Full DF-HF/RI-JK CCSD(T)/CBS: complete QM7 C1–C7 E+F

See [complete timings and semantic work](timings.md). This historical campaign
was replaced on the homepage by the
[matched Standard/Fast campaign](../readme-cbs-fast-20261020/README.md).
Its original plot, receipts, source patch and manifest remain recoverable from
the published Git revision pinned in [snapshot.json](snapshot.json), with every
original byte count and SHA-256 recorded. Recovery of all nine original files
and the original scientific verifier was checked before retiring these payloads
from the current tree. No measurements or scientific gates have been changed.

## Scientific endpoint

This campaign uses native CUDA **DF-HF/RI-JK throughout**, including signed
orbital-response actions, the fresh physical residual and analytic nuclear
derivatives. It does not substitute a DF-HF guess for an exact-HF final method.
The force includes fitting-center, metric, orbital and Pulay response.

- Spherical cc-pVDZ/cc-pVTZ MP2 correlation extrapolation, exponent 3.
- DF-CCSD(T)−DF-MP2 correction in cc-pVDZ; exponential HF extrapolation,
  alpha 1.63. All-electron closed-shell FP64; no frozen core.
- Each basis group uses **cc-pVXZ-JKFIT for both JK and correlation**. Separate
  JK-fit and correlation-fit records are not yet supported. This differs from
  the earlier exact-HF/RI-correlation campaign; no identical-model speedup is
  claimed between the two.
- Two sequential native groups: DZ owns CCSD(T), MP2 and HF; TZ owns MP2 and HF.
  Same-basis SCF/RI-JK ownership survives all companion response work. The
  correlation MO source is shared by its CC/MP2 consumers, but its construction
  does not yet borrow the SCF AO factors. Different orbital bases do not share
  orbitals or factors. Individual companion times overlap and must not be summed.
- Explicit **32-GiB total numerical-capacity budget**. This includes host and
  device owners and conservative retained allowances; it is not measured VRAM
  usage or a claim of 32 GiB of free GPU memory.

The fixed molecules, original float32 Bohr coordinates, dataset checksum and
indices are identical to the [earlier campaign](../readme-cbs-20261013/README.md#fixed-molecules):
QM7 zero-based indices **0, 1, 8, 48, 183, 293, 5604**. No recentering or
optimization is performed. Basis identities, AO/auxiliary counts and complete
component results are retained in the receipts.

## Measurement and provenance

One API-cold observation per molecule on **node1 RTX 5090**, driver 580.95.05,
600-W power limit, dual AMD EPYC 9354 host, Release CUDA 12.9.1/sm120 build.
Finite Slurm allocations request one GPU and four logical CPUs; numerical
thread pools use one thread. Other jobs may run on other GPUs of the host.
Clocks are not locked; these are individual observations, not latency statistics.

Each interval covers calculator construction through **returned composite
energy and analytic host forces**: both basis groups, SCF, CC, triples, Lambda,
orbital/nuclear response and in-call cleanup. It excludes Python startup,
preloaded DSO loading, dataset/basis snapshots, JSON and independent references.
No oracle output initializes the timed native calculation. There is no retained
SCF plan from a preceding call and compiler caches are not cleared.

Measured production source: `b400ac1fb9ba232d29e0069ea2c759ac5b3bc926` plus patch
`023d0739ada6df92ceb8acd2da7af280a62afceed8a63a7fb91f12351d2fac16`.
Implementation PR: [#2459](https://github.com/jinzhezenggroup/generativeqc/pull/2459).
The qualified library is
`30fc778fe44fd89100f5159e174b34faf5fb61a5d598de8ae73b753d47e6a01c`.
It is **not a measurement of later master**. The production PR identifies the
delivered implementation; later comments/docs do not change this frozen binary.
The build uses verified sccache 0.16.0, eight AOT shards, direct device linking
and fast compile disabled. The optional CPU linear algebra provider is scalar.

The first campaign used 24 GiB: C1–C6 completed; C7 refused Lambda's complete
numerical budget after 606.662 s. This was **not** a completed E+F time or a
measured GPU OOM. The retained RI-JK allowance alone conservatively reserves up
to 8 GiB across host/device owners. The primary series reruns **all seven**
molecules with 32 GiB, not just a favorable replacement for one point.
The recoverable `supplement.json.xz` retains the 24-GiB endpoint records and
failure. Failed runtime imports before native work are also separate from the
primary timing series; no partial outcome acquires a completion time.
Intermediate exact-reference whole-shell observations are retained in the
supplement as distinct source-matched controls, not full-DF primary points.

C7 completes in **1794.984 s (29.92 min)**: **1656.495 s** in the DZ
correction group and **138.488 s** in the TZ MP2/HF group. The former is **92.3%**
of the endpoint; the remaining bottleneck is no longer the large-basis exact
RHF response. These group timers do not isolate triples from CC/Lambda.
The table reports CC and response work, but leaves DZ SCF iterations unknown:
the frozen companion ABI exposes the primary CC counter there, not HF work.
TZ's MP2-primary counter does report its SCF iterations. No missing counter is
invented or counted once per companion.

## Scientific qualification and limits

Before any primary timing, six native GPU tests passed independent reconverged
PySCF energy/directional-force gates. They cover H2, water DZ/TZ, f orbital and
g auxiliary shells, Cartesian metric rank truncation, DF-CCSD(T) companions and
full CBS assembly. The 160-test legacy response/DF-MP2 GPU regression passes;
H2/TZ water pass Compute Sanitizer memcheck and synccheck with zero errors.
These method-level tests do not independently certify every large molecule.

Only the **C2 ethane** point has a separate complete independent oracle in this
series. Filled markers require all five constituents and the composite to pass
energy (5e-8 Eh), translation (1e-6 Eh/Bohr), and a seed-709 normalized force
direction at two independently reconverged steps, 1e-4 and 3e-5 Bohr
(5e-6 Eh/Bohr). Open markers mean independent accuracy **unchecked**, not
passing. Every plotted point must still return converged, finite, CUDA E+F for
all constituents. Missing/failed E+F never contributes a plot time.

This is not an independent full-vector force check, an error estimate against
the CCSD(T) basis-set limit, or a benchmark of direct TZ CCSD(T). The timings do
not support an inference that TZ CCSD(T) would fit or run efficiently.

A separate ethane check isolates the **HF fitting approximation**, holding the
JKFIT correlation auxiliaries fixed while switching only exact HF versus RI-JK.
The composite changes by **+4.315e-5 Eh**, with maximum/RMS force changes
**6.614e-6 / 2.959e-6 Eh/Bohr**. Individual constituent force differences reach
**4.912e-5 Eh/Bohr**; composite cancellation must not conceal them. These are
model differences, not implementation errors, and compare neither to true CBS
nor to the older RI-correlation protocol. All constituent differences and the
exact-reference result are retained in the supplement.

## Reproduce

Use the full-DF production implementation and explicitly import local basis
records; PySCF is only a benchmark snapshot/reference dependency. Supply the
pinned local QM7 file and the frozen/source-matched native library:

```bash
export GENERATIVEQC_LIBRARY=/absolute/path/to/libgenerativeqc.so
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_PRELOAD="$GENERATIVEQC_LIBRARY" LD_BIND_NOW=1 GENERATIVEQC_PROFILE=off
export QM7_DATASET=/absolute/path/to/qm7.mat
export README_CBS_SOURCE_REVISION=<actual-source-revision>
export README_CBS_HF_REFERENCE=df README_CBS_BUDGET_MIB=32768
export README_BENCHMARK_OUTPUT=.artifacts/readme-cbs-full
export README_BENCHMARK_SIZES=7 README_BENCHMARK_POINT_TIMEOUT=3500
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=01:00:00 \
  bash benchmarks/run_readme_cbs_benchmarks.sh native
```

Run each desired point in an appropriate finite allocation without changing
Slurm's `CUDA_VISIBLE_DEVICES`; exclude node3. Run `reference` separately, with
the same explicit model/budget, to produce independent CPU oracle receipts.
The original qualification driver is retained as `qualification-driver.py.xz`;
the generic runner adds provenance and equivalent protocol selection outside
the timer. `qualified-source.patch.xz` reconstructs the frozen source on its
recorded base. Endpoint records are normalized for the shared renderer with
their original hashes and scientific fields retained, not re-timed or relabeled
as another model. Full original run bundles, traces and qualification logs stay
in ignored local artifacts, not Git. The manifest records the local archive and
member hashes; no external archive or release has been published.

Restore and replay the original figure without CUDA or PySCF. The recovery tool
does not fetch implicitly; a shallow clone must first fetch the revision named
in `snapshot.json`:

```bash
python tools/restore_retained_evidence.py \
  --manifest benchmarks/results/readme-cbs-full-20261014/snapshot.json \
  --all --output .artifacts/readme-cbs-full-history
PYTHONPATH=.:python python \
  .artifacts/readme-cbs-full-history/benchmarks/results/readme-cbs-full-20261014/verify.py
PYTHONPATH=.:python python -m tools.render_readme_cbs \
  --summary .artifacts/readme-cbs-full-history/benchmarks/results/readme-cbs-full-20261014/summary.json.xz \
  --output .artifacts/readme-cbs-full-replay
```

When the retained originals are available locally, add
`--raw-archive .artifacts/cbs-readme/full-df-originals.tar.xz` to `verify.py` to
verify every original byte and bind the normalized samples to the raw receipts.
