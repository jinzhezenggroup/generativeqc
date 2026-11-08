# SM80 shared Rys x/y: measured gain, independent default promotion blocked

This is a **draft promotion candidate**, not a claim that A100 beats every
non-CBS README endpoint. In particular, HF Direct still trails GPU4PySCF.
The additional absolute `1e-7` force gate fails at default HF48 in incumbent
and candidate. Native parity, strict-request success, and a looser benchmark
contract must not turn that failed default diagnostic into a pass.

## Reproduce

```bash
python benchmarks/results/a100-shared-rys-xy-20261108/replay.py
python -O benchmarks/results/a100-shared-rys-xy-20261108/replay.py
python benchmarks/results/a100-shared-rys-xy-20261108/replay.py --require-promotion
PYTHONPATH=python:. pytest -q tests/python/test_sm80_shared_rys_evidence.py
```

The first two commands reproduce all observations, including failures. The
third intentionally exits nonzero while independent default promotion is
blocked. Tests require that failure to remain visible. `cohort.json` binds
the source/ELF and exact retained raw UTF-8 bytes in `observations.json.xz`;
`summary.json` is derived, never a replacement for raw populations.

## Protocol and implementation

Every size/card has four fresh native processes in ABBA order. Cold includes
constructor, preparation, SCF, analytic forces and host export. Each process
then makes three warm calls from its own frozen cold density. Final controls
compare explicit shared-XY `0` with unset/automatic on the **same final ELF**;
weighted Z is unset. Default energy/density/screening tolerances remain
`1e-9/1e-5/1e-12`. Strict HF48 implementation diagnostics explicitly request
`1e-12/1e-10/1e-14`, and are never labeled default-performance observations.

Only the exact SM80 profile is a candidate automatic default. SM90 retains
its opt-in and SM120 retains the incumbent. The existing compiler recurrence,
small-mask private fallback, output scatter and all ownership/resource refusals
remain. Weighted Z stays SM90-only. No GPU product-name check, CPU seed,
reference density or oracle import enters timed native production.

The measured commit is `62d9f2da7`, with master `74d06855c` integrated as
`76343fcc4`. "Final" in this capsule names that frozen cohort, not a later PR
head. Master subsequently advanced to `b29247c60` and was integrated as
`ac7915d27`; that post-measurement HF/reference change has host integration
checks but **no new source-bound GPU qualification** here. Its results must
not be relabeled as this cohort, and the current head needs fresh qualification
before promotion in addition to resolving the retained HF48 blocker.

Measured final source identity is
`33db42ef97a4ce677b614a0900e18ba395c5837b7021c47df7f1e4ca1664d817`,
ELF `60f40b1565c6147fec5cce0bcc77f16338a41eadcc436d8c8f675d08d1871c17`.
Slurm build16816 succeeds with the verified compiler cache; final actual5090
job16818 retains the unchanged default-owner route. LSF229002 is the final
authenticated setup and229003 is the exclusive A100 endpoint allocation.
Scheduler-assigned visibility is preserved; no real GPU work bypasses it.

The final same-image cohort retains these cold medians:

| Card/atoms | Control cold/s | Candidate cold/s | Reduction |
|---|---:|---:|---:|
| A100/96 |16.163515754|14.649752516|9.37%|
| A100/48 |6.096549792|5.737883616|5.88%|
|5090/96|16.905733509|16.886990728|0.11%|
|5090/48|6.299592309|6.299205655|0.006%|

Every paired warm passes3%, and all120 native sample pairs per size pass
E`1e-8`/F`1e-7`. The source-bound `summary.json` also reports initial-guess
Fock work separately from target-owner work and preserves invalid quartet
counters as invalid. The original/displaced/restored independent HF96 checks
pass on both cards. Slurm16819 verifies every original noncache source byte,
absence of candidate-only files, original source identity and original ELF
after restoring the borrowed build tree.

## Earlier endpoint cohort, not final-source timings

Historical source `6f3ea6ea`, ELF `5c222d02`, uses the actual new default
against the source-bound incumbent image (`50ca688e`/`29770494`). A100228864
and5090 16811 retain all populations, not a favorable subset:

| Card/atoms | Control cold/s | Candidate cold/s |
|---|---:|---:|
| A100/96 |16.142927356|14.624377469|
| A100/48 |6.068829596|5.702018391|
|5090/96|16.903542409|16.895718804|
|5090/48|6.325019540|6.339962678|

All120 native sample pairs per size pass absolute E`1e-8`/F`1e-7`. Every
paired warm passes the unchanged3% guard. Target Focks/iterations remain15
at96 and10 at48, with one-step warms. Invalid quartet counters remain invalid,
not evidence of zero work. No new full64 census or changed overall win count
is asserted. These historical timings cannot be relabeled as final-source data.

## Scientific blocker and retained failures

The authenticated independent HF96 oracle228320 passes both cards' retained
original geometries; displaced/restored HF96 checks also pass. Independent
Libcint synthetic cases and memcheck/racecheck/synccheck pass on A100228865
and5090 16809, with zero errors/warnings. A successful synthetic job exit does
not imply its separately checked HF48 default observables pass.

Default HF48 differs from the independently converged CPU reference by about
`1.8184e-7` in maximum force. Candidate/incumbent differences are tiny: this is
not a shared-XY numerical drift. CPU-only oracle16815 uses full Focks, integral
screening `1e-15`, energy `1e-11` and orbital gradient `1e-10`; its actual final
gradient norm is `4.448e-11`. Its energy is `-1215.6178506877025` Eh.
The old GPU reference differs by only `6.531e-10` in force, excluding reference
underconvergence as the explanation. Strict native A100 and5090 HF48 need21 target
iterations and reach about `3.8e-11` maximum force error, compared with10
iterations at defaults. This establishes a default-control accuracy boundary,
not which individual stopping or screening control causes it: all three were
tightened in this diagnostic. It does not license default promotion at a gate
that defaults fail; isolate those controls before choosing a numerical fix.

Retained adverse references include CPU16813's pre-computation import failure,
CPU16814 and GPU228912's100-cycle nonconvergence, and GPU228940's metadata
export failure after converged full-Fock SCF/gradient computation. GPU4's
`get_grad` requires an explicit Fock; its failed metadata call remains exit1.
The README's pre-existing size-normalized energy and intensive Fmax`1e-4` /
Frms`3e-5` gates are a different contract. Neither contract is weakened here,
and passing one cannot erase failure at the stronger diagnostic.

The proposed default remains blocked in the default-promotion inventory.
Resolve that scientific boundary without hiding additional work or regressing
the protected5090 route before merging the promotion. See issue2976 and the
[decision proposal](../../../.agents/notes/proposed/2026-11-08-sm80-shared-rys-xy-qualification.md).
