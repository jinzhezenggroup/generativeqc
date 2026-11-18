# Decision: admit four-root auxiliary-g DF derivative classes to Rys on sm_90

Status: implemented
Date: 2026-11-18

## Problem

The README DF auxiliary (GPU4PySCF's default fit for def2-SVP) has g shells.
`rys_available` excluded every C=4 class: the capability bit encoding
`16A+4B+C` cannot hold C=4, and the shared-axis moment builder was only asked
for its s/p/d/f domain. Every `(ab|g)` derivative therefore ran the polynomial
closed form. On H100 water-96 DF-HF, classes 004/104/114/204 took 110 ms of the
378 ms three-center derivative contraction, at 25-48 ns per shell task against
2.6-3.4 ns for the Rys auxiliary-f classes.

## Decision

- **Capability:** add `AUXILIARY_G_RYS_SHELL_CLASSES` for 004/104/114/204.
  Their first derivatives need at most four roots, the already qualified DF quadrature.
- **Moments:** emit their shared-axis moments through the existing
  `auxiliary_g_derivative` domain of `build_df_axis_moment`. The moment
  recursion is unchanged; only the domain check differs.
- **Encoding:** the capability bit becomes `15A+5B+C` (bra s/p/d, auxiliary
  s..g, below 45 bits).
- **Tuning registry:** admits auxiliary g, with orbital shells still s..f.
- **Promotion:** the classes are promoted only in a new `sm_90` exact-target
  profile, the 23 portable rows plus these four. Other targets are unchanged.
- **Five-root classes:** 214, 224 and 223 keep polynomial until a five-root
  quadrature is qualified.

## Evidence

`benchmarks/results/h100-df-aux-g-rys-20261118/`. The same binary was run with
the auto and candidate policies before promotion.
- **Warm timings:** water-96 DF-HF 0.85 → 0.75-0.77 s, B3LYP 1.70 → 1.60 s and
  PBE0 1.69 → 1.59 s. At 48 atoms, HF 0.182 → 0.160 s.
- **Force agreement:** same-phase forces agree within 4.3e-13 Eh/Bohr.
- **Independent oracle:** PySCF DF-RHF on water-12 with the README auxiliary
  gives force errors of at most 8.8e-12, against a 1e-8 gate.
- **Memcheck:** clean.

## Revisit when

- A five-root DF quadrature meets the DF accuracy contract; 214/224/223 are the
  next 34 ms.
- Another target is measured. A portable promotion needs its own cross-device
  evidence.
