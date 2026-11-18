# H100: Rys lowering for four auxiliary-g DF derivative classes

The README DF auxiliary (GPU4PySCF default fit for def2-SVP) carries g shells on
oxygen. On H100 every `(ab|g)` first-derivative class ran the polynomial
closed-form lowering (254-255 registers, 256 resident threads per SM).
In a water-96 DF-HF warm call, classes 004/104/114/204 took 21, 45, 28 and
15 ms (110 ms of 378 ms of three-center derivative contraction). By
comparison, Rys auxiliary-f classes cost 2.6-3.4 ns per shell task, against
25-48 ns for the polynomial g classes. GPU4PySCF's whole fused DF ip1
contraction is 28 ms per call on the same device.

Those four classes need at most four roots, the already qualified DF
quadrature. The `sm_90` exact-target profile now adds them as Rys/compact on top
of the unchanged 23 portable rows. 214/224/223 need five roots and keep
polynomial. Other targets keep the portable profile.

## Measurements

Fresh consecutive processes on one physical H100 80GB in qz Job
`gqc-h100-opt-1118-qa`, one build (identity `0c5dcf62…`). In that build the
classes were an unqualified `sm_90` candidate over the qualified portable
baseline, so `GENERATIVEQC_DF_SHELL_POLICY=auto` and `candidate` give an
interleaved same-binary A/B. The committed manifest promotes exactly that
candidate row set. That build also contained the then-unmerged #3115/#3116/#3118
changes, which are common to both arms.

| Case (DF) | auto warm (s) | candidate warm (s) | auto cold | candidate cold |
| --- | --- | --- | --- | --- |
| HF 96 | 0.847, 0.846 | 0.769, 0.747 (traced 0.758) | 3.520, 3.177 | 3.134, 3.086 |
| B3LYP 96 | 1.701 | 1.600 | 5.386 | 5.317 |
| PBE0 96 | 1.686 | 1.593 | 5.363 | 5.272 |
| HF 48 | 0.183, 0.181 | 0.159, 0.160 | 1.145, 1.141 | 1.121, 1.139 |
| B3LYP 48 | 0.493, 0.490 | 0.469, 0.466 | 2.580, 2.527 | 2.499, 2.482 |

Warm values are medians of three calls. Iteration and Fock counts are identical,
and every arm keeps its independent qualification status. HF DF stays failed for
the existing fitting-energy approximation, with unchanged errors. Between auto
and candidate processes, same-phase forces agree within 4.3e-13 Eh/Bohr.
`endpoint-384.json` and `endpoint-768.json` hold all arms, samples and
qualification errors. `results.json.xz` adds every force vector.

## Independent gates

- `oracle.json` compares against PySCF DF-RHF with auxiliary response, on
  water-12 with the README def2-SVP and auxiliary inputs. That auxiliary basis
  includes g. The native explicit shell consumer runs auto, legacy, candidate,
  auto, candidate in one prepared batch. Candidate selects Rys for
  004/104/114/204 and still selects polynomial for 214/223/224. Energy errors are
  at most 6.8e-13 Eh and force errors at most 8.8e-12 Eh/Bohr (gate 1e-8).
  Candidate and auto differ by 3.3e-12.
- `memcheck.log` is compute-sanitizer 12.9.79 memcheck over one candidate DF-HF
  energy+force call on the same system: 0 errors, without suppressing API errors.

These measurements do not qualify any other GPU. The portable profile and the
sm_120 tuning are unchanged.
