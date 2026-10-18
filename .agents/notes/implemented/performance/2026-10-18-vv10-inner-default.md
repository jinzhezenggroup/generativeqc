# Decision: refresh frozen VV10 before oversolving its inner density

Status: implemented
Date: 2026-10-18

## Problem

The cold restricted native VV10 schedule alternates complete physical target
Focks with provisional semilocal/frozen-potential density proposals. Tight inner
stationarity wastes strict J/K work on a potential that changes at the next
target refresh. Reducing only the total iteration count can nevertheless make
the complete endpoint slower when it adds expensive VV10 pair sweeps.

## Decision

For already-admitted non-budget frozen-potential proposals, use
`clip(0.01 * target_maximum_residual, min(1e-12, 0.1 * min(1e-9, density_tolerance)), 1e-5)`.
The incumbent used factor `0.001` and upper bound `1e-7`. Both change together;
the evidence does not isolate the contribution of either parameter.

Promote the conservative policy directly, without the experiment-only
environment selector. This helper remains shared by existing admitted native
VV10 domains, not selected by molecule, atom count or functional name. The
strongest large-system evidence is specifically WB97M-V/def2-SVP Direct RKS.
It does not qualify speedups for other functionals, bases or system sizes.

## Invariants

- Keep the initial semilocal stopping rule and the strict lower floor.
- Keep application-budget stationarity and confirmation thresholds unchanged.
- Preserve FP64 arithmetic, physical energy/density/residual gates and fresh
  complete-target confirmation. Provisional states never authorize publication.
- Retain finite proposal/total iteration limits, DIIS epoch resets and the
  reserved bounded ordinary full-target tail.
- Keep existing warm/imported-density precedence and schedule opt-outs.
- Do not introduce a reference engine into native preparation or execution.

## Evidence

The retained experiment is based on master
`e4aff7810b0462204831eb5168f19b6e217ef520`. For README water-96, spherical
def2-SVP/768 AOs, Direct RKS, FP64 and the 48/16/32 grid, the conservative
policy reduces provisional Focks `31 -> 23` (`9+8+7+7 -> 9+4+5+5`), keeping
five complete-target Focks and four SCF VV10 pair sweeps. Total SCF/endpoint
Focks change `36/36 -> 28/28`; the seven preliminary native atomic Focks are
counted separately. Unchanged energy/density/screening controls are
`1e-12/1e-10/1e-12`, with DIIS eight and a 100-iteration cap.

The matched single cold E+F observations on n4's RTX 5090 were
`1165.548656 -> 995.149441 s` (14.6% lower). Against the same-revision master,
energy, maximum force and maximum density-element differences were
`9.09e-13 Ha`, `7.80e-12 Ha/Bohr` and `8.50e-12`. The independent retained
GPU4PySCF water-12 E+F reference gates passed. There is no independent
water-96 oracle, statistical timing qualification or general ground-state
claim. SCF pair counters exclude force derivatives; endpoint times include them.

Production-default qualification is recorded separately in
`benchmarks/results/wb97mv-vv10-default-20261018/`. It authenticates the actual
production source/library, exercises the default without experiment/seed
overrides, and retains physical, paired and independent-reference gates.
These are validation observations, not a new matched timing campaign.
The actual-default n4 run (Slurm 1123) reproduces 23 provisional/five target
Focks, 28 SCF/endpoint Focks and four SCF pair sweeps, taking 995.677 s.
Paired master energy/maximum-force/maximum-density differences are
`1.364e-12 Ha`/`8.258e-12 Ha/Bohr`/`8.509e-12`. Actual-default water-12
independent-reference gates pass as well. Device schedule/exhaustion tests
(13) and the expanded native-library integration suite (221, n1 job 10075)
pass. The production source identity is distinct from both historical
libraries; the archived experiment selector is absent.

## Rejected alternatives

- Removing VV10 or disabling the semilocal initializer changes the question;
  fewer iterations alone do not imply less complete scientific work.
- Factor `0.1` with cap `1e-5` reduces provisional work to 20, but needs six
  targets/five pair sweeps and takes 1023.311 s, slower than the chosen policy.
- Loosening the lower floor gives no additional work reduction in the water-12
  screen and is not qualified at 96 atoms. Retain the tighter caller-aware floor.

## Consequences and revisit conditions

The change spends less effort on intermediate densities without weakening final
acceptance. An unfavorable outer trajectory can still spend the finite proposal
budget before its ordinary full-target tail. Revisit if broader independent
fixtures reveal convergence/fallback regressions, or matched complete endpoints
show that a different policy saves pair sweeps rather than just inner iterations.

## References

- `src/dft/application_precision.hpp`: shared proposal tolerance owner.
- `docs/user/initial_guesses.md`: current user behavior and opt-outs.
- `benchmarks/results/wb97mv-vv10-inner-20261018/`: original experiments,
  rejected policies, scope limits and source-bound compact evidence.
- `tests/python/test_application_precision_admission.py`: compiled boundary,
  floor, monotonicity, budget and confirmation contracts.
- `tests/python/test_nonlocal_scf_work_receipt.py`: device physical confirmation,
  total-budget exhaustion, warm-density precedence and ordinary-solve parity.
