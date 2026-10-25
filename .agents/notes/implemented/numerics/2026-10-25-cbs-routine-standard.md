# Routine CBS convergence with a strict reference-density boundary

Status: implemented
Date: 2026-10-25
Base: `98531471e79d5ce31119ec8224adf5c1e5e989e2` (master)

## Decision

CBS Standard uses the previous Fast composite allowance (`1e-5 Eh`, maximum
force `1e-4 Eh/Bohr`, RMS force `3e-5 Eh/Bohr`), divided by the L1 sum of
combined component coefficients. For the default DZ/TZ pair, CCSD stops at
energy change `1e-8 Eh` and physical residual maximum `1e-7`.
The allowance-to-stopping mapping is empirical; it is not an observable-error
certificate. The [public guide](../../../../docs/user/ccsdt_cbs.md) owns the
complete effective controls and compatibility behavior.

All CBS profiles retain a reference-density ceiling of `1e-11` and a CCSD
physical-residual ceiling of `1e-7`, including Fast. Fast remains an equivalent routine default and its existing custom
budget interface is preserved. Tight retains the former strict stopping
limits; FP64 retains those limits and the original all-FP64 schedule.
Standalone CCSD(T) and HF/DFT convergence defaults are unchanged.

The independent preconvergence selector remains available for all fitted
profiles. Final CCSD and Lambda solves, independent residual/transpose checks,
triples and force response remain FP64. The shape gates previously assigned
to strict Standard now apply to Tight. No still-looser Fast tier is invented
to produce a second benchmark curve. The homepage displays Standard only.

This implements the [default assessment](../../proposed/2026-10-25-cbs-routine-convergence.md)
and supersedes the profile naming in the earlier
[strict preconvergence decision](../performance/2026-10-25-cbs-standard-preconvergence.md).
That earlier evidence remains an immutable record of its original source and
settings; it is not relabelled as a measurement of the new Standard.

## Two failures that constrained the promotion

At H2 separation 8 Bohr, the old Fast density ceiling can terminate before a
valid canonical post-HF reference is exported. Both FP64 and mixed CC schedules
fail at that same HF boundary. Tightening only the SCF energy threshold does
not fix it; retaining density `1e-11` makes both schedules succeed. Thus the
promotion deliberately does not preserve Fast's relaxed reference density.
No failed reference is reused or silently replaced with an oracle.

The same geometry exposed a separate mixed-amplitude handoff defect. The
existing projection of FP32 pair-symmetry noise depended on admitting the
optional spectator-pair contraction schedule, which excludes one occupied
orbital and can be resource-refused. Lambda requires physical pair symmetry
under every contraction schedule. The handoff now admits the existing
compiler-owned restricted-pair projection independently of that optimization.
It runs only after mixed warmup, only for an exactly symmetric supplied state,
and before the fresh FP64 epoch. Asymmetric supplied guesses and FP64 iterates
are not projected. Neither Lambda's symmetry tolerance nor final physical
residual gates are weakened. No new scientific kernel/equation is handwritten.

## Independent reference branch selection

Stretched N2 (2.8 Bohr) converges to different DZ RHF stationary points with
native Hcore and PySCF MINAO initial guesses: approximately `-108.3561164063`
and `-108.6924876302 Eh`, respectively. Both fail PySCF's internal restricted
stability test. Tight and Standard choose the same native branch; tightening
CCSD cannot fix this reference-state choice. The original default-oracle
mismatch remains in the negative trials, including its large energy/force
differences. Do not describe this case as a verified ground-state calculation.

PySCF independently reproduces the native branch using its own `1e` initial
guess, without any native density, orbitals or amplitudes. Its DIIS CCSD
iteration stalls near a small update norm. A bounded SciPy Krylov root solve
of PySCF's amplitude-update equations converges the reference to an update
2-norm below `1e-10`; the prototype reached `8.21e-14`. Every displaced point
is independently reconverged. This is a reference-only solver, never a
production fallback. The retained full Cartesian finite differences use
two steps; every displaced component must also stay within `0.01 Eh` of its
central branch, preventing symmetric branch jumps from masquerading as a
stable derivative.

After fixing the reference-density boundary, the initial promoted residual
ceiling (`2.38e-7`) produced an N2 energy error `2.16e-7 Eh`, exceeding the
predeclared `1e-7 Eh` qualification gate. Tightening energy change alone to
`1e-9 Eh` did not improve it. A physical residual cap of `1e-7` reduces that
error to `1.32e-8 Eh`. Therefore both routine names retain this independent cap;
we did not loosen the acceptance gate or discard the stress case. The initial
candidate and all control trials remain in the evidence.

## Qualification and limits

The [routine convergence campaign](../../../../benchmarks/results/cbs-routine-standard-20261025/README.md)
retains physical cc-pVDZ/cc-pVTZ energy/full-force references for water,
stretched water, ammonia, ammonium, hydroxide, formaldehyde, equilibrium and
stretched nitrogen, a water dimer, and stretched hydrogen, plus an augmented
basis hydroxide case. Stretched water and ammonium also test correlation-only
DF on exact RHF references. All calculations are closed-shell and all-electron.

Predeclared independent gates are `1e-7 Eh` for every component/composite
energy, `3e-6 Eh/Bohr` maximum and `1e-6 Eh/Bohr` RMS composite force error,
and `1e-6 Eh/Bohr` two-step force stability. Relative energies include water
and nitrogen stretch, dimer binding and the diffuse-basis shift. Native
energy-only results, repeated fresh owners, strict controls and physical
residual replays are retained. Initial failures and source/binary identities
are retained separately rather than removed from the campaign.

These gates qualify numerical convergence on the declared branch and cohort.
They do not establish single-reference model accuracy for dissociation,
automatic ground-state/stability selection, frequency accuracy, force
continuity or long-time energy conservation. Numeric budgets are not measured
peak memory, and descriptive endpoint times are not a formal statistical
performance-promotion certificate.

## Rejected alternatives and revisit conditions

- Keeping CBS Standard permanently at `1e-12`/`1e-10` spends iterations without
  a demonstrated routine observable benefit in the qualified cohort. Tight
  preserves those controls for sensitive work and independent comparisons.
- Copying another package's residual number ignores norm, preconditioner and
  driver differences. The proposal records the external defaults and sources;
  the promotion relies on observable gates, not matching their spelling.
- Projecting every final T2 array or relaxing Lambda's symmetry gate would hide
  physical errors and violate supplied-state contracts. Fix only the bounded
  mixed-to-FP64 handoff.
- Redesigning SCF state selection is a separate scientific change. Revisit it
  with independent stability/root-selection tests, rather than concealing
  branch mismatches in a convergence benchmark.

Revisit the routine ceilings when new independent molecules/relative energies
fail the declared observable allowances, when a consumer needs derivative
continuity, or when new hardware changes the measured preconvergence crossover.
