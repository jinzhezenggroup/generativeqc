# Decision: retain strict CUDA HF stationarity for analytic force requests

Status: implemented; independently qualified HF48 default-control CUDA repair
Date: 2026-11-09 (proposed 2026-11-08)

The retained HF48 one-control experiment isolates a stopping-family failure:
default and screening-only requests miss the independent 1e-7 maximum-force
gate, while tighter energy or density controls pass. Those tighter requests
also restore the internal strict physical residual ceiling. Thus the evidence
does not isolate the public energy threshold as the sole cause.

This decision changes the CUDA HF iteration/final-state owner only, not the
separate compact density-fitting finalizer or CPU solver.

Do not loosen the independent force gate or special-case this geometry, AO
extent, device name or benchmark. Instead, allow the relaxed Standard residual
ceiling only for energy-only owners. Analytic force requests use the historical
strict HF ceiling, capped by the caller's density tolerance, in both iteration
and final-state consumers. Public controls, energy-only admission, independent
reference exports and already-strict unsupported/mixed owners retain their
existing contracts. The physical norm is not itself a force-error bound.

The new private context makes owner admission executable in the compiled host
controls without inspecting source spelling. Every force request must refuse
the energy-only relaxation, regardless of otherwise eligible owner facts.

The CUDA host driver now explicitly borrows the same scalar convergence-policy
leaf already used by the device convergence owner. Its dependency allowance is
exactly `scf/convergence_limits.hpp`, not a solver directory or implementation
prefix: the header contains only immutable facts and constexpr arithmetic, with
no provider, allocation, mutable solver state or lifetime ownership. Dedicated
boundary tests keep adjacent implementation headers and host final-state solver
headers forbidden. Keeping this numerical authority shared avoids duplicating
the admission predicate or introducing an indirect include just to evade the
ownership checker.

Scheduled RTX5090 baseline Slurm1532 reproduces the default force failure on
all four endpoints (maximum 1.818375e-7). The corrected strict force owner in
Slurm1533 passes all four unchanged independent energy <1e-8 and maximum-force
<1e-7 gates: worst energy 1.045919e-11 and maximum force 1.651619e-9. Both
processes retain constructor/preparation/SCF/force/host-export cold and three
frozen-own-density same-owner warm calls, with no production oracle imports.

That correction costs cold iterations/Focks 10 to 14 and complete time
6.366749 to 8.002465 seconds. Every warm retains one iteration/Fock. Do not
hide this cost, label the inaccurate control an accepted performance baseline,
or infer A100, other HF cases, DFT or all-method qualification. The source-bound
receipt and unchanged independent reference live in
`benchmarks/results/hf-force-stationarity-20261109/`.

The first candidate failed native compilation because the new policy header
was not explicitly included by the CUDA host owner. Retain that failure and
the corrected candidate's distinct source identity; host-only header controls
are not sufficient validation of the CUDA translation unit. Full scheduled
native compilation and all four independent endpoints pass after the include
fix, without any numerical gate, public-control or evidence-budget change.
