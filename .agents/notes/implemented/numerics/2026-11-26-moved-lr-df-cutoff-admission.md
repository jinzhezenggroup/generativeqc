# Decision: Recheck a fitted long-range metric after geometry changes

Status: implemented
Date: 2026-11-26

The original automatic WB97M-V strategy froze the initial LR approximation per
prepared batch item. A fresh owner for a geometry with an unresolved LR cutoff
chose exact LR before SCF. Reaching that geometry by moving a fitted owner kept
LR fitting without consulting the new metric's response admission. SCF could
converge, then the analytic derivative correctly refused the unresolved
retained/discarded subspaces. Clearing the previous density reproduced the
failure and a cold SCF retry on the same moved owner could not change admission.

Keep the existing fixed relative cutoff and resolution guard. A moved fitted
automatic owner checks its new LR metric before any SCF. An unresolved metric
retires that complete candidate and uses the established bounded exact-LR
preparation once. The matched energy and force then differentiate that same
model. The item retains the exact LR choice across later moves, even if another
geometry could admit fitting. This keeps an initially exact owner's behavior
and limits a trajectory to one automatic change. Explicit LR selectors keep
their existing semantics. Allocation and unsupported-provider fallback remain
limited to initial automatic preparation.

Full-range DF J/K, the auxiliary basis and SCF/grid controls are unchanged. A
previous converged density is just a trial for the rebuilt owner. Its previous
Fock, orbital and derivative-source leases are not transferred. The fitted and
exact LR routes use different approximations, so applications requiring a fixed
Hamiltonian along a path can explicitly select it. A forced fitted owner still
rejects an unresolved analytic response.

The two issue reproductions are water96 step 8 and n-C32H66 step 3. An independent
libcint/SciPy spectrum places their nearest LR eigenvalues 0.426 and 0.758
resolution bands from the cutoff. The preceding geometries stand 10.480 and
9.244 bands away. The full-range metrics are full rank and well separated.
The [issue evidence](../../../../benchmarks/results/issue3185-wb97mv-df-20261126/README.md)
records the endpoint diagnostics, reference gates and timing. See also the
[original automatic-admission decision](../performance/2026-11-04-lr-df-default-crossover.md).
