# Decision: Native CBS composition from completed component observables

The extrapolation authority already owns geometry-independent coefficients.
Apply its plan to six completed HF/MP2/CCSD(T) roles in native code instead of
maintaining Python/NumPy production algebra. Keep subtract-before-weighting
for correlation terms and compensated total energy to preserve cancellation
behavior. Energy-only composition allocates no force buffer. Force output and
the contribution energies publish together after finite validation; repeated
roles may borrow the same input arrays without a second electronic solve.

This bridge does not infer basis compatibility or convergence from scalars.
Python still authenticates the component owners and groups/schedules the
prepared calculations. The same native assembly boundary is available to a
future grouped SDK/CLI endpoint without copying the focal-point equations.
