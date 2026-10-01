# Decision: match composite nuclear owners to their emitted dispatcher

Status: implemented
Date: 2026-10-01

The shared composite CUDA force owner emits only `(("nuclear", ()),)` into its
primitive dispatcher. That dispatcher accepts plain kind zero; it is not the
sharded Cartesian integral dispatcher. Inferring encoded dispatch from d/f AO
topology instead produces kind 32, which the nuclear-only dispatcher rejects.
Selecting ordinary integral mode for s/p/d does not repair this mismatch.

Both composite source accumulators now explicitly use geometry-only mode for
all admitted s/p/d/f bases. `_CudaSources` enables component dispatch only for
integral-derivative owners. The generic integral owner retains its default SPD
admission and encoded bindings; geometry-only owners still reject integral tasks.
No native numerical rejection, tolerance, or resource bound is changed.

Host regressions execute the actual constructor mode expression and nuclear
submission method, then compile and execute the actual emitted nuclear primitive
as C++ for d and f topology. The previous mode expression causes both dispatches
to reject kind 32; the repair yields the independent analytic nuclear derivative.
These are host ABI tests, not NVIDIA acceptance. The final head still needs the
unchanged public def2-SVP/def2-TZVP CUDA force, oracle, replay and reconverged
finite-difference matrix.

Related implementation: PR #1651's existing geometry-only nuclear-kind repair.
This restores that contract on the independently restacked PR #1637 without
resurrecting the retired WB97M-V-specific owner.

Agent: dot
