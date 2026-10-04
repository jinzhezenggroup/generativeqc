# Decision: reject skipping K preparation in an unused complete-J wrapper

Status: rejected
Date: 2026-10-05

`enqueue_generated_coulomb(GeneratedExchangePlan&, ...)` unconditionally
prepares exchange spin density and density bounds before J. Removing that
preparation looked like a PBE0 work reduction when reading this function alone.

The production dispatcher already calls the other `GeneratedCoulombPlan`
overload whenever complete streaming J is available. It uses the exchange
wrapper only for bounded higher-l J, where those density bounds are required.
Thus the proposed change removes no work from the actual direct-PBE0 path.

The native J-only scratch regression also passed against the unchanged #1908
library. This negative control, retained locally as node1 Slurm job 5780,
exposed the mistaken assumption before any PR/default promotion. The prototype
commit `78a1edf18` is reverted; do not repeat this wrapper-only change as a
performance improvement. Always follow dispatch to the executed consumer.

A distinct remaining duplication is the two public-to-Cartesian transformations
for each full-range J+K request: standalone J transforms the total density, and
K transforms the spin density. Sharing this input within one request is a
separate candidate and requires independent matrices, actual scratch/work
observations and complete endpoint qualification.

Refs #1892, #1895, #1908.
