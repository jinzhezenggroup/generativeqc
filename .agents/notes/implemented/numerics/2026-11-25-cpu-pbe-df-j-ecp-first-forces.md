# Decision: compose scalar ECPs with the prepared CPU PBE DF-J force owner

Status: implemented
Date: 2026-11-25

## Decision

The exact orbital one-electron builder supplies kinetic energy, ionic
attraction and the residual ECP in H and H'. The explicit ECP-free auxiliary
basis supplies only fitted Coulomb energy and its streamed three-center and
metric gradients. The existing prepared KS owner authenticates D/W and publishes
the complete force. There is no Python snapshot/gradient fallback for this
admitted context and no new integral recurrence.

## Boundaries

Qualify ordinary strict-FP64 local PBE RKS/UKS first forces only. Validate the
additional ECP workspace before constructing providers. Keep global CPU DF
host-capacity promises unavailable for ECP until both quoting and prepared
admission charge the same ECP setup phase. Accept a model-level automatic CPU
guess request because native system-level resolution rejects ECP seed work;
do not admit an explicit preliminary seed here.

## Evidence and revisiting

Run `pytest tests/python/test_ecp_stationary_cpu.py -k native_cpu_df_j`.
The NaH scalar-ECP fixture pairs Cartesian and spherical orbital/auxiliary
layouts against independent PySCF DF-J with grid and auxiliary response,
including a restricted neutral and unrestricted cation. The installed
endpoint gates must also compare CLI and SDK energy/force with Python fallback
forbidden, an energy-only replay, and reconverged displaced-energy checks.
Different ECP, angular, precision, fitted-K or second-order domains need their
own complete endpoint qualification. Refs #2227.
