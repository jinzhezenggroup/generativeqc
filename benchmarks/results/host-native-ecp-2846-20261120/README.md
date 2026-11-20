# CPU def2-ECP forces on a 25-atom Ni/I fixture

The isolated candidate wheel recorded in [force.json](force.json) ran the
public `Calculator.singlepoint` energy-and-forces endpoint at charge +1,
multiplicity 3, with Gaussian/VWN-RPA B3LYP-D3(BJ), ATM off, real-spherical
def2-SVP, I28 scalar ECP, the native default grid, and direct CPU FP64. The
independent matched reference reported max force and energy differences of
`2.2883257788652855e-12 Eh/Bohr` and `4.092726157978177e-12 Eh`. The complete
endpoint took `6065.623248809949 s` and peaked at `20489788 KiB`, with 134 SCF
iterations and 136 Fock builds. The forces were bitwise equal to those of a
separate root-matched cold prepared batch. The singlepoint itself did not
export orbitals; the independent engine was only used for qualification.

The measurement is bound to baseline `7ad2f06db0a0ade439869ec48b61c6f38224b24b`
plus the exact candidate source changes, not to this integration build.
The local experiment preserves reproduction inputs and source at
`/data/jzzeng/qualify-2844-2848-20261120/host-native-ecp-2846-2847-20261120/`;
its `compact.sha256` digest is
`ef2cec5f6d01497e51df7d693ec1df7d47ec3566eef6cc2707865e28babe00eb`.
This is the representative scale force test for #2846, not a 40-atom rerun,
Hessian, or saddle result.

## Force-only integration build

The same public request was independently requalified with the force-only
integration source against `d154546d2d6629ab8c865ca08f680002d68b4152`.
[The installed integration receipt](integration-force.json) binds the exact
source, repaired wheel and loaded library. The cold call took `6174.80 s`
with `20491552 KiB` peak RSS (134 SCF iterations and 136 Fock builds); its
energy and forces match the preceding accepted endpoint bit for bit. The
existing sealed independent reference was replayed for comparison only.
The force/basis suite passed 21 tests.

The local `singlepoint-sealed-reference/installed-integration-gate.sha256`
under `/data/jzzeng/qualify-2844-2848-20261120/integration-2846-d154546d2/`
has digest `e1719e13b5fdb5038d3aa702b2c1392204cfb751083f2c05c0bd12dde2734a8a`.
The exact-wheel result was posted in #2846 comment `7040586294`.
