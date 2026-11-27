# Moved WB97M-V DF cutoff fixtures

The full-precision Bohr coordinates are the preceding and failing steps of two
GPU4PySCF PBE Direct L-BFGS-B trajectories produced in the
`codex/h100-real-trajectory-20261125` campaign on native revision
`53442660ae08d965890632b6bb538c2f45aa253d`. Each `frames.json` pins the complete
original sequence SHA-256 and the hashes of the two retained frames. The
auxiliary files are the campaign's spherical PySCF default DF inputs for
def2-SVP, copied byte-for-byte for their respective element sets.

The target energy and analytic force vectors were calculated independently by
GPU4PySCF 1.9.0 Direct with PySCF 2.14.0, XC grid level 5 and VV10 grid level 3.
The oracle used its own cold density, full-density operator builds, full moving
grid response, 1e-11 Eh energy tolerance and 1e-7 orbital-gradient tolerance.
Its comparison with the independently reconverged level 4/2 result passed
one quarter of the common gates. The original sample identity is in each
`reference.json`. The [qualification](../../../benchmarks/results/issue3185-wb97mv-df-20261126/README.md)
describes the gates and acquisition.
