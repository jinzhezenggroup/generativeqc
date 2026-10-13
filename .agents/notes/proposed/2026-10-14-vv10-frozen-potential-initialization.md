# Experiment: native frozen-potential VV10 initialization

Status: default-off prototype; no endpoint qualification or performance claim
Date: 2026-10-14

The native semilocal-only initializer still needed 16 expensive target Focks at
12 atoms. At 96 atoms one exact VV10 sweep costs about 82.6 seconds. Exact
ingredient reuse helps frozen replays but cannot remove changing-density cold
sweeps. This motivates a nested initialization rather than another kernel tweak.

`GENERATIVEQC_VV10_FROZEN_POTENTIAL_SEED=1` starts with the bounded same-owner
semilocal/RSH initialization, then periodically evaluates the full VV10 target.
Between these anchors it solves a provisional problem with the anchor's fixed
VV10 potential and current semilocal XC and exact J/K. The anchor's gradient,
screening weights and generated pair derivatives remain together in the existing
resident buffers. No extra pair storage, CPU solve, reference density, precision
change or different quadrature is involved. Its constant nonlocal energy is
explicitly provisional, not a physical target energy or convergence certificate.

Each inner stage stops at maximum residual below 1e-9 or after 16 iterations.
Changing the anchor clears DIIS and the energy baseline. A physical residual
below 1e-8 switches to the ordinary exact target tail. The last 20 iterations
are reserved for that tail if the nested scheme stalls. All stages share the
original total iteration budget. Acceptance still requires two consecutive
physical target evaluations and the original energy/density/RMS and maximum
residual gates. Failed/provisional states never publish forces, convergence or
last-good warm state. Resident frozen warm replays skip initialization.

The optional private history-scope receipt distinguishes target, semilocal seed
and frozen-potential inner rows, including interleaved anchors. The public C ABI
is unchanged. Work receipts count every provisional/target Fock and exact pair
sweep. Unsupported UKS, lower-precision, incremental-J/K and device-chunk paths
retain their existing solve. Default selection is unchanged.

Required evidence: independent complete molecular E/F, geometry rebuild and
warm replay, finite-budget nonconvergence, exact pair reuse/invalidation and
sanitizers, then source-matched 12-call populations on 3/12 before larger
48/96 trials. This proposal does not close #2426 or promote benchmark points.

See also `2026-10-14-native-vv10-density-seed.md` and
`2026-10-14-vv10-exact-scf-component-reuse.md` in this directory.
