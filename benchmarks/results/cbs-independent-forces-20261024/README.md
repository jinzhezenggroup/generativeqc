# C4/C7 independent directional-force qualification

Status: **PASS**, scoped to the frozen native C4/C7 two-step directional-force
and translation qualification. All eight fresh-geometry reconvergences completed
with exit 0; the controller and terminal source/environment audit also pass.
No new native-source/device endpoint or full-vector qualification is claimed.

| Point | Step (Bohr) | Standard max error (Eh/Bohr) | Fast max error (Eh/Bohr) |
| --- | ---: | ---: | ---: |
| C4 | 1e-4 | 1.2213363326327858e-8 | 1.2275609025374362e-8 |
| C4 | 3e-5 | 2.3297564630958556e-8 | 2.314922362654792e-8 |
| C7 | 1e-4 | 6.451835189800331e-9 | 6.484474723671141e-9 |
| C7 | 3e-5 | 1.4367193644851395e-8 | 1.4372056532721555e-8 |

These maxima include all five constituents and the composite; the gate stays
5e-6 Eh/Bohr. The largest translation residual is 1.2624909798142347e-12
Eh/Bohr, below 1e-6. The four C4 reference calls took 337.372, 305.060, 313.587 and
305.686 seconds, with maximum process RSS 923036 KiB; these are oracle resource
observations and are not performance denominators.
The four C7 timers are 2779.829, 2785.575, 2631.795 and 2629.316 seconds;
maximum process RSS is 4808844 KiB. These are independent-oracle observations.
Refs #2679; the parent stays OPEN pending the coordinating audit.

This family extends the frozen `cbs-fast-memory-20261021` qualification with eight
independent PySCF energy reconvergences, for C4 (QM7 index 48) and C7 (5604),
at both signs of 1e-4 and 3e-5 Bohr along the normalized NumPy default_rng(709)
direction. Every basis and geometry starts from fresh MINAO; neither native
state nor the preceding geometry's density, orbitals or amplitudes seed it.
The pinned QM7 float32 Bohr centers, spherical all-electron cc-pVDZ/cc-pVTZ,
shared JKFIT for RI-JK and correlation, HF exponent 1.63, MP2 exponent 3, and DZ
DF-CCSD(T)-DF-MP2 correction are unchanged. The driver requires PySCF 2.14.0.

The separately delivered central ENERGY publication at upstream
`334e7f489870dccc22ae8864c01f2353aa9689a4` is reused read-only. Its constituent
and composite 5e-8 Eh gates apply to the original centers. There is no native
displaced-energy comparison, new native build, new device execution or timing
campaign here. Oracle wall times and CPU resources provide provenance only.

`replay.py` reconstructs the original native source and verifies both historical
publications from committed LF bytes. It projects each Standard/Fast constituent
and composite force on the exact direction and compares it with **minus** the
central energy difference. The unchanged error gate is 5e-6 Eh/Bohr at **both**
steps for **both** policies; translation is max(abs(sum(force, atoms))) <=
1e-6 Eh/Bohr for each constituent and composite. Finite directional coverage
does not establish full-vector accuracy or the true CBS basis-set limit.

With `h=(-1/expm1(1.63), 1+1/expm1(1.63))` and `m=(-8/19, 27/19)`, the
composite is `h[0]*HF_D + h[1]*HF_T + m[0]*(MP2_D-HF_D) +
m[1]*(MP2_T-HF_T) + CCSD(T)_D-MP2_D`. Forces use the same linear coefficients.
Replay also checks the stored HF, MP2-correlation and delta force decomposition
against its constituents; the comparison records the expanded coefficients and
the explicit `F dot direction = -(E_plus-E_minus)/(2*h)` convention.

The initial host-only replay reported INCOMPLETE with all references missing;
its exact Git-LF bytes remain in `initial-comparison.json.xz`. Missing outputs,
failed executions and absent terminal proof cannot satisfy strict verification.

Run host input/force-sign negative checks:

Replay, verification and publication require unoptimized Python with
`PYTHONOPTIMIZE` unset or `0`. They refuse `-O`, `-OO` and inherited optimization
at import and before entry-point evidence I/O, because assertions enforce the
scientific gates. Subprocess regressions cover these refusals before retained
files can be read, restored or overwritten; this is an offline reader repair
and requires no new scientific calls.

```powershell
$env:PYTHONPATH='.;python'
python benchmarks/results/cbs-independent-forces-20261024/test_replay.py
python benchmarks/results/cbs-independent-forces-20261024/test_pack.py
python benchmarks/results/cbs-independent-forces-20261024/verify.py
```

For a POSIX shell, set `PYTHONPATH=.:python`. The verifier builds a fresh committed LF
view, validates all publication bindings, restores the raw JSON/XZ collection
into a fresh directory and requires complete scientific PASS, including terminal
source/library stability and all eight task/controller outcomes. Before artifacts
are committed, `--working-tree` verifies the candidate publication instead.

The CPU runner uses a separate
absolute root, environment and output namespace. Each task is capped at 21600
seconds plus a 30-second termination grace. Atomic directory creation and scoped
leases prevent retries from overwriting receipts or companion logs. Failed,
timed-out and partial outputs are retained; there is no automatic retry-to-green.
The outer controller is capped at 180000 seconds plus a 30-second grace; each
invocation is still bounded by the smaller per-call cap and there are at most
eight calls, with no automatic retries. Controller exceptions, timeout, stdout
and outcomes are retained independently from scientific convergence.

The received qz CPU Notebook is the early `i2679-cbs-prep-1023`, creation time
1792689025000, container `--0020ef93afaa-`, node `cpu-nat-576`, HIGH/6. The opaque
Notebook ID is UNAVAILABLE. Fresh receipt checks confirmed `cpu.max=400000
100000` and `memory.max=17179869184`; the visible cpuset 0–127 is not the
allocation. Numerical pools use four threads and PySCF max_memory=8000 MB.
Its image is `generativeqc-science-py311-cuda129:20261015`. A separate root,
environment, outputs and leases are under
`/inspire/ssd/project/chemicalreaction/czxs25220150/i2679-cbs-forces-20261024-5def`.
Old ENERGY artifacts and environment remain immutable; the instance is retained
until platform closure under the existing resource authorization.

Measured oracle/runner sources match committed LF bytes at
`6478a8fd82a5fa4605e29e1f9f5cf88485fa4ed2`; native qualification retains its
separate historical source/library/patch identities. `execution-initial.json.xz`
retains 114 original preflight/fixture/launch text records with sizes, hashes and
observed POSIX modes. The command snapshots retain the actual executed text;
their POSIX file modes are UNAVAILABLE because the CLI executed command strings.
Nine runner and two outer-controller checks used diagnostic stubs and performed
zero scientific energy calls. `execution-final.json.xz` retains 13 terminal
records, verifying unchanged source/library hashes, dataset identity, original
modes and no remaining scientific workers. The suite used exactly eight calls
without retries; no dependency changed during execution. The raw collection
retains all 40 receipt/log/outcome files byte for byte, with observed modes.
