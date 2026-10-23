# C4/C7 independent directional-force qualification

Status: **INCOMPLETE**. No displaced scientific reconvergence has started.
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

The initial host-only replay verifies native translation (largest residual
1.2624909798142347e-12 Eh/Bohr) but reports both molecules and the overall result
INCOMPLETE because all eight displaced references are missing. The absence of
an output file does not turn into a zero or a scientific PASS.

Run host input/force-sign negative checks:

```powershell
python benchmarks/results/cbs-independent-forces-20261024/test_replay.py
python benchmarks/results/cbs-independent-forces-20261024/replay.py --raw benchmarks/results/cbs-independent-forces-20261024/raw/outputs --output benchmarks/results/cbs-independent-forces-20261024/comparison.json --require-pass
```

The second command must currently exit nonzero. The CPU runner uses a separate
absolute root, environment and output namespace. Each task is capped at 21600
seconds plus a 30-second termination grace. Atomic directory creation and scoped
leases prevent retries from overwriting receipts or companion logs. Failed,
timed-out and partial outputs are retained; there is no automatic retry-to-green.
Resource ownership, environment and source/input hashes must be frozen before
starting the suite. Resource handoff is pending; no other owner's root or
environment is changed.
