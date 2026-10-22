# Comparison conditions

This campaign measures native Standard/Fast and one GPU4PySCF default SCF
profile. All eight methods, Direct/DF, and 3/12/48/96-atom water clusters are
attempted. Timing and accuracy are separate observations: a faster sample is
plotted only when it passes the common independent energy/force gates.

| Condition | GenerativeQC | GPU4PySCF 1.8.1 |
| --- | --- | --- |
| Geometry and electronic state | Identical Bohr coordinates, charge 0, spin 0 | Same |
| Orbital basis | Spherical def2-SVP; r²SCAN-3c uses def2-mTZVPP | Same |
| Composite corrections | Native D4 + gCP, included in r²SCAN-3c timing and observables | Independent D4 + gCP, included |
| XC quadrature | Treutler radial / Lebedev angular / NWChem pruning, level 3 | Installed default, level 3 |
| VV10 quadrature | Same default level-3 prescription | Installed default nonlocal grid, level 3 |
| Moving-grid forces | Full coordinate and partition-weight response | `grid_response=True`, explicitly enabled for the shared force definition |
| VV10 density domain | Cutoff 1e-10 in energy and forces | Default cutoff 1e-10 in both consumers |
| VV10 weight screening | Keep nonzero signed weights and their derivatives | Remove points with absolute weight ≤ 1e-14 |
| Standard SCF stopping | Energy change 1e-10 Eh, density RMS 1e-8, physical residual closure | Default energy change 1e-9 Eh and orbital-gradient norm ≈ 3.16e-5 |
| Standard SCF iteration cap / DIIS history | 100 / 8 | Defaults 50 / 8 |
| Fast SCF stopping | Public `PrecisionCalculator(policy="fast")`; actual candidate, refinement, fallback and audit work retained | No GPU4PySCF Fast profile |
| Integral screening | Standard 1e-12; admitted Direct Fast candidate 1e-10 | Default direct-SCF tolerance 1e-13 |
| XC AO screening | Optional AO-jet selection uses cutoff 1e-16 when admitted; bounded dense fallback retained | Shell screening uses `AO_THRESHOLD=1e-10` |
| Initial density | Native default Hückel | Default MINAO |
| Direct Fock updates | Native default, without incremental J/K | Default incremental Fock retained |
| DF auxiliary basis | Exact frozen tables selected by the reference, including auxiliary g functions | Default selection, checked against the frozen tables at runtime |
| DF metric factorization | Eigensystem with relative cutoff 1e-10 × largest eigenvalue | Cholesky first; only on failure, eigensystem with absolute cutoff 1e-7 |
| ωB97M-V DF exchange | Fitted full-range exchange plus exact long-range correction | Fitted full-range and long-range exchange |
| Arithmetic and storage | Native Standard/Fast precision and bounded storage policies | Installed defaults |
| Cold timing | Constructor, preparation, SCF, complete forces and host export | Same declared boundary |
| Warm timing | Three complete calls on the same owner, each from its immutable cold density | Same policy; own immutable cold density |
| Execution | One exclusive Slurm GPU allocation on n1, RTX 5090; one math thread | Same |

Fast is method dependent. Admitted Direct KS candidates use energy/density
thresholds of 1e-6/1e-4 with strict final validation. Where DF executes an
unchanged strict FP64 Standard endpoint at the default budget, Fast now returns
that qualified fallback without an unnecessary second complete audit. Explicit
targets and tighter budgets retain their audit contract. Such a fallback can
match Standard work; it is not an independent relaxed DF algorithm or a promise
of lower latency. The [work table](work.md) retains every exposed solve.
GPU4PySCF's Direct profile keeps `direct_scf=True`; its stock DF wrapper sets
it to `False`.

Density RMS and orbital-gradient norm are different residuals. Assigning the
same number to both would not establish equal accuracy. The user-selected
comparison retains each public stopping policy and applies common observable
gates: energy error ≤ 1e-5 Eh/atom, maximum force error ≤ 1e-4 Eh/Bohr, and RMS
force error ≤ 3e-5 Eh/Bohr. Every cold and warm sample must pass. Independent
denser-grid stability and CPU/GPU spot checks must satisfy one quarter of these
gates. DF approximation error is included against the independent direct target.
No tolerance is changed after observing a failure.

Equal quadrature rules do not imply identical point ordering, zero padding,
active-domain screening, arithmetic or operator factorization. The remaining
differences above are retained in the comparison. In particular, the two DF
metric cutoffs cannot be ordered merely by comparing 1e-10 and 1e-7, and the
factorization policies alone do not establish the rank actually used in a case.
Likewise, AO-jet selection and reference shell bounds are different screening
criteria; their numeric cutoffs alone do not establish an accuracy or timing
ratio. Native optional AO/point batching also depends on resource admission.
For water-96, the native grid has 1,078,336 points while GPU4PySCF stores
1,078,528 entries: its installed 256-entry alignment adds 192 zero-weight
padding entries. This array-length difference is not a different quadrature.

Cold is an application-object cold call, not a clean compiler-cache or process
startup measurement. Both engines use installed/prebuilt code; native ordinary
and s/p/d RKS AOT manifests are built before measurement. Persistent compiler
and runtime caches are retained. All public precision validation inside the
endpoint is timed; independent publication qualification runs outside it.
The three warm calls quantify within-process variation, not variation across
independent process launches. The finite jobs use the scheduler's device
visibility without overrides. Different populations can run concurrently on
different cards; GPU assignment and software identities are retained per record.
The accelerator snapshot is taken before the timed calls; its inherited
`sampling_point` label says "after benchmark measurements" and must not be used
to infer post-run temperature or clock stability.

Only cold and warm are measured. Moved geometries and CBS are outside this
campaign. Failed, timed-out and accuracy-rejected populations remain in the
full inventory and timing table; they are not speedup evidence.

## Why the old ranking changed

The historical comparator imposed energy/gradient thresholds of 1e-12/1e-10,
screening 1e-14 and full-density Fock rebuilding on GPU4PySCF. The new comparator
retains its defaults. A prior same-binary PBE0 diagnostic reduced SCF from 37
to 9 iterations and cold time from 79.58 to 29.24 seconds when restoring the
convergence settings alone. This is a reference-configuration effect, not a
native optimization.

Changing the old auxiliary basis to the reference default exposed native g
response fallback. A separate same-binary PBE-DF/48 diagnostic changed warm
time from 2.571 to 19.083 seconds although the default auxiliary had fewer
functions; three-center response took 16.602 seconds. The repaired source accelerates auxiliary-g shell derivatives, avoids
redundant default Fast audits, selects generated strict Coulomb work when it
outperforms the generic mixed route, and uses bounded raw-panel response.
Eligible source-only Coulomb forces also use shell derivatives by default.
Large DF repeated value generation remains a bottleneck. The native VV10
density cutoff is 1e-10. These changes preserve the declared acceptance gates.

The [earlier settings investigation](../../../.agents/notes/implemented/performance/2026-10-20-readme-default-alignment-and-ranking-reversal.md)
retains controlled diagnostics and historical source identities. Their timings
are not substituted into the repaired figures.
