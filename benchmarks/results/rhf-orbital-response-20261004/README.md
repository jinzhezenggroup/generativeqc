# Physical RHF orbital and nuclear response

The 230-AO ethane endpoint uses conventional FP64 RHF and correlation-only
DF-RCCSD(T), aug-cc-pVTZ / aug-cc-pVTZ-RI, o=9, v=221, q=488 and a 64 GiB
complete numeric budget. Native endpoints compute their own orbitals, factors,
amplitudes and response; independent oracle energies are used only for validation.

## Measurement protocol

All retained timings use n2, PRO6000 UUID
`GPU-54595246-dbdc-a633-dc38-7bd8eea3831a`, CUDA 12.9.1 and driver 595.91.07.
Each configuration has one cold sample. Stage2 (job2226) measures legacy
three-pass, canonical bilinear and energy-only with one frozen binary. Stage3
(job2238) measures symmetric two-pass and its screened variant with a second
frozen binary. Stage3's source changes the nuclear schedule and its controls;
CCSD, Lambda, triples and the physical J/K operator are unchanged. The
legacy/two-pass comparison uses the same GPU across separate allocations and
binaries; it is not a same-binary pair. Each force call independently converges
its RHF and correlation state.

`provenance.json` records source commits, binary/input/archive hashes, build
flags, allocation identities and test jobs. Archived source files were checked
byte-for-byte against the remote build checkout before freezing. No GPU work or
compilation was submitted to node3. The input and retained independent oracle
are in `../df-lambda-gemm-20261004/`.

## Complete endpoints

Both stage3 endpoints completed successfully. Raw endpoint records preserve all
counters and forces.
`summarize.py` recomputes phase ratios, every cold force pair, independent FD
gates and idealized Amdahl bounds without discarding failures.

| Seconds | Legacy three-pass | Two-pass | Two-pass, mask 1e-12 | Energy only |
| --- | ---: | ---: | ---: | ---: |
| RHF | 122.480 | 128.638 | 171.973 | 147.399 |
| DF source | 0.986 | 0.981 | 0.986 | 0.989 |
| CCSD | 171.624 | 171.650 | 171.764 | 171.719 |
| (T), including responses for forces | 110.539 | 110.581 | 110.590 | 4.918 |
| Lambda and parameter VJPs | 274.359 | 274.389 | 274.389 | absent |
| Factor/source response | 3.689 | 3.691 | 3.691 | absent |
| Orbital/nuclear response | 668.733 | 616.828 | 616.589 | absent |
| Nuclear two-electron subset | 155.771 | 103.814 | 103.812 | absent |
| Complete endpoint | 1352.438 | 1306.786 | 1350.010 | 325.026 |

The nuclear phase improves **1.500x**, saving 51.957 s. Complete wall time
improves 1.035x, saving 45.652 s; the independent cold RHF takes 6.158 s longer
in the candidate. Complete numeric capacity is unchanged at 7170696363 bytes.
Both actual consumers use shell passes, reduced from three to two. J/K remains
512.631/512.691 s and is not credited with this improvement.

The fixed 1e-12 mask has **no meaningful measured benefit**. It removes
4924248 of 16117903620 complete J/K ERI evaluations (0.03055%), while canonical
visits remain unchanged. It uses 24 screened actions and four exact actions,
12 provisional iterations and no exact corrective solve. Z solve changes from
439.348 to 439.312 s; total response changes from 616.828 to 616.589 s. These
subsecond differences in a single sample do not establish a speedup. Complete
force rises to 1350.010 s, almost entirely because cold RHF independently rises
by 43.335 s. This total increase is not attributed to screening. Keep threshold
zero; this experiment does not establish profitability for other masks or sizes.

The measured canonical experiment is rejected for default promotion: nuclear
two-electron response grows from 155.771 to 258.643 s, despite replacing three
passes with one. Complete force grows from 1352.438 to 1486.362 s, but RHF also
differs by 31.178 s; only the isolated nuclear difference of 102.872 s is assigned
to that changed phase. It loses primitive/component reuse supplied by the
existing bounded shell consumer. Energy-only takes 325.026 s. Subtracting cold
energy/force totals does not measure an exact incremental force cost.

## Work, memory and applicability

For one unscreened quadratic derivative consumer, write its actual work as W(X)
for density X. The legacy schedule costs `W(D+P)+W(D)+W(P)` and symmetric
polarization costs `W(D+P)+W(D-P)`. If individual passes cost the same, the
derived phase ratio is 2/3; this assumption is not an internal shell FLOP count.
The identity follows from homogeneous quadratic `E2(D)=D:G(D)/2` with a fixed
self-adjoint G. The nuclear gradient holds D and P fixed during differentiation.
The implementation reuses its combined N*N host matrix after each consumer
drains. No new asymptotic storage is needed, and the retained three-pass complete
capacity bound covers the two-pass schedule. Both schedules retain optional
specialized/bounded shell consumers and generic capacity fallbacks.

The measured legacy path performs **three bounded shell passes**. Its internal
quartet/jet work is unmeasured and stays `null`. The hypothetical generic
ordered-AO count `3*N^4` does not apply to this baseline. The rejected canonical
consumer uses M=Nc(Nc+1)/2 Cartesian AO pairs, M(M+1)/2 unique pair pairs, and
at most u-1 three-axis derivative jets for a quartet spanning u distinct atoms.
Its measured 575639415 visits and 1276669675 jets match geometry-derived bounds;
they are not hardware FLOPs. Nc=260 while the public basis has N=230.

Fixed geometry-only Schwarz screening preserves linearity and self-adjointness
by retaining complete canonical ERI permutation orbits. This differs from the
ordinary provider's density-dependent screening. For Z solve action counts n,
the cost changes from `n0*C(0)` to `ns*C(tau)+nc*C(0)` plus any extra audit;
ns and nc must be measured. The mask does not remove canonical traversal visits,
only evaluated ERIs. It can lose overall if exact refinement outweighs saved
integral work. The warm-start solution adds a charged o*v double vector
(15912 bytes here). Missing optional canonical storage retains the exact solve.

Screening defaults to zero. Positive thresholds are provisional solver controls:
the final Z residual is checked with the unscreened physical operator at 1e-10,
with exact GMRES correction if needed. Final weights and nuclear sources remain
unscreened. Canonical bilinear evaluation also remains an explicit experiment.

From the legacy complete endpoint, idealized zero-cost Amdahl bounds are 1.130x
for the entire nuclear two-electron phase and 1.481x for the whole Z solve.
Removing all J/K including mandatory weights and audits gives a looser 1.610x
bound. These are not achieved speedups or predictions for screening.

## Numerical gates and limits

Force endpoints are compared to the retained independent directional energy
derivatives at h=1e-4 and 3e-5 bohr, using the unchanged 3e-7 Eh/bohr gate.
The analytic direction is `-(F[2]-F[14])/sqrt(2)`. A separate cold-pair maximum
component gate is 3e-9; all measured pairs are retained. Z residual and frame
stationarity gates stay 1e-10 and 1e-8 respectively.

The two-pass directional FD errors are 4.896e-9 and 5.620e-9; Z residual is
1.360e-13 and stationarity is 1.139e-11. Its cold force difference from legacy
is 1.119e-9, and from rejected canonical is 1.850e-9. Both pass the extra
3e-9 gate. Across all four force schedules, the maximum of all six cold-pair
component errors is 2.702e-9, so the complete retained set passes. The screened
endpoint also passes FD (3.885e-9/6.631e-9), exact Z residual (1.358e-13) and
stationarity (1.036e-11). These are residual/force checks, not RHF stability tests.

The two-pass implementation passes 38 small-system real-GPU response/complete
force tests. Eight signed 2-/4-center spherical/Cartesian derivative cases pass
compute-sanitizer with zero errors. Additional gates include fixed-mask signed
linearity/self-adjointness, dense masked libcint J/K, aggressive-mask correction,
nonzero-Z schedule parity, independent two-step FD and finite-publication checks.
Twenty-eight compiler/matrix tests and 18 host RSH-weight tests pass. The native
canonical RSH/FD regression passes; the broader native RSH suite stops at an
optional shell derivative lease absent in this build for both baseline and
candidate. It is not reported as a full-suite pass. A stale Lambda diagnostic
test double is repaired separately; all 47 delayed-copy failure tests pass.
`canonical-rsh-validation.patch` retains the exact local native test adaptation:
it excludes the unavailable shell-only checks, orders diagnostic evaluation and
adds progress labels. Canonical and independent CPU/FD acceptance thresholds
are unchanged. This patch is validation evidence, not a production test edit.

Small-system tests and total energy/force agreement do not qualify the outstanding
large per-source-factor `atol=rtol=3e-10` gate. Global RHF stability is not
certified. The earlier #1829 cold-force discrepancy remains a separate unresolved
record. Neither performance improvement nor a local Z residual closes these
scientific qualification limits.

## Reproduction

Check out the source identity in `provenance.json`. Configure the CUDA library
with explicit CXX/CUDA ccache launchers, `CCACHE_BASEDIR` set to the checkout root,
Release and sm_120. The recorded build disables AOT shells, stationary-force AOT,
PCH, tests and CLI. Link `benchmarks/df_ccsdt_force_endpoint.cpp` separately with
C++20, `GENERATIVEQC_HAS_CUDA=1`, repository `include`/`src`, and CUDA headers.
Freeze the library, endpoint and input; set `LD_LIBRARY_PATH` to the frozen
library directory. Preserve Slurm-assigned device visibility.

In a finite n2 allocation (`main`, `--gres=gpu:pro6000:1`, one task), run:

```bash
./endpoint ethane230.input legacy.json 1 1 1 1 8 0 1 0
./endpoint ethane230.input canonical.json 1 1 1 1 8 0 1 1
./endpoint ethane230.input symmetric.json 1 1 1 1 8 0 1 2
./endpoint ethane230.input screened-1e-12.json 1 1 1 1 8 1e-12 1 2
./endpoint ethane230.input energy.json 1 1 0 1 8 0 1 2
```

The actual measurements are split across stages as described above; these
commands reproduce all controls on the current selector interface. J/K profiling
is enabled consistently and its timing overlaps enclosing phases. Work units
distinguish visits, integral evaluations, jets and contraction summands. Optional
census availability must be checked before interpreting raw numeric fields.
