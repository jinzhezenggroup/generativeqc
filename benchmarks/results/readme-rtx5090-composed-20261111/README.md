# RTX 5090: default HF/DFT endpoints, 2026-11-11

Master `4024f95b809c70b404f2ed202a67278d1fb86395` plus exact nuclear derivative
and overlap-product optimizations. `samples.json.xz` pins the complete source
identity, patch/archive and ELF hashes, build receipts, software and allocation.

| Mode | Cold wins | Warm wins | Qualified cold | Qualified warm |
| --- | ---: | ---: | ---: | ---: |
| Direct | 32/32 | 32/32 | 31/31 | 31/31 |
| DF | 32/32 | 32/32 | 27/27 | 27/27 |

Fresh 64 pairs / 128 processes / 512 complete E+force calls; no failures,
retries, historical timings or point substitutions. Direct ωB97M-V96 warm:
9.42×. These are observations, not universal rankings. CBS is unchanged.

DF HF96 cold is 16.258817 s native versus 16.558978 s default GPU4PySCF;
warm medians are 3.774124 / 6.910326 s. Native target iterations/Focks remain
11/12 plus seven preliminary atomic builds; reference iterations/veff builds
remain 10/11. A separate six-permutation cohort (36 processes / 144 endpoints)
finds all six native HF96 cold samples faster than every reference sample:
16.249662–16.257009 versus 16.533740–16.549327 s.

Finite exclusive node4/main Slurm job 1569 uses one stock 575 W RTX 5090;
48 CPUs allocated, endpoint CPUs 24–27 bound to GPU-local NUMA1, math threads=1.
Engine pairs run consecutively, first-engine order balanced 32/32. Monitoring
retains 4,366 host samples and GPU telemetry; no unexpected compute PID,
concurrent running job or monitor error. CPU busy median/max 2.1843/2.8719%;
CPU/memory PSI avg10 max zero, IO 0.18. No unqueried throttle claim is made.

Cold includes construction, preparation, SCF, forces and host export; import,
context startup and cache-miss compilation are excluded. Warm uses the median
of three immutable own-cold-density calls, with min–max bars. Caches remain;
native timing processes reject reference-engine imports.

The native constructor omits `policy`, matching public `PrecisionCalculator`
and `PrecisionPolicy()` defaults. No convergence, guess, grid, tile, XC-order
or profile selectors are supplied. GPU4PySCF defaults are checked unchanged.
Both engines retain full grid-response forces. Method/basis, spherical AOs and
frozen auxiliary inputs define the common model. Native physical final-state
proofs and strict/unsupported-owner fallbacks remain mandatory; see
[application precision](../../../docs/user/application_precision.md).

Both engines independently qualify 58/64 pairs: E/atom ≤1e-5 Eh, Fmax ≤1e-4
and Frms ≤3e-5 Eh/Bohr. All DF HF sizes and Direct/DF M06-2X3 still fail these
independent gates and remain plotted. All 1,024 cross-engine full-vector
comparisons pass; this does not waive those six failures.

Replay scalar errors, gates and both figures with Matplotlib 3.10.9:

```bash
PYTHONPATH=python:. python -m benchmarks.readme_defaults_summary \
  --campaign benchmarks/results/readme-rtx5090-composed-20261111 \
  --output .artifacts/readme-optimized-replay
```

Full vectors, source-matched build/qualification receipts and acquisition scripts
remain in `.artifacts/df-hf-overlap-20261111/` and the corresponding
`n4:/data/jzzeng/df-hf-overlap-20261111/`. Prior publication is retained in Git
history and ignored `previous-publication/`, not mixed into these measurements.
