# Full-DF source/metric sharing qualification

Measured source `b221c5d3e`, library SHA-256
`154debe52a595b0de608c8cf78660b8198ee3889a66036fe78785a444d991cc6`.
RTX 5090 on n4, finite Slurm job 1351, CUDA 12.9, FP64. Native compilation
used ccache 4.5.1; runtime JIT used verified sccache 0.16.0. The native probe
and Python loader resolved the same final library.

`records.json` retains five alternating repeats after unreported per-policy
initialization, complete cold/warm E+F and separate actual construction work.
Cold includes calculator creation and the complete native endpoint. Each cold
sample has a fresh owner; warm reuses that owner. PySCF reference calculations
and source tracing are outside native clocks. Raw logs remain ignored under
`.artifacts/df-shared/` and the node staging bundle.

| Full-DF water endpoint | Separate source, cold / warm | Shared source, cold / warm |
|---|---:|---:|
| MP2, spherical cc-pVTZ / cc-pVTZ-JKFIT | 1.2860 / 1.2847 s | 1.2795 / 1.2790 s |
| CCSD(T), spherical def2-SVP / def2-SVP-JKFIT | 0.8104 / 0.8093 s | 0.8040 / 0.8032 s |

Cold improves only **0.50%** and **0.79%**, respectively. This is a small setup
saving, not removal of the dominant correlation/response work. MP2 cold ranges
are 1.2854–1.2884 s separate versus 1.2791–1.2811 s shared; CCSD(T) ranges are
0.8098–0.8141 versus 0.8033–0.8044 s. Retain these limited-workload bounds.

The correlation stage constructs/factorizes a second source/metric once in
the separate arm and zero times in the shared arm. Compensated MO generation
is identical: MP2 consumes 467,596 raw values, 60 GEMMs and 119,236,980
summands; CCSD(T) consumes 65,088 values, 26 transform GEMMs, 10,479,168
summands and five core block GEMMs. Reference construction still happens once.
MP2 retains 17 reference iterations and 14 orbital-response iterations in both
arms. CCSD(T) retains 16 reference, 16 CCSD, 18 FP64 Lambda and 11 orbital-response
iterations. Optional public MP2 phase fields are unavailable; its source values
are measured by the separate correlation-source trace, not synthesized.
Reported `shared_reference_bytes` is an existing conservative admission lease,
not a new allocation or a claim that this many physical bytes were saved.

Independent PySCF energy and reconverged directional-force acceptance gates
are 5e-9 Ha and 3e-6 Ha/Bohr. Observed maxima are 2.28e-13 Ha and 3.19e-9
Ha/Bohr. Production performs no oracle work.

## Validation

Twelve GPU cases pass: five independent full-DF MP2/CCSD(T) cases, four CBS
reference/correlation combinations, and three sharing/lifecycle checks.
They cover spherical/Cartesian sources, auxiliary g, replay, companion outputs,
independent finite differences and explicit disable/default. A combined-process
attempt exceeded PySCF's default 4 GB memory ceiling after accumulating native
state; unchanged CBS oracle gates pass in fresh processes. This was an oracle
resource refusal, not a passed numerical sample.

The compiled probe compares every factor/block to fresh construction, exercises
identity/capacity/policy refusal and rejection of unsupported specialized backends, retains physical pullback after the outer HF owner
is destroyed and checks final retirement. Compute-sanitizer reports zero errors.
29 existing host policy/retry tests and repository pre-commit checks pass.

Inside a finite Slurm allocation with the qualified source/library:

```bash
python -m benchmarks.df_shared_source --source-sha b221c5d3e \
  --output .artifacts/mp2.json
python -m benchmarks.df_shared_source --method 'df-ccsd(t)' \
  --orbital def2-svp --auxiliary def2-svp-jkfit --source-sha b221c5d3e \
  --output .artifacts/ccsdt.json
```

Final integration includes #2765 and the explicit backend-admission guard. All
twelve cases and the lifecycle memcheck passed again on this exact library.
The earlier parent receipts remain in Git history and ignored local artifacts.
