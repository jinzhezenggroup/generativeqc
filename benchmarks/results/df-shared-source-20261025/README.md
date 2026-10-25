# Full-DF source/metric sharing qualification

Measured source `e93b34dc9`, library SHA-256
`477444e51f31ff10f9a7609b901bb0f7c15e9a8d271caa97b6d67a53a7a091ad`.
RTX 5090 on n4, finite Slurm job 1350, CUDA 12.9, FP64. Native compilation
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
| MP2, spherical cc-pVTZ / cc-pVTZ-JKFIT | 1.2860 / 1.2849 s | 1.2798 / 1.2786 s |
| CCSD(T), spherical def2-SVP / def2-SVP-JKFIT | 0.8099 / 0.8089 s | 0.8033 / 0.8032 s |

Cold improves only **0.48%** and **0.82%**, respectively. This is a small setup
saving, not removal of the dominant correlation/response work. MP2 cold ranges
are 1.2855–1.2881 s separate versus 1.2792–1.2812 s shared; CCSD(T) ranges are
0.8096–0.8131 versus 0.8032–0.8044 s. Retain these limited-workload bounds.

The correlation stage constructs/factorizes a second source/metric once in
the separate arm and zero times in the shared arm. Compensated MO generation
is identical: MP2 consumes 467,596 raw values, 60 GEMMs and 119,236,980
summands; CCSD(T) consumes 65,088 values, 26 transform GEMMs, 10,479,168
summands and five core block GEMMs. Reference construction still happens once.
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
identity/capacity refusal, retains physical pullback after the outer HF owner
is destroyed and checks final retirement. Compute-sanitizer reports zero errors.
29 existing host policy/retry tests and repository pre-commit checks pass.

Inside a finite Slurm allocation with the qualified source/library:

```bash
python -m benchmarks.df_shared_source --source-sha e93b34dc9 \
  --output .artifacts/mp2.json
python -m benchmarks.df_shared_source --method 'df-ccsd(t)' \
  --orbital def2-svp --auxiliary def2-svp-jkfit --source-sha e93b34dc9 \
  --output .artifacts/ccsdt.json
```

The initial receipt predates the merge with #2765 and the additional explicit
backend-admission guard. Final integration qualification is pending; these
measurements retain their original identity rather than being relabelled.
