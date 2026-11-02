# WB97M-V fitted long-range diagnosis and correctness

This capsule retains complete endpoint records, original source identities,
resource failures and test logs from the long-range DF investigation. It does
not qualify a new default or claim production performance. The measured builds
have `BUILD_TESTS=ON`; intrusive instrumentation was enabled in the large runs.
The published production integration has a different source identity and needs
its own production-build endpoints.

The final A100 diagnostic (LSF 224641, scientific source
`f72dc560b8fc9330c78665dbc3242c2ea1169fd5ac713020256984b313e5484f`)
completes 96-atom WB97M-V/def2-SVP with the README JKFIT auxiliary basis and
explicit fitted long-range exchange. Native cold SCF takes 28 Fock builds;
GPU4PySCF takes 10 iterations. Warm calls use each engine's own frozen cold
state and take one iteration. Energy differences are below 8.4e-10 Eh/atom;
maximum force differences are below 2.4e-7 Eh/bohr. The diagnostic warm times
are approximately 137.0 s native and 29.2 s reference: this source is still
slower. Its long-range generic derivative accounts for about 127 s of the
first force response. These inclusive intrusive intervals are not additive
independent endpoint components.

Earlier converged-SCF attempts 224548/224584 and Slurm 15254 did not return a
complete force endpoint and cannot supply timing samples. Their failures led
to a bounded rank-truncated metric adjoint, early mixed-force routing and an
independent KS-owned occupied-factor lease. The exact/mixed LR default remains
unchanged; fitted LR still requires `GENERATIVEQC_DF_RANGE_EXCHANGE=df`.

The 360 independent raw-integral controls pass on A100 (224652) and RTX 5090
(15258). On A100, the preceding test attempts failed first in the compiler-cache
parser and then in the probe linker, before numerical execution; those logs
remain failures. The corrected runner resolves nvcc and supplies explicit
headers and the task-local link search path. Native range, occupied-factor
lifetime and semilocal gates pass on both devices. Slurm 15258 additionally
passes 247 response/endpoint tests, with one expected skip. No PRO 6000 or other
device qualification is inferred.

The explicit LR shell candidate (Slurm 15268, source
`df3d9cbbe4f2fddc6b100cd0d4056fbdd36300b0cb65e19eb927418c14b92558`)
completes all four RTX 5090 endpoints. Cold times are 279.33 s native and
373.44 s GPU4PySCF; warm medians are 66.58 s and 124.56 s. All paired energy
and force gates pass, with maximum force error 2.46e-7 Eh/bohr. These remain
instrumented diagnostic timings. Slurm 15269 passes 48 molecular tests with
one skip. The auxiliary-g native fixture attempt 15274 fails because its CPU
LR oracle only supports through f; the corrected f fixture needs a new run.
The separate molecular libcint gate covers the actual auxiliary g basis.
Slurm 15257 has no completion receipt and remains incomplete despite its
partial native/reference samples. The capsule retains its last snapshot.

`receipts.json.xz` and `fp64.xz` use the existing exact v3 reader. Original JSON
formatting and FP64 bytes are authenticated during recovery, including failure
logs. Full intrusive progress/component streams remain in the original ignored
local archives; their full lengths and SHA256 values are listed in
`intrusive_stream_inventory`. The capsule retains labeled component aggregates
and all complete endpoint samples and SCF work counts. It does not claim that
aggregates replace those raw profiling streams or support final timing promotion.

Run the independent offline endpoint comparison:

```bash
PYTHONPATH=python:. python benchmarks/results/wb97mv-fitted-range-20261102/report.py
```
