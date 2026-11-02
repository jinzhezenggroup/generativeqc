# H100 SM90 current-source stage

Measured native source: `cd2b4514be0a38c4920934481b1f02d157a694b4`, tree
`c76f5bd3cfa62416255c5d6914cf701d34e10a8d`. Production core SHA256:
`37b8bd02c1dddbf42385c494ae21ff51b50d874dae1dd0c01ab2b75fb0eb194e`.
The fixed stock GPU4PySCF reference recipe uses source
`15ab66a68a061355fda833a93e88484dbfee6360`. These are frozen measurements;
later repository changes do not inherit their qualification.

Three exact terminal archives retain build metadata, source/producer bindings,
platform status, complete numerical outputs, comparison operands, traces and logs.
`cohort.json` authenticates the delivered archives, manifests and CPU readers.
The actual libraries remain remote; local build replay checks metadata and the
16 AOT manifests. It does not inspect downloaded binaries or establish every
companion library's runtime loading.

| Stage | Actual result | Boundary |
|---|---|---|
| SM90 production build | Core + 16 AOT libraries + 16 manifests | Build metadata only |
| DF molecular regression | 44 nodes, 299 public full E/F outputs | 41 nodes contain CPU references; 3 are native controls |
| Direct CPU E/F comparisons | 250 of 299 outputs | Other 49 outputs have other control roles |
| Fixed-policy screening | 66 processes, 264 full E/F, 528 native/reference cross-comparisons | One process per arm/input |
| Primary scientific qualification | 30/30 fixed inputs | Original energy/force gates retained |
| Exploratory performance | 21/30 ratios >1.05; geometric mean 1.2657558348 | Not final three-pair acceptance |

The 30 primary inputs cover eight methods, 48/96 atoms and Direct/DF, with the
two preexcluded WB97M-V DF combinations retained as controls. HF DF uses the
explicit shared cc-pVTZ-JKFIT input. The global tile1024 policy was carried from
the earlier freeze and was not reselected from these timings. Each process
retains one cold and three frozen-seed warm full E/F calls. Numerical gates are
1e-5 Hartree/atom, force maximum 1e-4 and force RMS 3e-5 Hartree/Bohr.

CPU PySCF and stock GPU4PySCF are independent numerical implementations, not
dataset ground truth. The preserved review is same-family/provisional, with
actual model/effort unverified. Historical oracle producer-byte/work-precision
limits remain in its report. Its input seal covers the original private cohort;
not every listed audit input is part of this public stage packet.

Run from a repository clone with Python 3.12+ and NumPy:

```sh
python benchmarks/h100_current_cohort.py
```

The reader makes temporary copies, checks the fixed Git objects, replays all
three receivers in separate processes, and never calls a GPU or quantum-chemistry
solver. A shallow clone must first fetch the two frozen commits shown above;
the reader reports the exact missing commit and performs no network request.

This stage does not establish the native/sanitizer, broader resource/lifetime,
fresh dense-reference corroboration, or three-independent-pair final gates.
The final majority target remains at least 20/30 wins, with each winning case
faster in every pair and a median speedup above 1.05. No A100/RTX5090, CBS,
universal MD, or whole-molecule custom-RSH UKS qualification is claimed.
