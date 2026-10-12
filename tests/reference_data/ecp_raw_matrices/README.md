# Fixed raw ECP matrix references

These are genuine independent PySCF 2.14.0 results for exactly four existing
`test_raw_matrices_libcint_and_quadrature` cases: spherical/cartesian NaH,
with/without the existing Na d shell. The molecule is in Bohr, with charge and
spin zero. Na uses LANL2DZ basis/ECP and H uses STO-3G.

## Acquisition

- Repository: `njzjz-bot/vibeqc`
- Checkout: `cda5457f9fdbab43674a0826f0b634e433f9921f`
- [Successful CPU run 38237868353](https://github.com/njzjz-bot/vibeqc/actions/runs/38237868353),
  job `114772903727`
- Python 3.12.12; PySCF 2.14.0; NumPy 2.3.5; SciPy 1.17.0; h5py 3.16.0;
  threadpoolctl 3.6.0
- Generator SHA-256:
  `e03529770f7639075a149c1a0829466cf5885eacaf332fc09daf238c259d40b5`
- Imported audit-helper SHA-256:
  `baf613386e7c00c3524ac9cddf3f9396c202de5b7c3017fd263a52ea1b1f1700`

Two independent Python processes (2392 and 2408) each constructed all four
molecules afresh. Their complete scientific records and provenance agree
exactly; the bounded replay tolerance was 1e-13, and measured maximum difference
was zero. Both raw replicas are retained byte-for-byte alongside the validated
receipt. The manifest records their file hashes and the GitHub artifact identity.
The JSON bytes were recovered from the generator's bounded log envelopes and
verified against the reported SHA-256 and byte counts; all packed FP64 array
hashes were independently recomputed before retention.

The receipts retain resolved basis coefficients, complete ECP radial channels,
molecule atoms/effective charges, AO labels, shell/atom AO ranges, overlap,
raw ECPscalar and normalized ECP matrices, and AO normalization vectors.
They also retain pinned distribution/installed-file hashes, allowlisted native
library hashes, original test/helper/input hashes, and upstream installed basis
data hashes. No implementation-under-test output supplies any expected number.

## Normalization and preserved numerical coverage

The independent matrix is exactly the original sequential expression:
`ECPscalar / sqrt(diag(int1e_ovlp))[:,None] / sqrt(diag(int1e_ovlp))[None,:]`.
Cartesian d-shell AOs have nonunit raw overlap norms; replacing the normalized
matrix with the raw matrix is an explicit failing mutation control.

The four numerical tests still execute both original GenerativeQC ECP matrix
and refined quadrature calculations. Their matrix tolerances remain
atol=2e-9/rtol=1e-9, derivative refinement tolerances remain
atol=2e-8/rtol=1e-8, and translational cancellation keeps atol=2e-13 and the
original default relative tolerance. Dynamic displaced-molecule and other ECP
oracles remain separate. The fixed consumer fails closed on missing or invalid
receipts and does not import PySCF or fall back to a nearby case.

## Explicit regeneration

Ordinary CI only consumes these retained records. The temporary automatic
acquisition workflow is removed after retention. To reproduce this historical
acquisition, use the exact acquisition checkout above, its input manifest,
reviewed generator and audit helper, an empty independent CPU environment with
the pinned dependencies, and invoke the generator twice in distinct processes
with `--replica first` / `--replica second`, followed by `--verify FIRST SECOND`.
Each output is exclusively created and capped at 256 KiB. The generator itself
selects an explicit empty regular PySCF configuration file.

The historical source check deliberately requires the pre-conversion original
test bytes. A future generator/input change requires a new reviewed acquisition
and new provenance; formatting or consumer edits must never rewrite the
historical hashes in these receipts. Generation is optional and never imports
GenerativeQC. Upstream resolved basis/ECP parameters are test-only scientific
data from the pinned PySCF basis libraries, not a new runtime basis source.
