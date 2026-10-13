# Fixed converged Li/UHF Lambda oracle

This fixture replaces only the exact neutral `Li 0 0 0`, spin 1, unbroken,
STO-3G spherical entry in `test_complete_cpu_lambda_against_independent_pyscf`.
It preserves the complete original SCF-polish/UCCSD/Lambda protocol and all five
independent multiplier blocks. It is distinct from the seed-37 random-amplitude
fixture; no random-profile results or claims are reused here.

## Authentic bounded acquisition

- Fork branch: `njzjz-bot/vibeqc`, `oracle/li-uhf-lambda-reference-20261012`
- Acquisition checkout: `8edcf95601c8296bd260ed3c09eb2417da0d63f3`
- Original consumer source: `c7bf888361f57685b6c21cf7922c2f95df6c7bf8`
- [Successful run](https://github.com/njzjz-bot/vibeqc/actions/runs/38285625548)
- [Successful CPU job](https://github.com/njzjz-bot/vibeqc/actions/runs/38285625548/job/114933178178)
- Artifact: `11762544362`, 23,355 bytes
- Artifact SHA256: `d18dc4bcabe4d89584261627b0774312f9eee1cf88e4fb8a5a5166d1f77f693b`
- Generator SHA256: `d97ba385f3179d334212a664dad2a911662caa2a3de417612236b1e6c0cab7aa`
- Python 3.12.12; PySCF 2.14.0; NumPy 2.3.5; SciPy 1.17.0;
  h5py 3.16.0; threadpoolctl 3.6.0

Two fresh CPU-only processes, IDs 2581 and 2674, independently rebuilt the
molecule and converged UHF, UCCSD and Lambda. Both retained full-array receipts
agree exactly, with maximum absolute difference 0. The workflow had a 10-minute
bound, exact fork/branch guard, pinned packages/actions and one native thread
per provider. No GenerativeQC scientific implementation was imported.

Each receipt uses an explicitly empty regular PySCF configuration file,
distribution-owned imports, hashes of installed scientific Python/native files
and wheel RECORDs, native provider basenames/hashes, resolved basis/AO identity,
and pinned installed UCCSD/Lambda source. No credentials or environment dump is
retained. The original source/helper hashes identify historical acquisition
inputs, not permanent fingerprints of production code.

## Complete data and protocol

All 50 arrays/scalars are complete lossless C-order little-endian FP64, with
canonical base64, exact shape, byte count and SHA256:

- Full AO overlap, core, alpha/beta density and Fock matrices; separate alpha/beta
  MO coefficients, orbital energies, occupations and MO Fock matrices
- Full 5×5×5×5 AO ERIs and both ERI-object orbital-energy arrays
- Every element of `t1a`, `t1b`, `t2aa`, `t2ab`, `t2bb`, independently converged
  Lambda blocks, and their respective one-step PySCF updates
- Complete CC denominators and physical residual blocks, HF/correlation energy,
  and the polished SCF gradient norm

Populations are occupied-alpha/occupied-beta/virtual-alpha/virtual-beta 2/1/3/4.
Mixed-spin axes remain occupied-alpha, occupied-beta, virtual-alpha,
virtual-beta. Same-spin antisymmetry and every explicit zero beta-beta block
are retained. No coefficient-column phases or degenerate-orbital rotations
were canonicalized after acquisition.

The original default-minao UHF uses `conv_tol=1e-13`. The stationary polish
uses `direct_scf=False`, `conv_tol_grad=1e-10`, `max_cycle=300`, and its previous
density. The original Newton fallback remains in the generator and was not
needed in this acquisition. UCCSD uses `conv_tol=1e-13`,
`conv_tol_normt=1e-11`, `max_cycle=400`; `solve_lambda(eris=eris)` uses that same
norm tolerance and iteration limit. All three convergence flags are checked.
Complete independent update/residual consistency and Lambda fixed-point
checks are additional acquisition integrity gates.

Byte-exact first/second receipts are 41,800/41,801 bytes; the validated receipt
is 42,231 bytes. The manifest binds inputs, both raw receipts, the validated
record, all 50 arrays and the acquisition provenance. Readers reject missing,
corrupted, incomplete, nonfinite, differently identified or unconverged data.

## Preserved numerical coverage

Normal CI constructs GQC's `UHFReferenceSnapshot` and dense AO source from the
independent saved arrays, then runs the existing `ConventionalProvider`,
`PreparedUCCSD` and `PreparedUCCSDLambda` calculations. All five multipliers are
still compared with `atol=2e-7, rtol=0`. Original primal/stationarity/residual,
backend, semantic-work and memory-bound assertions remain unchanged.

The shared `_problem` helper is unchanged. Other molecules, restricted-limit
comparisons, NH2 reconverged perturbed-Hamiltonian finite differences, corrected
Lambda/physical-response consumers, CUDA cases and ownership/failure tests
remain distinct live tests. There is no fake PySCF object or fallback from the
frozen reader to a live PySCF call. GQC's compilation/solve cost remains; this
change does not claim to remove the whole test's runtime.

## Explicit historical regeneration

Normal CI never runs this generator. The temporary acquisition workflow exists
only on the fork acquisition branch and is absent from the canonical change.
To reproduce the specific receipts, use the complete pinned acquisition
checkout above, its original input/source files and exact Python/package pins.
Its reviewed workflow contains both independent process commands and replay
verification. Running the old manifest against modified consumer sources is
intentionally rejected; a changed profile needs a new reviewed acquisition.
Synthetic mutations in the negative tests are never reference data.
