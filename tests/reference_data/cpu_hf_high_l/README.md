# Fixed mixed-spdf CPU HF reference

These are authentic PySCF results for the eight fixed cases originally tested by
`tests/python/test_cpu_hf_high_l.py`: RHF/UHF, Cartesian/spherical, and both
hydrogen-x displacements (0 and 0.013 Bohr). Geometry, explicit contracted basis,
charge/spin, initialization, SCF tolerances, units and force sign are retained in
`inputs.json` and repeated in each record. No GenerativeQC output supplied an
expected value.

## Acquisition and integrity

- Source: [fork acquisition commit](https://github.com/njzjz-bot/vibeqc/commit/3c490a652b2844fe88b26379c97ea4992b23e7c7)
- Actual execution: [successful CPU job](https://github.com/njzjz-bot/vibeqc/actions/runs/38217314542/job/114704971638)
- Python 3.12.12; PySCF 2.14.0; NumPy 2.3.5; SciPy 1.17.0;
  h5py 3.16.0; threadpoolctl 3.6.0
- Every case converged in two separate interpreter processes. Maximum replay
  differences were exactly zero for both energy and every force component.
- The generator enforces an empty regular PySCF configuration file, owned
  distribution imports, single-thread native pools, complete finite FP64
  shapes, exact case identity, and a 256 KiB cap on each receipt.
- `reference.json` is the unchanged 30,325-byte validated receipt retrieved from
  the job log. Its SHA-256 is
  `0770d91ec511fd8db30256ddbb308104afa638793947cd7014e008a5d232673d`.
- `manifest.json` binds the input and receipt bytes to the actual generation
  commit/job, generator hash, two raw-replica hashes and the bounded ZIP artifact.
  The receipt additionally records actual installed package/native-library
  identities, AO ordering/counts, electron counts and resolved SCF settings.

The initial workflow attempt stopped before dependency installation or numerical
execution because uv 0.10.7 does not provide Python 3.12.14. Only the interpreter
pin was changed to its supported Python 3.12.12 catalog entry. The successful
receipt records the actual interpreter; it does not claim the failed attempt
generated data.

## Regeneration

Regeneration is explicit and outside ordinary numerical CI. The fork workflow
is manual-only after this acquisition; it installs the pinned independent
dependencies and runs the standalone generator twice, followed by `--verify`.
No credentials, GPU job, GenerativeQC build or production fallback is required.

For an exact historical replay, check out the acquisition commit above and use
its workflow or standalone generator. Use a fresh output directory: receipt
creation is exclusive and will not silently overwrite existing results. Do not
edit numeric values or combine records from different inputs/environments.
Review any regenerated receipt, package/source identity and numerical changes
before replacing these records.

The consuming GQC tests must continue computing both energy-only and complete
energy/force endpoints, with the original convergence/backend assertions,
absolute energy tolerance 1e-9 Hartree, absolute force tolerance 1e-8 Hartree/Bohr,
zero relative force tolerance, and translational sum tolerance 2e-10 Hartree/Bohr.
This retention commit does not yet change those tests.
