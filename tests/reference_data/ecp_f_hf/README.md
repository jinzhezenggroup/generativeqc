# Fixed NaH orbital-f HF quartet

This is the complete authentic PySCF oracle for exactly four fixed endpoint
profiles in `test_ecp_f.py::test_f_complete_hf_energy_and_force`: RHF/UHF ×
Cartesian/spherical NaH, with 19/16 AOs respectively.

## Exact scientific inputs

Na (0.13,-0.21,0.17) and H (0.43,0.19,3.2), in Bohr; Na LANL2DZ orbital/ECP
parameters and H STO-3G orbital parameters from pinned PySCF 2.14.0. Append only
Na orbital shell `[3, [0.55, 0.81], [0.17, 0.23]]`. RHF uses charge=spin=0 and
populations (1,1); UHF uses charge=spin=1 and populations (1,0). There is no extra
d shell, H f shell, f projector, displacement, or compact-oracle substitution.
Both fresh processes independently reload byte-pinned upstream basis files and
reconstruct the same resolved inputs.

The original call is preserved exactly:
`(scf.UHF(mol) if spin else scf.RHF(mol)).run(conv_tol=1e-12)`.
RHF uses the original default minao guess and converged in eight cycles. For the
one-electron charged profiles, the original UHF factory returns
`pyscf.scf.uhf.HF1e`: its configured `init_guess` property is minao, but the
analytic one-electron solution uses no initial guess and zero iterative cycles.
All HF1e alpha/beta coefficients and orbital energies retain the original hcore
eigensystem, including virtual/beta orbitals. No iterative UHF, polishing,
retry, phase adjustment, or Fock-canonicalized virtual spectrum is substituted.

## Authenticated acquisition

- Repository: `njzjz-bot/vibeqc`
- Separate branch: `oracle/ecp-f-hf-reference-20261012`
- Exact acquisition checkout: `2c0458410387d211e3b1e5e19bb3694775390e17`
- [Successful CPU run 38303387532](https://github.com/njzjz-bot/vibeqc/actions/runs/38303387532)
- [Successful job 114991477301](https://github.com/njzjz-bot/vibeqc/actions/runs/38303387532/job/114991477301)
- Artifact: `11770584151`; 155,737 bytes; SHA256
  `41fa7f217b45600399a50b82e93580a31092002d642f43a9e856fba395893ba3`
- Historical generator SHA256:
  `8b1a289607876c50062fa70a2714bed62881c23355160ab985547bb66d654988`
- Inputs SHA256:
  `20f95e88f94c8598e62a20456290b2ceeaf2d2bdb4715341305c58596ab61676`
- Original manifest SHA256:
  `35a7ef21310483b7293d7da409e1e4ebf49622e40ed741582092c9c798644161`

Python 3.12.12; PySCF 2.14.0; NumPy 2.3.5; SciPy 1.17.0; h5py 3.16.0;
threadpoolctl 3.6.0; setuptools 83.0.0. The temporary fork-only job was limited to
10 minutes on ubuntu-24.04, with one native thread per provider and hidden CUDA
devices. It installed no GenerativeQC implementation and ran no GQC/GPU solve.

Eight distinct fresh process IDs were 2459, 2467, 2474, 2482, 2489, 2496, 2503,
and 2511. Each exact profile ran twice. All 128 complete arrays agreed exactly
between the two processes; the reviewed replay acceptance bound was 1e-13.
The downloaded ZIP passed CRC and exact nine-file inventory checks. Every
extracted original file's byte count/hash matched the successful job log and the
manifest. The bytes in this directory are those exact downloaded originals.

Independent review checked acquisition/source/workflow identities and complete
retained-state consistency, including density/Fock reconstruction, all forces,
net forces, orbital gradients and the HF1e hcore eigensystems. Nothing was
inferred from GQC outputs or fabricated from selected entries/pass-fail logs.

## Complete retained arrays and provenance

Each profile has 16 complete losslessly encoded C-order little-endian FP64
arrays/scalars, with exact shapes, byte counts and SHA256 hashes:

- Full overlap, hcore, raw ECP and normalized ECP matrices, and AO normalizations
- Full density, Fock, coefficients and commutator residual, with both actual
  alpha/beta spin blocks for UHF and the actual single restricted block for RHF
- All orbital energies, occupations and occupied–virtual orbital gradients
- HF and nuclear energies, full analytic gradient (2,3) and full forces (2,3)

The endpoint consumes no four-index ERI tensor, so none is added. Scalar norms
never replace matrices/vectors. Resolved atom/basis/ECP data, core/effective
charges, shell/AO locations, structured/printed labels, atom AO slices,
convergence, solver class, cycles and resolved SCF settings are also retained.

All eight original process files and the original `manifest.json` are retained.
The reader authenticates the immutable input/manifest bytes and every original,
then checks both complete replicas and every array. It reuses the existing
stdlib-only bounded JSON/FP64 transport; it imports neither PySCF nor GQC.
Unknown profiles, incomplete vectors, shape/dtype/finite-value failures,
modified provenance, process mismatches and tampered originals fail closed.
Historical source hashes are acquisition provenance, not current-source CI pins.

The reviewed generator independently binds all installed wheel RECORDs and
Python/native-file hashes/counts, pinned upstream SCF and basis files, exact
fork checkout/run, an explicit empty regular PySCF config, and the actual loaded
one-thread CPU BLAS/OpenMP provider metadata/native bytes. Replay verification
initializes only the same pinned package/provider imports, without another
molecule, SCF or gradient calculation. Setuptools is an unconditional dependency
in the official [PySCF 2.14.0 metadata](https://github.com/pyscf/pyscf/blob/v2.14.0/pyproject.toml),
and is explicitly pinned and included in the full package identity inventory.
The official [setuptools 83.0.0 release](https://pypi.org/project/setuptools/83.0.0/)
provides a universal wheel for Python >=3.10, SHA256
`29b23c360f22f414dc7336bb39178cc7bcbf6021ed2733cde173f09dba19abb3`.

## Actual reference-only timings

Seconds for first/second independent processes, measured around the original
solver `.run(...)` and `nuc_grad_method().kernel()` calls respectively:

| Profile | SCF first / second | Gradient first / second |
| --- | --- | --- |
| RHF Cartesian | 0.04455 / 0.04272 | 1.09951 / 0.13283 |
| RHF spherical | 0.04096 / 0.04145 | 0.13125 / 0.12950 |
| UHF Cartesian (HF1e) | 0.00628 / 0.00648 | 0.12917 / 0.12950 |
| UHF spherical (HF1e) | 0.00615 / 0.00617 | 0.13076 / 0.13007 |

These exclude package-provenance scanning, molecule preparation and serialization.
The first gradient observation is retained rather than discarded. The previously
observed 132.26-second hosted RHF/cartesian duration included the complete GQC
endpoint and is not measured PySCF time. This migration removes a runtime oracle
dependency and its failure surface; no two-minute or paired endpoint speedup is
claimed.

## Retained runtime and dynamic coverage

The consumer still runs the original GQC Calculator for every invoked
spin/representation/device. It retains convergence, energy atol=2e-8/rtol=0,
every force component at atol=2e-6/rtol=0, and net force atol=2e-7. The full
parameter matrix, node IDs, platform/device skips, routine CI deselections and
resource/backend tests remain unchanged. The other three CPU profiles are not
restored to routine CI by this change.

The shared fixture, raw-f all-center finite differences (2e-4 and 7e-5),
detached-center derivatives, moved-coordinate prepared replay and CUDA
directional finite differences remain live and unchanged. There is no global
fake PySCF molecule/module. Reader tests exercise full-array corruption,
shape/type/finiteness, forces/sign, historical provenance, AO/spin/solver identity
and blocked-PySCF input construction. Full CPU consumer qualification explicitly
blocks both PySCF imports and import-based skips without adding four duplicate
expensive permanent-CI endpoint tests.

## Historical reproduction

Ordinary CI never runs this acquisition. To reproduce these exact receipts, use
the full pinned acquisition checkout above, including its original consumers,
generator, helpers, inputs and temporary bounded fork workflow. That historical
checkout intentionally checks original source bytes. Do not update those hashes
to make a later consumer revision pass; a new profile/acquisition needs new
review and provenance. All synthetic mutations exist only in negative tests.
