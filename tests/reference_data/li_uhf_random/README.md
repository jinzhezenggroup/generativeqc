# Fixed neutral Li/UHF random-amplitude oracle

This is one fixed neutral Li/STO-3G spherical UHF profile at the origin:
charge 0, spin 1, 5 AOs, and occupied-alpha/occupied-beta/virtual-alpha/
virtual-beta populations 2/1/3/4. The original default-minao UHF uses
conv_tol=1e-13. Its random CC amplitudes use NumPy default_rng(37), scale 0.04,
and all 37 independently packed draws in t1a/t1b/t2aa/t2ab/t2bb order.

## Authentic acquisition

Two fresh CPU-only Python processes independently built the molecule, solved UHF,
transformed the integrals, and evaluated the PySCF UCCSD update, correlation
energy, and slow/optimized triples formulas. No GenerativeQC implementation was
imported, and no CC convergence or Lambda solve was substituted for this profile.

- Repository: njzjz-bot/vibeqc
- Acquisition checkout: 767b110a4ad75a784fe2b297d7f1ba3daa0ec8e6
- Run: https://github.com/njzjz-bot/vibeqc/actions/runs/38252729108
- Successful job: https://github.com/njzjz-bot/vibeqc/actions/runs/38252729108/job/114822875145
- Artifact: 11748277369; 40,540 bytes; SHA256 d42dcb30c70ad713405013dcd31c8298fa071a8747438d6c901bee3a92c4055b
- Historical generator SHA256: daad51defd9de8945182f109d3c9c6c482c8bf43350b0bd303f4896a3eb4e460
- Python 3.12.12, PySCF 2.14.0, NumPy 2.3.5, SciPy 1.17.0, h5py 3.16.0, threadpoolctl 3.6.0
- Distinct process IDs: 2251 and 2268; maximum difference over all full arrays: exactly 0

The generator used an explicitly empty regular PySCF configuration file, one
native thread per provider, distribution-owned scientific imports, and hashes
of installed scientific Python/native files, wheel RECORDs, native provider
basenames/bytes, pinned STO-3G data, and pinned upstream triples source files.
No environment or memory dump is retained.

## Complete retained data

All 48 arrays/scalars use lossless little-endian C-order IEEE FP64 bytes encoded
as canonical base64, with exact shapes, byte counts, and SHA256 digests. This
includes the full AO overlap/core/density/Fock matrices, separate alpha/beta
coefficients/occupations/orbital energies, both MO Fock matrices and ERI-object
energies, complete 5^4 AO ERIs and 10^4 antisymmetrized spin-orbital ERIs, all five
complete amplitude/update/denominator/residual blocks, all 37 packed draws,
HF/correlation energies, four independent triples channels, and both triples
totals. The zero beta-beta amplitude block and impossible aaa/bbb/abb channels
are retained explicitly.

The spin order is occupied-alpha, occupied-beta, virtual-alpha, virtual-beta.
The integral convention is g[p,q,r,s]=(p r|q s)-(p s|q r), including exact spin
selection masks. Resolved basis contractions, atom coordinates, structured and
printed AO labels, shell AO locations, spin populations, and actual SCF settings
are retained. No coefficient-column phase or degenerate-orbital rotation is
silently canonicalized after acquisition.

Byte-exact acquisition-first.json (147,004 bytes) and acquisition-second.json
(147,005 bytes) are retained separately. reference.json (147,435 bytes) is the
first complete receipt plus the successful independent replay receipt.
manifest.json binds every file and all array metadata. The reader checks both
original objects against the complete validated record, preserving their own
process/replica identities. No values were inferred from GQC or pass/fail logs.

## Normal test coverage

Only the exact Li seed-37 cases in the random UCCSD equation gate and the
four-channel complete/partitioned triples gate consume this record. The GQC
calculations, all five residual blocks, correlation energy, each channel,
impossible-channel zeros, and complete/partitioned tile calculations remain
active, with the original 3e-11 equation and 3e-13 triples tolerances.
The current GQC amplitude layout is still exercised and compared exactly with
the independently enumerated full amplitudes. Regression controls execute both
fixed numerical cases while PySCF imports and import-based skips are blocked.

Dynamic molecular cases, independently converged CC amplitudes, Lambda,
restricted-limit profiles, perturbations, and the pinned upstream source
identity gate remain distinct tests. This slice does not freeze those profiles.

## Explicit historical regeneration

Ordinary CI never runs the acquisition workflow or generator. The temporary
automatic acquisition workflow is removed from this change. To reproduce these
specific receipts, use the full pinned acquisition checkout 767b110a4ad75a784fe2b297d7f1ba3daa0ec8e6,
including its historical generator, audit helper, original numerical test
sources, and inputs, with the exact Python/package pins above. That checkout
contains the reviewed bounded CPU workflow and its two-process commands.

Do not run the historical input manifest against modified consumer sources and
then update the old hashes to make it pass. The generator intentionally rejects
changed original source bytes. Later formatting of the current tool does not
change the historical generator digest recorded in these receipts. A new
acquisition or profile requires independent review and new provenance.

All synthetic mutations exist only in negative tests. They are never fixtures.
