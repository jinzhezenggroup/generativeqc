# Evidence: fresh installed minimal UKS saddle

Status: implemented
Date: 2026-11-12

## Scope

The standalone native ELF from an isolated repaired wheel completed a fresh
SiH3 doublet saddle search, starting from the original displaced geometry, with
Gaussian/VWN-RPA B3LYP, uncut two-body D3(BJ), ATM off, spherical STO-3G and the
unchanged native v4 level-3 grid. This is a genuinely open-shell **minimal-basis
control**, not def2-SVP, Ni, ECP, mixed-element or practical-scale acceptance.

This supplements the [Python-free CLI decision](../architecture/2026-11-12-python-free-molecular-saddle-cli.md).
The prior source pytest remains failed; its corrected independent post-validation
and this new installed execution are separate evidence. Neither an old timeout
nor a failed attempt is relabelled.

## Actual execution

The finite 3600-second runner and raw outputs are retained under
`build/qualification/host-installed-uks-saddle-20261112/`. The first invocation
was refused because `--grid-level` is not a saddle-parser option. The distinct
`installed-v2` attempt omitted that override and used the identical native
default grid. Both attempts, their commands and actual exit statuses remain.

The successful endpoint ran with an empty environment and nonexistent Python,
compiler and executable-search paths. It did not enter Python, a reference
engine, runtime compilation, displaced production SCFs, lower-level
preoptimization or an external optimizer. It returned in 1264.684 seconds;
`/usr/bin/time -v` records 27696 KiB maximum RSS for this native process. This is
not a speedup claim or an installed dependency-resolver/portability qualification.

Actual installed identities:

- ELF SHA256: `4b370e871dc3dfab64d683c75f888a7535d0bb4bb1a4d00b4e7fbd33d677802e`.
- Core SHA256: `3e4781c2551e9d3857e39f82f1af1fa6cdb86b169e84182657087fd7ae2e375d`.
- Native source identity: `a4d6cb5e23b39dac184cfaf6a4a23de5d544dded74c031d11a969d88eb411f56`.

Three accepted attempts, five energy/gradient evaluations and five complete
Hessians produced a freshly verified first-order saddle. The final independent
controls were energy change `3.673e-7` Hartree, gradient RMS/max
`1.412e-7`/`2.129e-7` Hartree/Bohr and displacement RMS/max
`2.612e-4`/`3.917e-4` Angstrom. Their original thresholds remain, respectively,
`1e-6`, `3e-4`, `4.5e-4`, `1.2e-3` and `1.8e-3`. The attempted-step budget
remains 100, including rejected attempts. Every accepted electronic record
has two spin channels with occupied counts 9 and 8.

## Independent final-result audit

The separate `validate.py` authenticates the actual binaries, input, output and
parent archive/member hashes. It compares the new installed final coordinates,
gradient and **raw** matrix with the immutable source result and three sealed
full fourth-order independent-gradient matrices. These references include 144
cold displaced states and an occupied-root minimum of `0.9988586907`; this
audit reuses them rather than claiming fresh independent SCFs.

The installed raw asymmetry is `4.627e-14` Hartree/Bohr squared. Largest raw
matrix errors for `0.003`, `0.001` and `0.0003` Bohr stencils are
`8.098e-11`, `1.149e-9` and `5.601e-9`, below the unchanged `1e-5` gate.
Frequency error is at most `9.938e-6` cm^-1 against the unchanged 1 cm^-1 gate.
The audit checks mass-weighted and Cartesian mode normalization and projected
independent-Hessian eigenmode residuals, not arbitrary signs/bases within
degenerate subspaces. All six unscaled vibrational modes remain, with exactly
one imaginary frequency, `-1024.6858603` cm^-1.

Harmonic analysis may average its own copy only after the original raw-symmetry
gate; the published matrix remains untouched. An installed/public ECP saddle,
the requested exact-level Ni transition state and complete 25--40-atom electronic
work/time/memory remain separate unfulfilled gates.
