# Evidence: canonical IH2 doublet ECP total curvature

Status: implemented (low-level canonical control only; public admission closed)
Date: 2026-11-12

## Scope and independent state

The separately declared, nonaxial IH2 doublet control actually exits zero with
its complete raw 9-by-9 matrix and two fresh HVPs accepted. It uses physical
Gaussian/VWN-RPA B3LYP plus exactly-once uncut two-body D3(BJ), ATM off,
real-spherical def2-SVP, scalar I def2-ECP28, charge zero, occupations `(14, 13)`,
36 AOs and 41,574 native-default-v4 grid points. Density fitting is absent.
This fixture is not a retry or acceptance of the original failed IH+ oracle.

Native and independently converged central states differ by
`1.0800249583553523e-12` Eh in energy, `1.8509649368780856e-10` in maximum
density element and `1.609461897089659e-9` Eh/Bohr in gradient. The independent
orbital-gradient norm is `1.4542025624676514e-11`, below the original `2e-9`
gate. No native/oracle cross-seeding or production displaced SCF is used.
The independent references use the identical discretized grid and input basis.

## Complete numerical evidence

Full raw total matrix errors against independent displaced gradients at
`0.003`, `0.001`, `0.0003` Bohr are respectively
`5.607519210609757e-6`, `6.270560905796696e-7`,
`7.37471833045511e-8`, all below the original `1e-5` gate. Electronic-only
errors pass that same gate. D3 matrix errors are
`7.583239995384431e-10`, `8.425698542189643e-11`,
`7.582833456466504e-12`, below the unchanged `2e-9` gate.

Raw symmetry is `2.507216656510991e-12`; translation error is
`2.968861267937939e-12`. Neither post-hoc symmetrization nor removal of modes
is used. Two freshly evaluated total HVPs agree with the matrix actions to
`0` and `1.2997034004591512e-12`. Their comparisons with all three independent
matrices pass `1e-5`; the derived bilinear error is
`1.3644523878808545e-12`. The full-matrix maximum response residual is
`6.1550358686728915e-12`, below `2e-10`.

All eleven required electronic/ECP/grid/Pulay/nuclear/D3 sources are present.
The ECP owner reports analytic Gaussian execution, with no radial-grid layers,
AO quadrature points or diagnostic refinement substituted into production.

## Work, artifact identity and retained data

The full control takes `7367.209976562299` seconds; native preparation, complete
matrix and two HVPs take `6973.659807978198` seconds. The outer time receipt
records `2:02:48`, exit zero and maximum RSS `2,309,796` KiB. The central SCF
takes 32 iterations and 41 Fock builds. Full-matrix diagnostics record nine
response solves, 153 response iterations and 324 grid actions. These are not
complete installed endpoint totals: separate HVP iteration/work diagnostics are
not persisted in that receipt. Concurrent scientific CPU jobs were present;
this is not isolated timing, scaling or representative large-system evidence.

Evidence root: `build/qualification/canonical-ih2-uks-total-20261111/`.

- Driver: `/tmp/gqc-canonical-ih2-uks-total-matrix.py`, SHA256
  `df03080b4cff083c1c96349a82e4f4ac38ee78fd51d3bf66c49197e35fb57eec`.
- Frozen native library: `build/wheel-public-saddle-cpu/libgenerativeqc.so`,
  SHA256 `afb07c88c86bcfa2959cfa35bfcc138266c0d6fed8fe5abe16cac02b10ac0ec9`.
  The receipt labels it compiled before the subsequent checkout commit;
  acceptance authenticates this binary and declared sources, not a rebuild of
  every current uncommitted file.
- Accepted receipt SHA256:
  `6e2f4120f061dffca95c63190be97502a721e1df7e3dfed616fa9e85ed9dce17`.

The completed postrun verifier reconstructs all 27 independent matrix columns
from the 54 retained displaced-gradient points, compares them byte-for-value
with the retained independent matrices, rechecks the raw matrix/actions and
spin populations, and rehashes all four declared numerical sources, driver,
library, basis and license. Its source is
`/tmp/gqc-canonical-ih2-postrun-20261112.py`, SHA256
`8217e9f88af6a59ef13ce4e0b4333bf6930364f2149edc03458a7fd15018641b`.
It actually exits zero. The initial verifier-only probe selected the Ni-only
basis-manifest directory rather than `global_hybrid_ecp`; its original script
and failing log remain retained. No scientific gate is changed to pass it.

## Verified local recovery and exclusions

`build/qualification/canonical-ih2-completed-recovery-20261112/`
contains `ih2-uks-complete-total-curvature.tar.gz`: 369 members, 46,591,784 bytes,
archive SHA256
`16654bd77ff47ad40abf1ba32134bc79878333c56a9ec8e2ca6016ae9527ecfd`,
manifest SHA256
`993fce48f9e2e4aa26fc6c8623564d98dced213562c0d2ea2fedc4af3f01b74a`.
Every archived member and original is rehashed after archive creation. The
archive includes retained matrices/gradient columns/JIT source, declared
scientific sources, basis/license, frozen library, drivers and completed logs.
It excludes the live SiH3 run. Earlier IH+ failures and recovery archives remain
unchanged. No release, remote asset or external backup is created.

The original receipt keeps `public_ecp_qualified=false`,
`installed_qualification=false` and `mixed_ni_or_scale_qualification=false`.
Do not promote public B3LYP/PBE0/ECP, mixed Ni/I/Cs, 25–40-atom resource/work
qualification, native TS or final frequencies from this control alone. Native
CPU stationary ownership and installed Python-free CLI remain separate gates.

## References

- Issues #2844–#2848; PR #3062.
- [Original ECP spin gate and retained failures](../../proposed/2026-11-11-analytic-ecp-uks-spin-response-gate.md).
- `postrun-verification.json` under the evidence root.
- `verification.json` under the completed recovery directory.
