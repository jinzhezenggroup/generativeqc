# Evidence: installed Ni UKS total curvature, source cohort e5c601

Status: implemented (qualified installed cohort only; no broader admission)
Date: 2026-11-11

## Scope and decision

The complete installed public CPU Hessian and two fresh public HVP endpoints
passed their original gates for all-electron spherical def2-SVP NiH3, charge
zero, doublet, 46 AOs, occupations `(16, 15)` and 50,064 native-default grid
points. The method is physical Gaussian/VWN-RPA B3LYP with exactly-once uncut
two-body D3(BJ), ATM off. This is a four-atom curvature control, not a transition
state, ECP, mixed Ni/I/Cs, scaling or Python-free CLI qualification.

Acceptance belongs exclusively to installed source commit
`e5c60110175b658459a81c0a7e83dcfd589877d0`. The later checkout and published B88
fix `53e733d9fc50dc81b8caf5a893211c9434569942` are not requalified by this run.
The installed prefix stayed immutable throughout. Independent displaced SCFs
and production endpoints did not cross-seed densities; production callables
were not replaced and production matrices were not replayed.

## Complete numerical gates

The driver actually terminated with exit zero. Its receipt records
`stage=accepted`, `public_installed_total_qualified=true` and
`sources_unchanged=true`. The complete raw 12-by-12 matrix differs from three
independently displaced-gradient stencils by respectively
`1.9040776164658268e-6`, `4.925145997702884e-7` and
`2.059900112705293e-7`, below the unchanged `1e-5` gate. The raw matrix was not
post-hoc symmetrized: asymmetry is `4.2634451524747874e-11`, translation error
`4.672240372372016e-11`, both below `2e-8`.

Two separately evaluated public HVPs agree with the raw matrix actions to
`1.7274298311220626e-12` and `2.1372487113424654e-12`; bilinear error is
`5.471892470382267e-14`. They also agree with each of the three independent
matrices below `1e-5`. Maximum response residuals are
`1.5455936231119348e-10` for the full matrix and
`2.83622151735528e-12` / `4.2669709076623005e-12` for the HVPs, below `2e-10`.

## Endpoint work and provenance

The three fresh production endpoints take `23637.5420286851` seconds in total;
the driver takes `23638.519885` seconds, and qualification including retained,
verified preparation takes `24538.44709784072` seconds. Production performs
369 SCF iterations, 393 Fock builds and 14 nuclear response solves. These are
concurrent CPU-load measurements, not isolated performance or scaling evidence.
The retained outer `/usr/bin/time -v` receipt reports exit zero, `6:33:59` wall
time and maximum RSS `3,067,580` KiB for that complete installed run. Peak RSS
does not replace the per-owner resource/work counters or establish scale gates.

Evidence root: `build/qualification/public-ni-uks-20261111/`.

- Driver: `/tmp/gqc-public-ni-uks-installed.py`, SHA256
  `f555e52757c245c84529355fe4a072f50d132a44006e2c321fed4d1b8453ac46`.
- Installed prefix: `/tmp/gqc-public-uks-e5c60110175b-repaired`; library SHA256
  `7ff0721c476b1eddacb4038973b5805cb2d46df784c910ebfa61bcaaf172d7bc`;
  native source identity
  `be8cdeccb535a385e9ef8eb69de752d2ca54f7258255d77b33f9c4ce2cb6ac74`.
- Repaired wheel: `repaired-with-blas/` under the evidence root, SHA256
  `73e430f9a1c0c6012ffd0fca533fa00865a91a30ce2b22788f392d78e7c9d9df`.
- Accepted receipt: `public-endpoint-json-contract/receipt.json`, SHA256
  `d85f247bfed0d804157aecdee1a63fef85683a7959e4e1f6fa3921fc10f1dd58`;
  source-provenance SHA256
  `a045432a16a866e8b1b87db02d2d7880c86956c9c92e6536c1774ad6738f0b7e`.

The separately completed postrun verifier rehashes every declared source,
all 72 retained independent point receipts and arrays, the repaired wheel and
installed library. It recomputes independent-matrix HVP comparisons and checks
matrix versus HVP diagnostic schemas separately. Two earlier verifier-only
schema probes failed and remain retained: the actual representation is
`real_spherical`, and matrix-only symmetrization fields are not HVP fields.
Those probes are neither native endpoint failures nor a reason to change gates.

## Local recovery and boundaries

The completed local archive
`accepted-e5c601-local-recovery/installed-ni-e5c601-total-curvature.tar.gz`
contains 2,371 members, 154,613,411 bytes, SHA256
`e858aba1d3d1c3f4336e9d41bc5ad345fdb30b35af296a2d4c457efa07167a07`.
Its manifest SHA256 is
`a741a058a0c969279efdbfca68740982a738d9ed22fcd4b3ea1115f7bd7b7e11`.
The archive generator actually exits zero and its verification receipt confirms
that every member and original was rehashed. It retains cohort/JIT assets,
the installed prefix and repaired wheel, all independent points, provenance,
the original failed preflight and all verifier probes. Running IH2 and SiH3
outputs are excluded. No release, remote asset or external backup is created.

Earlier output-budget/preflight receipts and original installation metadata
retain their original false qualification flags. Do not rewrite them, attribute
this acceptance to newer sources, or use it to open public ECP/TS/scale gates.
Current-source installed qualification, broader public ECP and mixed-element
controls, bounded 25–40-atom endpoints, the requested-method representative Ni
saddle and final frequencies, and a complete native CPU owner remain required.

## References

- Issues #2844–#2848; PR #3062.
- `docs/developer/dft_hessian.md`.
- `public-endpoint-json-contract/postrun-verification.json` under the evidence root.
- `accepted-e5c601-local-recovery/verification.json` under the evidence root.
