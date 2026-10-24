# Screened HF full-energy refinement

Opt-in diagnostic; not default/production qualification. Master base
`4e28477e96d257528a60bcb68c00529f22585bc1`. Source identity
`c7a15685cd6709d078e3de239c6190261384cc38b9526563af222f1977fac108`;
library SHA256
`a91856f07320a0eb7c1350591a32a819715960f6be758af7c907c26663fbfeb4`.
Release CUDA 12.9.1/sm120, scalar CPU linalg, generic/non-AOT.

Finite Slurm n1/RTX 5090 jobs 8810/8812 pass native and focused sanitizer
checks; existing full-HF synccheck blocker remains. Jobs 8811/8814 retain
90 accepted endpoints (72 formal, 18 priming), with no retry/discarded target
work/fallback. CPU PySCF E/F gates: 1e-8 Eh/1e-7 Eh/Bohr; observed maxima
1.319e-11/9.330e-11. SCF E/D/screening: 1e-12/1e-9/1e-12, max100.
The formerly failing 384-AO moved point converges in 16 steps:
4 full + 12 delta SCF, then 2 final full. All fixed8 warm points:
1 full + 0 delta + 0 final full.

Four-repeat OFF/fixed8 median endpoint seconds:

| AOs | Cold | Warm | Moved |
| ---: | --- | --- | --- |
| 96 | 1.4271/1.3822 | .1851/.1838 | 1.1325/1.0787 |
| 192 | 13.2643/12.6283 | 1.9177/1.9451 | 11.2413/9.6951 |
| 384 | 47.2921/44.4167 | 5.6044/5.6018 | 41.8924/32.3607 |

Cold includes setup through host E/F; common bootstrap excluded, teardown
separate. Warm is parity, not a uniform win. 384-AO order is AB/BA/AB/BA;
96/192 order is AB/BA/BA/BA, a timing limitation. Streaming quartet work
at 192/384 AOs is unknown, not zero.

`validation.json.xz` retains scalar samples, references, hashes, harnesses
and validation logs. Returned force vectors/full packet remain only in ignored
local `.artifacts/delta-anchor/full-energy-refinement/`, not a permanent
remote archive. No budget increase or old-evidence deletion.
See [decision and limitations](../../../.agents/notes/implemented/numerics/2026-10-15-incremental-hf-full-energy-refinement.md).
