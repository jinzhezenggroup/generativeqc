# Required native progressive-HF policies: historical CPU results

This compatibility migration requires the current native planning and final
verification protocols for bounded integers. It does not change scientific
equations, acceptance thresholds, ABI or backend admission. Arbitrary-width
integer behavior is unchanged. Missing independent observed accuracy can still
leave final accuracy unverified.

Author qualification at production `e2157668e` used baseline
`131144549f91223cf751a27874c365834cc7c443`. The current library SHA-256 was
`fa9450537f4018bfb472503a8de41b90ac69abc5e321cdd83e1b4e65e3a51b8a`.
Two separate CPU image campaigns each contained 672 timed complete endpoints:
RHF/UHF x Direct/DF, two geometries, ABBA/BAAB, three warmups and 21 samples.
Reported maximum median ratios were 1.0016859405931222 (current image) and
1.0041193828828907 (preceding shared image), below the unchanged 1.05 gate.
Corresponding energies, full forces/density, physical residuals and semantic
work matched exactly in the reviewed populations. No CUDA, exclusive-host,
speedup or later-master qualification is claimed.

The author also reported 127 CTests, 95 focused Python checks and installed
public-header SDK probes. These are historical receipts, not fresh test runs.
[Review and verification limits](https://github.com/jinzhezenggroup/generativeqc/pull/2879#pullrequestreview-5694073201).

The optional 27,524-byte raw capsule is not added to the merged checkout.
[Original PR capsule](https://github.com/jinzhezenggroup/generativeqc/blob/7ca4b35058d503292318f01b6fbec63cc373649c/benchmarks/results/required-hf-policies-20261029/evidence.json.xz)
contains both populations, replay source, identities and receipts; SHA-256:
`0e84b308694db716159856c1f7b9660f238542afd7910a5ba7ca301c428f7540`.
This PR-only history link is not a protected-history archival guarantee:
a clean offline master clone cannot be assumed to recover or replay it.
No sample values were filtered or rewritten, and no new run was requested.
