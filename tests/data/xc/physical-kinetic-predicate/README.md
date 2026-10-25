# TPSS boundary inputs

`tpss.json` and `rks-moment-rounding.json` retain the existing physical kinetic
predicate and restricted AO-rounding controls.

`empty-beta-h2plus.json.gz` retains the 1,728-point, two-AO H2+ empty-beta capture
and all seven historical directional rows from PR #2276, Slurm job 7693. It is
input/provenance data, **not a passing physical-birth qualification**. The subset
retains AO jets, raw density/gradient/tau features, weights, points, density,
orbitals, overlap, and historical secants/contractions. Unused full coefficient
arrays are omitted. JSON float roundtripping is lossless and gzip mtime is zero.

The original source was `6fb6d8277c37a8a362cfb62d5979c71926c1055a`. Original
capture SHA256 is
`a22d85fd7f50ce10e692ada32db12213a09cf7259e8d650ec9442edd223ba71a`;
classification SHA256 is
`658ca32db8417b49f1aa3f2f10d45615eb0e414bc4628045a5a89249d71de7ca`.
The retained gzip SHA256 is
`c5e5eb64e717c0a07ddd7b07097764d5d5aa400b40ee6e786f0884fe96abbdf9`.
The JSON also records original library, driver and source-manifest hashes.

Replay with PySCF 2.14.0, Libxc 7.0.0, a host C++ compiler and verified
sccache/ccache:

```sh
PYTHONPATH=python:. python benchmarks/replay_tpss_empty_spin_boundary.py \
  --output .artifacts/tpss-empty-beta-replay
```

This compiles the current canonical point and compares energies on identical
rounded raw features with Libxc. AO orbital births increment those features
linearly. Historical native CUDA secants are kept separately; this host replay
does not reproduce the historical native AO reduction or time a KS endpoint.
The density-only subfloor control has an independent analytic energy derivative
with a `1e-10` acceptance gate. Other short secants are diagnostics, not asserted
one-sided derivative limits. See the accompanying Agent Note for the remaining
physical-birth blocker.
