# Retained CUDA AOT default-policy input

`production-shell-classes.json` is the exact stock manifest from
`44f34575f0c91b846c7265e5c388f068de09826c` (the P1b policy-dimension repair).
Its identity is retained in
`manifests/maintenance/cuda_aot_default_decisions.json`.

This is an immutable historical **input**, not a newly qualified result, new
benchmark campaign, or authority to promote additional defaults. It preserves
the existing tuned/portable decisions while the checker evaluates future
semantic changes against source-matched cold/warm/moved campaigns. Formatting
of the current manifest is not compared byte-for-byte with this input.

The preceding P1b migration compared generated CUDA, registry and derivative
outputs against `8d2ad48175b79f2a3cb5749d3552413cdf5b7d4e`: stock auto bundles
for SM80/SM90/SM120 and forced-portable SM120 were byte-identical. That was a
one-off migration observation, not a permanent generated-source fingerprint.
No GPU numerical or timing result is asserted by this retained manifest.

Existing scientific and performance records keep their original source and
device identity. In particular, retained provisional f-shell selections are
not certified merely because they are present in this manifest.
