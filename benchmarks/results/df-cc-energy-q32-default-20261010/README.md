# Automatic energy-only Q32 cap: lossless Git recovery

All six original files (19,127 bytes, including every accepted ABBA/pilot sample)
remain in merged commit `9f67e7806e3151454530baf0ee66ae8808d826f0` (#2197).
`snapshot.manifest.json` pins each byte count and SHA-256. Restore offline with
`python tools/restore_retained_evidence.py --manifest benchmarks/results/df-cc-energy-q32-default-20261010/snapshot.manifest.json --all --output .artifacts/q32-recovery`.
The restored original README/recipes describe the frozen-source reconstruction.
This frees checkout headroom for the successor workspace evidence without
raising the aggregate cap, losing samples, or publishing an external archive.
