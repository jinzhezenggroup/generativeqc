# Clean CPU correlated derivatives: numerical evidence only

One physical H2 DZ/TZ Standard CBS complete E/F publication on clean source
`f8c78a3700cb8eca6b49128842546990613d9c7d`: 41.736168964765966 seconds including
construction and owner teardown, excluding independent oracle validation.
All five components and the composite pass 1e-9 energy / 2e-6 complete-force
gates. **Not ABBA/BAAB, a 1.05 performance qualification, or #2227 completion.**
The earlier `../required-native-policy-20261104/` campaign remains incomplete.

Run `python replay.py` offline: authenticate storage, recompute all twelve E/F
gates and coefficient reconstruction, and refuse broader promotion. No native
library, compilation, PySCF or GPU executes. `samples.json.xz` retains both
journal records, environment/build provenance, the exact original probe and an
audited diagnostic summary. The full trace stays ignored; scoped counters do
not certify whole-endpoint work or allocation peaks, which remain unknown.

To repeat, extract `probe_source` to a new file in the measured checkout. Use
`GENERATIVEQC_LIBRARY` for the source-verified Release CPU/AOT library,
`PYTHONPATH=python:.`, one OpenMP/OpenBLAS thread, a new native trace path, no
interposer or source/cache/tile override, and
`timeout 300s python /tmp/probe.py /tmp/new-science.jsonl`.
Verify the compiler cache before building; never overwrite retained receipts.
