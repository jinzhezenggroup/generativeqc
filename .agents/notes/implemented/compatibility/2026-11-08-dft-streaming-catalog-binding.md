# Decision: refresh stale Libxc catalog bindings for independent DFT streaming

Status: implemented; metadata reconciliation, not native requalification
Date: 2026-11-08

## Problem

PR #3051 independently adds the direct shared-Rys streaming owner on master
`2bf3fd98da9c6cfba46ca83345e63e1d934e4780`. Hosted core-a stops in the existing
native Libxc catalog check because its two checked input identities changed.
The dependent native CLI package job then stops on that failed prerequisite.
Neither failure is an executed numerical test or evidence of a GPU regression.

## Decision

Follow the existing catalog rotation procedure rather than disabling the check,
importing the old HF branch's snapshots, or stamping old measurements as current.
Compare the actual live report to the retained master report before writing it:
only `source_identity` and `native_source_identity` differ. The complete
capability summary is equal; every identity/stage transition and regression list
is empty; both reports mark the historical native campaign stale.

Bind `previous` explicitly to master 2bf3fd98d. Its exact accepted summary is
already byte-identical to the existing previous-summary record, so no replacement
or recompression is necessary. Keep its verified checksum, empty acknowledgements,
campaign bytes, measured source/ELF and historical bindings unchanged. Regenerate
the snapshot and rendered report with the unchanged official writer and verify
them with the unchanged checker. No qualification claim is promoted or demoted.

## Invariants

- Never restamp an old source, binary or endpoint as a new measurement.
- Keep scientific, transition and stale/missing evidence guards enabled.
- This metadata update does not change either frozen normal-build source identity
  or qualify the ongoing A100/5090 endpoint experiment.
- Do not enlarge the benchmark evidence budget or publish release assets.

## Validation

The preflight compares all live report fields and requires exact equality outside
the two checked source identities. The official writer and checker pass, as do all
75 existing native-catalog and capability-transition tests. The complete native
source identity remains exactly the frozen candidate identity 75d94b32. Benchmark
retention and the full-PR change check pass, with 425086 of the unchanged 2097152
review-byte limit. GPU execution is neither needed nor claimed for this
metadata-only repair.

## References

- `tools/check_libxc_native_catalog.py` and
  `manifests/libxc/native-qualification.json`.
- `2026-11-08-derivative-ad-catalog-baseline-rotation.md` and
  `../architecture/2026-10-28-native-libxc-evidence-catalog.md`.
- Issue #2976; PR #3051, initial hosted run 42493215176.
