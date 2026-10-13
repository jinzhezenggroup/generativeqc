# Retained one-electron value endpoint records

The historical sp8/batch-3 [thread](sp8-b3-thread.json.gz) and
[shell-warp](sp8-b3-shell_warp.json.gz) records use lossless deterministic gzip.
[storage.json](storage.json) binds stored and decoded hashes and byte counts to
commit `da3aa79cedf4dea987c677bf4a5cd2daf9255982`. Every original sample,
comparison status, numerical array, timing, source/library identity and scheduler
record is preserved byte-for-byte; this storage change neither reruns nor
requalifies the historical measurements.

Read with `tools.generativeqc_validation.record.load_json` or
`gzip.open(path, "rt")` plus `json.load`. The plain originals remain recoverable
from the named Git commit. Other records in this directory are unchanged.
