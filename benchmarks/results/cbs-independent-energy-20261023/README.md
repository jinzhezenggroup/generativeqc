# Independent full-DF CBS energies

Refs #2679. Six QM7 molecules (indices 0, 8, 48, 183, 293, 5604), both
Standard/Fast: all 72 constituent/composite comparisons PASS at 5e-8 Eh;
maximum error 2.37639596889494e-9 Eh. ENERGY ONLY: forces/directional/translation
and parent #2679 remain open. No current-master device or timing qualification.

`canonical.json` lists the modes, sizes, SHA-256 and storage bindings for all
29 original files (211606 bytes). The JSON/XZ container is self-contained in this
commit; the source commit is informational, never a recovery dependency.
The original publication, raw logs/outcomes, C1 diagnostic failure, measured
wrapper, later retry fix and its 1/9-negative versus 9/9-positive fixture reports
are restored byte-for-byte. Original sources/tests and full protocol/resource
documentation are available after safe unpacking. No tar extraction or eval is
used; only regular 100644 files in a new directory are allowed.

From the repository, with this directory as `$CAPSULE`:

```sh
python "$CAPSULE/replay.py" --check
python "$CAPSULE/replay.py" --unpack NEW_DIRECTORY
python NEW_DIRECTORY/storage-test.py "$CAPSULE/replay.py"
python "$CAPSULE/replay.py" --require-pass --output NEW_COMPARISON.json
```

The first two commands do not execute restored sources. Strict replay validates
both stored and original manifests, runs the unchanged original replay, and
compares its full output. It requires existing upstream history at `4a783a27e`,
as before. Existing destinations/output files are refused. The restored tool
rebuilds identical JSON/XZ bytes with `storage-test.py ENTRY --repack ROOT NEW.xz`.
