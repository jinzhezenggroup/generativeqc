# Lossless transport of historical JSON evidence

`index.json` records the original size and SHA-256, payload identity and storage
format of each compacted document. The 1,192,712 original bytes occupy 217,636
bytes as existing XZ storage descriptors/payloads and gzip streams. The index
itself is additional provenance, not part of that payload-size comparison.

Every decoded byte remains available offline. No numerical value, rejected
population, scientific input or frozen receiver changes. These are storage
transports, not a new numerical codec or scientific requalification.

Use `tools.restore_retained_evidence.read_stored_json_bytes` for an explicit
`generativeqc.compressed-manifest.v1` descriptor. Ordinary JSON returns its
original bytes. The common `tools.generativeqc_validation.record.load_json`
reader also authenticates this local transport before returning the original
document. Historical automatic-force verifiers and their bound documentation
remain byte for byte unchanged. The reader authenticates both stored and decoded size/digest,
rejects linked payloads and bounds decompression. H100 delivery wrappers decode
before checking the historical attachment digest or invoking a frozen receiver.
The post418 frozen receiver already supports the retained `.json.gz` fallback;
its source and numerical gates are unchanged.

Run `pytest -q tests/python/test_lossless_json_storage.py` to authenticate every
original byte and exercise collection-member failure controls. This check does
not need Git history, a compiler, a GPU or a network fetch.

`source-snapshot.json` separately binds four build/source-only provenance files
(507,403 original bytes) to an already published Git ancestor. Restore explicitly
with `tools/restore_retained_evidence.py --all --manifest` and a new output
directory. This provenance inventory contains no numerical observation packet.
