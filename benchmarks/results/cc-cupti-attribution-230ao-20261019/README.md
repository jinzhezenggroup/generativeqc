# 230-AO CC attribution

Measured native: `7045237d47aa56a7938da3776f65b86f4bd74be4`, H200/595.58.03.
Recovery is not measurement provenance. Two clean children plus one
diagnostic passed FP64 E/(T)/R1/R2, bits/work gates.

`campaign.json.gz` is a derived whitelist/path-normalized capsule, not original
receipts. Original SHA/size retained; private SDK/access omitted.
`oracle.json.gz` preserves the public8580-byte independent reference.

## CPU: Linux/Python3.11+

```bash
B=benchmarks/results/cc-cupti-attribution-230ao-20261019
python -m benchmarks.cc_cupti_capsule --capsule "$B/campaign.json.gz" \
  --oracle "$B/oracle.json.gz" --output .artifacts/cc-cupti/summary.json
```

Summary checks scalar gates/bits/work; raw stays UNAVAILABLE. Add
`--campaign ORIGINAL_DIR` for raw audit and full capsule binding.
Raw/DSOs are not bundled; author-retained originals may expire. Missing originals
leave full raw verification UNAVAILABLE; no external archive/Release is used.

Add `--source-revision PUBLICATION_MERGE_SHA --tooling-output NEW_DIRECTORY`
(replace placeholders) using a local clone of the delivered publication tree.
16 source companions are bound by path/blob/SHA/size; reverse patch restores
two tools and worker to17 original byte identities. No source execution/downloads;
transports/prompts blocked, also old Git. Missing/wrong objects, patches or
non-new output refuse.

Full reproduction requires raw AND source recovery. No new native/science
execution or upstream science/performance qualification.

Global fine attribution INCOMPLETE; implicit waits/causality/profit UNAVAILABLE.
Per-call outer dimensions unrecorded. Original1800s native bounds lacked a
prelaunch outer-worker guard; late empty guard did not cover execution.
No264-AO/force/response/double-buffer/default/parent completion. Older129d/Nsight
and preparation failures remain unchanged. Details: [decision note](../../../.agents/notes/implemented/performance/2026-10-19-cc-phase-source-observation.md).
