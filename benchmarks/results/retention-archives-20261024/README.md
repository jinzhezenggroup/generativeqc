# Historical benchmark ZIP checkout trim — 2026-10-24

Seven historical raw run archives (**1,090,500 bytes**) are
removed from the working checkout, not Git history. Their full original bytes
remain in ancestor commit `f5d8df5a0d2c5ba4602e8aaaa5eefe94147a28fc`; the
[snapshot manifest](snapshot.manifest.json) records every path, byte count,
SHA-256 and Git blob identity. Reviewed scientific summaries and original
`raw-evidence.manifest.json` member checksums remain in each family.

This is storage maintenance, not numerical requalification. No active
scientific fixture, benchmark publication, production source or acceptance
tolerance is changed. Full clones retain their existing Git objects; this
does not reduce historical .git storage or rewrite history.

## Restore a historical archive when needed

```bash
python tools/restore_retained_evidence.py \
  benchmarks/results/ecp-consumers-171/raw-evidence.zip \
  --manifest benchmarks/results/retention-archives-20261024/snapshot.manifest.json \
  --output .artifacts/retention-archives-20261024/ecp-consumers-171/raw-evidence.zip

python -m tools.unpack_evidence benchmarks/results/ecp-consumers-171 \
  --archive .artifacts/retention-archives-20261024/ecp-consumers-171/raw-evidence.zip \
  --output .artifacts/retention-archives-20261024/ecp-consumers-171/unpacked
```

Substitute another snapshot entry/path to recover another archive. The restore
script verifies stored size and SHA-256 and writes to ignored `.artifacts/`
without fetching; `unpack_evidence` independently checks every ZIP member.
An ordinary complete Git clone containing the ancestor restores offline;
a shallow clone/source archive may require an explicit fetch of the recorded
commit first. All family README restore recipes are updated to use `--archive`.

Agent: ChatGPT
Model: GPT-6
