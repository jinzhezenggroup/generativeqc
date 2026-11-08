# Decision: rotate the accepted derivative AD catalog baseline

Status: implemented
Date: 2026-11-08
Agent: /root/audit_3044
Model: GPT-6 Astra Pro

## Reason and decision

Integrating conventional HF/correlated changes from
`d689716b5a444e4bfef00fd1fbc967a596fd650f` into the published derivative AD parent
`373f67df3183b3e9fd32fae577019e5e4082b066` changes the two current checked source
identities. The computed capability summary and retained evidence are exactly
equal to that parent's snapshot. There is no new identity transition, stage
demotion, promotion or executed qualification, and retained evidence stays stale.

The catalog writer requires the exact immediately retained summary as `previous`
before any report update. Keeping the older pre-AD baseline would violate that
guard even though this integration changes no capability. Follow the existing
[catalog rotation procedure](../architecture/2026-10-28-native-libxc-evidence-catalog.md):
retain the published parent's complete summary as `previous`, update its binding
and source commit, then run the unchanged official writer and checker.

The 222 earlier identity acknowledgements belong to the already accepted AD
transition. Clear them from the current manifest because the new immediate
transition is empty; the transition validator rejects stale tokens. The
snapshot's `changes` field and rendered revision diff therefore become empty.
This does not undo or conceal the earlier transition. Its exact records remain
in the published parent above:

| Original path under `manifests/libxc/` | Git blob at the published parent |
| --- | --- |
| `native-qualification.json` | `0ef8e5122b270ef602dc68164cea0d378905d596` |
| `native-qualification-previous.json.xz` | `8ad2b5bfbfa95455de2cdaf964d3d6d6202fc43f` |
| `native-qualification-snapshot.json.xz` | `cd3d351bda95a1bb0f8ef30d9cccbd7712ab8ca2` |

## Preserved contracts

All AD functional identities, evidence subject identities, qualification/status
sets, measured source and binary identities, campaign bytes and historical
bindings remain unchanged. The only snapshot changes are its two current source
identities and its baseline-relative transition list. No measurement is
restamped and no guard, threshold or production code is changed by this catalog
maintenance. Removing the writer guard or copying the incoming pre-AD catalog
would lose the intended transition accounting and is rejected.

## Validation

Use `tools/check_libxc_native_catalog.py --write` followed by the same command
without `--write`. The existing native-catalog and capability-transition tests
cover retained-evidence integrity and rejection of stale/missing tokens. The
indexed evidence change check uses the incoming source commit as its prospective
baseline; this catalog rotation adds no benchmark result bytes.

The official writer and checker pass, as do all 75 existing native-catalog and
capability-transition tests. The indexed evidence check passes with 1,915,731
review bytes against its unchanged 2 MiB limit. No native or GPU execution was
performed for this metadata reconciliation.
