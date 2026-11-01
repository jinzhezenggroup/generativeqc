# Direct-hybrid metadata catalog identity refresh

The non-VV10 direct-hybrid metadata expansion in PR #2927 changes the
compiler source identity through `xc/spec.py`. The native catalog gate therefore
reports 221 exact identity transitions, even though this change does not qualify
any new runtime XC implementation.

Preserve the exact previously retained snapshot summary as the transition
baseline, acknowledge only the computed identity-change tokens, and regenerate
the snapshot and reference using `tools/check_libxc_native_catalog.py`.
Do not suppress the identity check or relabel historical campaign measurements.

The refresh verifies that every functional's non-identity capability fields,
qualified/ready/blocked stages, endpoint counts, retained endpoint records,
observed contexts, structural count and unmeasured inventory are unchanged.
Retained evidence is stale before and after the refresh. No scientific threshold,
native execution qualification, or public-method admission is changed.

The merge reconciliation uses the exact retained master snapshot at commit
`c2d776e75818433bf7a702341a1c9b4c48983ada` as its immediately preceding baseline.
Its exact summary is retained in
`manifests/libxc/native-qualification-previous.json.xz`. The combined source
changes only the same 221 identities; no intervening qualification transition
is hidden. Both generated conflicts are regenerated from the combined source.
