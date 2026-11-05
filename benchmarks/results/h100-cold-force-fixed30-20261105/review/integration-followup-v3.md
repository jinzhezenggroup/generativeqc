# Delta integration follow-up — #3009 and #2985

**PASS: no blocking integration concern.** Original WARN and v1/v2 limits remain unchanged.

Head `1cbf5adaa58ca2db98cd71940e8f7a7d7c83817c` normally merges published `be7badb6785b331042b2bd2b3acc1be11f85d938` and upstream `415d8caba12a8f7ecb4e81e48d5dae913720e695`.

- **3009 is CI-only:** its11 files are workflow/action/shard/cache tooling, CI tests and their note; no production runtime/native input changes.
- **2985 runtime scope:** the sole executable change removes the restricted density-shape check from `CompositeRKSResponse.from_native`'s CPU-first full-range candidate predicate (lines721–727). AST comparison confirms other execution, native-state/Fock provenance, resources, CUDA branch and cleanup are unchanged.
- **No first-E/F effect:** Calculator reaches this explicit second-order binder inside `_run_public_rks_second_order` (lines2870–2920), not ordinary cold/warm E/F. CUDA still fails the CPU-only candidate arm; public full-range spin admission remains unpolarized. Low-level UKS extension does not promote public UKS Hessians or H100 acceptance.
- **Preservation:** native src/API/include/CMake/compiler scope and all9 declared E/F/default/timer inputs are unchanged. Exactly4 frozen scientific payload members match be7. All6 original/v1/v2 reports match saved hashes and publishedbe7 review bytes. No member expansion or numeric/device/test replay ran.
- **Libxc:** only checked source/native identities differ against both actual parents. All scientific/capability/execution/evidence/stale fields remain equal; `stale:true` persists. Prose changes only those identities.

Head1cbf5adaa remains unmeasured. Original720/72 cohorts retain their frozen source/device boundaries. This supplement does not qualify upstream CPU UKS/Hessian science; normal CI covers merged interfaces. Same-family/provisional assurance; actual backend identity unknown.

`integration-followup-v3.json` records22 relevant input hashes and exact scope evidence. No blocker remains. Original reports and production/public files were not altered.
