# Fresh H100 cold-force experiment and scoped code audit

**Verdict: WARN; no blocking defect found in the audited scope.** Review complete; scope complete. Requested gpt-6.1-sol/ultra, fresh fork; actual returned backend identity unknown. Assurance is **same-family/provisional**.

Code checkout head: `8e31d0b19ccb4795229c05444267438e393f0022`. Run paths below are relative to `h100-cold-20261105`; `code` paths are relative to its sibling code checkout. The report uses read-only primary inspection, byte/hash checks and independent standard-library arithmetic. No GPU, compiler, solver, profiler, fetch, scientific rerun, production edit or publication was performed.

## A–F checks

### A — WARN: Independent-reference provenance and scientific identity

All40 retained dense/CPU-spot references satisfy strict declared controls, exact geometry, matching CPU quadrature and quarter-gate stability/spot checks. All18 original frozen input tables and the two original WB DF exclusions remain exact. Native execution is independently calculated from its own Hückel/cold density; no oracle density is passed. Cross-revision/environment reuse binds every exact reference receipt to basis, spherical representation, charge/spin, driver hash and accepted package maps. Two inherited refined-oracle producer executable byte snapshots remain unavailable in the supplied/ boundedly searched retained sources. Oracle source02337e is scientific context and does not authenticate those executable bytes. This bounds reproducibility of old reference generation; current full-vector gates are recomputed from the accepted original receipts.

Evidence: `received-fixed30-v1/oracles.json.xz:all40 records.samples.controls/geometry_bohr`; `received-fixed30-v1/reference-reuse.json:entries`; `received-fixed30-v1/current64de-final-inputs-frozen-v1.json:groups`; `received-fixed30-v1/fixed30-protocol-v1.json:excluded_rows`; `received-fixed30-v1/endpoint-fixed30-v1.py:34-38,118-123,180-184`; `code benchmarks/readme_defaults.py:195-224,299-327,379-383`; `code benchmarks/results/h100-cold-force-fixed30-20261105/README.md:144-162`; `oracle-refined-source-lookup-v1.json:target_hashes/matches`; `review oracle-controls-independent-v1.json; oracle-producer-git-independent-v1.json; reference-source-continuity-independent-v1.json`.

### B — PASS: Full-vector arithmetic and process-paired timing

Independent standard-library arithmetic verifies every720 frozen30 and72 later-WB2 E/F output, all closed-study independent/stronger matched gates, every three-process pair ratio, full denominators30/Direct16/DF14, strict median>1.05 and every-pair>1 rules, and complete-set geometric means. No failed/unfavorable case or successful first-call outlier is discarded. Current-WB2 every matched warm-call index passes1.05. Both WB2 cases meet the2% cold target; earlier rejected companion pilots remain rejected, including the anchor pilot with only1.7438% reduction. The combined work performance:false flag remains unchanged while the independently preregistered force study passes.

Evidence: `received-fixed30-v1/fixed30-v1/*.json:samples.energy/forces/seconds/phase/repeat`; `received-fixed30-v1/fixed30-protocol-v1.json:order/numerical_gates/win_rule`; `received-main-integration-v2/main-integration-v2/*.json:samples and observed_execution_controls`; `received-main-integration-v2/main-integration-preparation-v2.json:matched_gates/warm_each_call_ratio_limit/cold_median_reduction_target`; `work-analysis-v1.json:performance_pass=false; work-candidate-decision-v1.json:studies`; `review independent-reduction-v1.json; closed-study-provenance-and-matched-v1.json; independent-wb2-v1.json`.

### C — WARN: Execution, source/build/image and platform authentication

All frozen180 populations/720 E/F are one physical H100 GPU-a21ed680 with measured source db0788b/native0ee and mapped versioned core6b48. Later18/72 are one different H100 GPU-a04c1eb0 with measured4ee/nativeed33 and mapped cored164. All supplied1398 public member bytes and1344 raw transport members across seven capsules authenticate. Actual mappings observe only libgenerativeqc.so.0.1.0 in both new cohorts; the25-artifact build/AOT inventory is separate. Observed76 frozen30 and69 WB2 distinct reference-site images match the old installation catalog. Old runtime/wheel catalog scope does not become a fresh all-library/before-import witness. V5 timestamp skip and V6 wrapper false negative authenticate; both platform jobs remain FAILED, and V7 is CPU-only inspection. The original WBv1 cancellation was quota-blocked/unstarted with zero E/F.

Evidence: `received-fixed30-v1/fixed30-v1/campaign.json:device_uuid/source/build_artifacts/runs`; `received-main-integration-v2/main-integration-v2/campaign.json:device_uuid/source/runs`; `reference-installation-history-provenance-v1.json:scope`; `received-main-integration-v2/base/main-source-mtime-correction-v6.json:failed_receipt_sha256/source_sha256_before/source_sha256_after/source_mtime_before`; `received-main-integration-v2/base/build-receipts-v6/build.stdout:3/7 and4/7 CXX compiles`; `received-main-integration-v2/base/build-metadata/build_identity.hpp:5`; `received-main-integration-v2/base/build-metadata/c_api_tuning.cpp:2,16`; `main-build-platform-terminal-v5.json:data.status; main-build-platform-terminal-v6.json:data.status`; `main-integration-platform-terminal-v2.json:data.status=SUCCEEDED`; `received-main-integration-v2/main-integration-unstarted-v1-events.json:data.items FailedScheduling/StopJob`; `review transport-independent-v1.json; public-and-build-independent-v1.json`.

### D — PASS: Live selector, defaults, bounded ownership and work meaning

The live fused strict RKS capability now admits automatic Direct nonlocal force batching independently of fitted SCF admission. Explicit force requests retain the previous branch and explicit SCF inheritance; Direct SCF remains256. All-electron/through-d/48-128 atom/512AO/524288point/CC>=8/free1GiB/stationary/resource/strict-budget gates remain. Force fallback drains/resets the controller and last token once, retries at256 with the same authenticated density token, rechecks leases/allowances and propagates non-allocation faults. Resolved capacity queries use the same stationary_tile_points consumed by the actual controller. Native408 new E/F samples are Standard strict64 with one candidate solve. Selector cold work is18 iterations/19 Focks in both arms; warm1/1. In later-WB2, WB48 off/default work is17iterations/18 versus17Focks, WB96 is18iterations/19 versus18Focks, in every round; every native warm remains1/1. This supports the observed saved cold Fock without shifting it into first warm. Equal SCF counts and invalid quartet-work flags do not provide a semantic force/AO/quartet census; broad kernel-level mechanism confirmation remains unavailable.

Evidence: `code src/methods/dft_method.cpp:651-715,2201-2229,2277-2296,2472-2485,2496-2523`; `code src/dft/ks_tile_policy.hpp:15-48`; `code docs/user/ks_options.md:213-239`; `code src/dft/cuda_ks.cpp:3136-3162,3177-3192`; `received-work-v1/work-study-v1/*force-tile*.json:samples.settings/fock_builds/iterations`; `received-main-integration-v2/main-integration-v2/*.json:samples.settings/fock_builds/iterations`; `received-force-host-v1/corrected/pytest.stdout:8passed; zero-test-failure/pytest.stdout:0items`; `review precision-observation-independent-v1.json; independent-wb2-v1.json; source-code-git-independent-v1.json`.

### E — WARN: Source-qualified claims, storage migration and public delivery

Public source/device separation is correct: selectorD60, frozen0ee original30, later4ee/ed33 affected-WB2, and current8e31 are distinct. The force-admission source section is byte-identical across frozen0ee, later4ee and8e31; the later head includes upstream Hessian/snapshot/B3LYP coefficient plumbing and remains unmeasured by either full matrix. Reuse of unchanged-source evidence is warranted within scope; neither new cohort becomes a latest-head30 aggregate. All five frozen cold nonwins remain explicit. Historical DB36 storage recovers only the two declared missing owned parts from already-merged40f126, with exact hashes/blobs; present corruption/symlink/directory/other missing members fail and no implicit fetch is added. Synthetic normal tests avoid historical science. Exactly the two retired exception entries were removed; all policy caps/other exceptions are unchanged. Libxc decompressed snapshot comparison changes only checked source/native IDs, preserving stale:true and all capability/scientific content. Prepared public bundle/readback is verified; its review attachment/status update follows this completed report and normal repository gates.

Evidence: `code benchmarks/results/h100-cold-force-fixed30-20261105/README.md:3-19,73-92,123-133,156-162,191-230`; `upstream-hessian-integration-scope-v1.json:head/measured_frozen30/measured_later_WB2/scope`; `review source-code-git-independent-v1.json:force_admission_slices`; `code benchmarks/h100_db36_final_cohort.py:16-49,69-78`; `code tools/restore_retained_evidence.py:147-178,182-213`; `code tests/python/test_h100_db36_final_cohort.py:12-38,74-119`; `code benchmarks/results/h100-sm90-db36-final-20261103/historical-snapshot.json:files`; `review migration-metadata-independent-v1.json; evidence-policy-independent-v1.json`; `libxc-force-metadata-refresh-v1.json; libxc-main-integration-refresh-v1.json; libxc-hessian-integration-refresh-v1.json`; `code docs/reference/libxc_qualification.md:3-9; manifests/libxc/native-qualification-snapshot.json.xz:stale/summary`.

### F — PASS: Appropriate evaluation classification

This is source-bound computational numerical-reference qualification plus real-device endpoint performance measurement on static original-recipe water48/96 inputs. The numerical references include independent dense GPU calculations and matched CPU spots under explicit reuse contracts. They support scoped numerical agreement and the stated H100 timing regimes; they are not experimental chemical ground truth, a general molecular performance certificate, A100/5090/UKS/CBS/displaced-geometry acceptance, or cross-family scientific review. Requested reviewer route is gpt-6.1-sol/ultra/fork none; actual backend-reported identity is unknown. Assurance remains fresh same-family/provisional.

Evidence: `received-fixed30-v1/fixed30-protocol-v1.json:scope`; `received-main-integration-v2/main-integration-preparation-v2.json:scope`; `code benchmarks/results/h100-cold-force-fixed30-20261105/README.md:135-162,201-230`; `cold-force-final-audit-dispatch-v1.json:requested_model/requested_effort/actual_reported_identity`; `review audit-request.json; audit-trace.json`.

## Independent frozen30 reduction

Source `db0788b2032ba99029d1e7a7a995380777d4981f`, native `0ee40c0ab15addfe978f93ae293c6ea5fd0cacec4147161b283ee5159134a73e`, H100 `GPU-a21ed680-6184-b931-377d-81ff7cf9f52e`. All180 fresh processes/720 E/F pass the accepted original dense/CPU-spot reference gates. Cold and warm are separate complete endpoints after CUDA-context initialization; retained compiled caches, installation, process/import startup and AOT compilation are outside the timer.

| Phase | Population | Wins | Complete-set geometric mean reference/native |
| --- | --- | ---: | ---: |
| cold | all (30) | 25 | 1.64609958264485 |
| cold | direct (16) | 12 | 1.4942833813618 |
| cold | df (14) | 13 | 1.83858007210117 |
| warm | all (30) | 26 | 1.43830276053956 |
| warm | direct (16) | 13 | 1.4698184633483 |
| warm | df (14) | 13 | 1.40311142274808 |

All five cold nonwins remain in the denominator:

| Input | All three paired reference/native cold ratios | Median |
| --- | --- | ---: |
| hf-water-96-direct | 0.625485248521544, 0.603433460327104, 2.48641667687177 | 0.625485248521544 |
| hf-water-96-df | 0.847919246384766, 1.03817678834183, 3.90373567211559 | 1.03817678834183 |
| r2scan-3c-water-96-direct | 1.05015418075223, 1.03729059971265, 1.03897780313542 | 1.03897780313542 |
| wb97m-v-water-48-direct | 0.677771772346475, 0.680295301636272, 0.697102051434557 | 0.680295301636272 |
| wb97m-v-water-96-direct | 0.526747032330503, 0.512045759400546, 0.517069393928523 | 0.517069393928523 |

Worst frozen full-vector errors: energy/atom `8.204134796111854e-06` Eh, maximum force `5.6885556734442844e-05` Eh/Bohr, RMS force `2.5819326394557067e-05` Eh/Bohr. Original thresholds remain1e-5/1e-4/3e-5. The large successful HF pair ratios remain visible; none was trimmed.

## Selector and later affected-source integration

The D60 force selector independently gives2.4086156013% median cold reduction and11.7764627048% warm reduction, while SCF remains18iterations/19Focks cold and1/1warm. Its three candidate/OFF cold ratios are0.9759138439872199,0.9799960226572006,0.9754743745013382. The worst individual warm-call ratio is0.8904828718474376. Original stronger matched and independent gates pass. The residency companion remains rejected and the combined work performance flag staysfalse.

Later WB2 uses measured `4ee811b567938df2d8200389b4918481297511cb` / native `ed33e0ca6d2f06ec039545283699be2c668245021dfae13f11c7ce7ae2356d05`, core `d164314b4ad97818c906b96648fe6f22a2b23cd29f85ad29d43713688b39f58e`, H100 `GPU-a04c1eb0-72d5-2f4a-2ea4-d8c6e3e1db59`; all18 fresh processes/72 E/F pass. OFF is eager-anchor0 with automatic force in both native arms. Default control is actually observed unset.

| Case | Cold default/OFF ratios | Median cold reduction | Worst warm-call ratio | Actual cold iterations/Focks OFF → default |
| --- | --- | ---: | ---: | --- |
| wb48-direct-current-main-anchor | 0.895299710202, 0.913221187767, 0.910588172535 | 8.941182746% | 1.00224403963 | 17/18 → 17/17 in all3 rounds |
| wb96-direct-current-main-anchor | 0.896536052827, 0.894965874448, 0.890428860567 | 10.503412555% | 1.00727131007 | 18/19 → 18/18 in all3 rounds |

Both cases pass the predeclared2% cold target and every-index1.05 warm guard. All native warm calls remain one iteration/one Fock, including the first. WB48 resolves force256 and WB96force1024; both SCF256, no explicit tile request. The corresponding median reference/default cold ratios are0.8309959200592919 and0.6040455378954146; both remain cold nonwins against the reference. These timings are not pooled into the frozen30 or multiplied by selector gains.

## Readiness and claim boundaries

- **code: READY_WITH_SCOPED_REVIEW** — No blocking defect found in force admission/default integration, bounded fallback, historical byte recovery or metadata changes. All later-head native behavior is not certified by this narrow code inspection.
- **device: PASS_MEASURED_SOURCE_COHORTS** — Frozen0ee180/720 and later4ee18/72 are actual H100/SM90 source/image-bound completed cohorts; other architectures remain outside acceptance.
- **scientific: PASS_UNDER_ACCEPTED_RETAINED_REFERENCE_CONTRACT_WITH_PROVENANCE_LIMIT** — All observed full-vector gates/stability/CPU-spots pass. Missing exact refined historical producer bytes bound regeneration provenance. No experimental ground-truth assertion.
- **default_integration: PASS_FROZEN0EE_AND_AFFECTED4EE_WB2** — Observed automatic force1024/SCF256 on WB96; laterWB48force256/WB96force1024 both observed with0/unset anchor control; strict64/scientific settings preserved. Later8e31 untouched force policy reuses scoped evidence without becoming a measured latest-head matrix.
- **performance: SUPPORTED_SCOPED_GAINS_WITH_REMAINING_GAPS** — Selector2.4086% cold gain and both laterWB2>=2% targets pass. Frozen25/30cold wins preserve5 nonwins and incomplete universal cold objective. No cross-device pooling.
- **public_delivery: PREPARED_BUNDLE_AND_READBACK_VERIFIED_REVIEW_ATTACHMENT_PENDING** — All1398 original members/1060blobs verify, public summary matches independent reductions, and supplied complete CPU readback passes. Public cohort still labels this fresh review pending at audited snapshot; append this actual report/request/trace/hashes and retain applicable repository checks before final delivery. No external publication performed.

Inherited exact refined-oracle executable snapshots remain unverified; the public README now discloses that source-context commits do not authenticate those producer bytes. Original observable/stability/CPU-spot gates were recomputed. The runtime witness observes the versioned native core only; the25-artifact build/AOT audit is separate. Current8e31 is unmeasured, while its force-admission section is unchanged. Other devices, broader chemistry, complete semantic AO/force work and cross-family review remain outside acceptance.

Both failed build Jobs remainFAILED. Authenticated V5 changed-source mtime0 caused the compiler skip, caught by source identity. V6 actually compiled cuda_ks.cpp and c_api_tuning.cpp and passed native/AOT checks before its filename assertion failed. V7 inspects existing header/owner/build bytes on CPU. The earlier v1 quota cancellation was unstarted, with zero E/F.

Historical DB36 parts match merged40f126 byte-for-byte and retain fail-closed recovery. Exactly two owned stale exception entries were removed; all storage limits and other exceptions remain equal. Libxc snapshot changes only checked source/native IDs; stale:true and capability/scientific contents remain equal.

The prepared public container has1398 exact original members/1060 blobs and1373032 stored bytes. Its existing complete CPU readback equals the public summary and these independent reductions. Public review metadata was stillpending at this audit snapshot; attach this actual completed review and rerun applicable checks on the added review artifacts. The supplied local pre-commit log passed; this reviewer did not rerun compilers or tests.

Full per-claim decisions, exact raw-field evidence and all ratios are in `audit-report.json` and the independent reduction files. Final audited input inventory: `audited-input-hashes.json` (1525 files), SHA256 `8d912df6cdbd48cc162a3261c1c00a2d805a0ed462059fb094f1b7bac9eea85f`. Request/trace: `audit-request.{txt,json}` and `audit-trace.json`.
