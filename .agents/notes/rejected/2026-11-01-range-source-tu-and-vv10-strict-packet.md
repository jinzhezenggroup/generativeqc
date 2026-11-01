# Decision: reject source-TU and strict-packet specialization without endpoint benefit

Status: rejected
Date: 2026-11-01

## Alternatives and evidence

A separate scalar range-source CUDA translation unit removed unused derivative
branches from its entry point. Final device linking retained large call-graph
stack requirements; unlinked resource reports did not describe executed code.
The original generic J/K kernel source and its linked resources were unchanged.
Source identity `63fe20eefc437ad8c31be17a1fd6b0e97793ba2849474ec304a4ffc5f9a6d866`
and library `bcfbcc9458afbe3e87338b6f55ecf31388f7bbefdcec635dfec9ccefa206bf65`
bind this withdrawn candidate.

Complete WB97M-V/96-atom energy-and-force ABBA populations, with one conditioning
owner and cold plus three warm calls per measured owner, were effectively tied:

| GPU | Cold change | Cold + three warm change |
| --- | ---: | ---: |
| RTX 5090 | -0.101355% | -0.077754% |
| RTX PRO 6000 | -0.005975% | -0.003459% |

Negative percentages mean less candidate time. These differences do not support
promotion. Ten populations and 36 complete energy/force calls passed independent
README reference gates. A separate 108-atom, 32-GiB six-phase lifecycle took
904.3600114563 s for the parent and 904.2541382627 s for the candidate; all twelve
paired outputs and semantic-work checks passed. Fixed-input disk probes also
showed no reproducible native regression behind the initial pytest wall-time
difference. The missing disk-fixture precondition was fixed independently.

A separate non-RDC source-only compilation reduced some resource counts but was
never linked or benchmarked as a production endpoint. Those counts are not a
speedup result.

Specializing the VV10 packet template to remove mixed-arithmetic branches from
strict FP64 also failed to help: the 80k-point complete pair consumer regressed
by 0.4614% for features and 0.6610% for geometry. Its 206 small GPU admission and
independent-oracle tests passed. At 320k points, the experiment's parent-versus-
parent numerical gate failed before the candidate ran. That failed population
remains failed; neither its gate nor its producer identity was rewritten.

## Decision and retained evidence

Neither production change is submitted. Local ignored frozen sources, binaries,
raw results, failures, scripts and patches remain under
`.artifacts/range-source-specialization-20261101/`,
`.artifacts/range-source-whole-20261101/` and
`.artifacts/vv10-strict-packet-20261101/` in the qualification worktree.
The source experiment's complete archive compresses 226 text records and 33,464
FP64 values to 90,841 bytes, exceeding the remaining tracked evidence budget;
it was not added to `benchmarks/results/` or published externally.

Parent endpoint profiling instead found VV10 feature and geometry kernels using
48.3% of accumulated GPU time. The native stationary force owner omitted the
standalone owner's exact seed reuse, making removal of repeated pair sweeps a
better next hypothesis. Profiling is diagnostic: its source-revision argument
was incorrect, while the retained identity records correctly bind actual source
`6be420fb4ee14175922194d61b41e3a20f13665968bc012b474b21cada96f4cc` and library
`bb2c3e70046ebc753dbb9722b53df1eb9094bb6d3bb8882ff3ae14887f471e10`.

Revisit specialization only with final linked resource evidence and a measured
complete-endpoint benefit. Keep failed qualification and slower samples; do not
substitute isolated register counts or synthetic throughput for that evidence.
