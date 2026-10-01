# Source-work audit: first qualified findings

This is source evidence, not a performance benchmark. The scan covers 514 native
and 477 Python production files at
`530f0a355cd8488deab887c7a6dba6a99e5486eb` (2026-10-01). Native/Python source was
clean; uncommitted scanner changes are identified by the analyzer digest. No
runtime allocator, CUDA byte/synchronization counter or endpoint timing was run.

- `scan.json.gz`: complete machine-readable inventory, source/scanner identity,
  locations, confidence, unresolved limits, actions and unique fingerprints
- `triage.json`: five independently source-reviewed groups with exact source
  links, scientific role, false-positive checks and next action

## Useful findings

1. **Streamed host DF whitening repeats twice.**
   `src/scf/cuda/df_coulomb.cpp:218–228` and `256–264` calculate the same
   raw-times-inverse producer, followed by H2D uploads at 231/267 and waits at
   244/280. The scanner detects identical blocks. Manual unchanged-input review
   establishes two passes on successful execution. This is the host-staged
   fallback, and bounded-memory policy constrains any reuse strategy.
2. **CPU RI-MP2 creates four vectors per tile.**
   `src/posthf/mp2_energy.cpp:320–322` constructs `g`, `x`, `ea`, and `eb` inside
   every occupied-pair/virtual-tile iteration. The source-derived count is
   `4*nocc^2*ceil(nvirt/tile)^2` sized constructions, with
   `16*tile^2+16*tile` requested bytes per tile. These are constructor counts,
   not measured allocator events. Reuse must preserve zeroing/padding.
3. **Conventional MP2 derivative scratch is recreated at shell-loop depths.**
   `src/posthf/mp2_derivative_common.cpp:53,64,76,131` allocates four transform
   stages. A reusable bounded workspace is a candidate; required transform work
   and force correctness are unchanged by the observation.
4. **Dynamic Jet fallback allocation is visible through helper calls.**
   The inventory follows native loops to `boys_values` in
   `src/integrals/s_integrals.cpp:131/149`. The manually reviewed Coulomb scratch
   at 230–270 has `(L+1)^4` slots but simplex write support: at `L=8`, 6561 slots
   versus 495 written slots. That support is **not automatically proved** by the
   scanner. This retained fallback/oracle is not a new generated-value defect.
5. **Response matrices have exact off-diagonal support and a dense consumer.**
   The Python AST finds `response_problem.py:148/156`. After manually checking
   layout invariants, support is `2*nocc*nvirt` of `(nocc+nvirt)^2` entries.
   `response_operator.py:140–142` consumes the density matrix via dense
   congruence. A producer-only packed replacement would break that contract.

## False-positive and scope decisions

The scanner does not count default or zero-sized vectors, known moved storage,
iterator/allocator constructors, capacity reuse, or first growth of loop-external
storage as repeated allocation. It refuses ambiguous same-file calls and known
escapes. Native scalar-call loop-invariance was deliberately withheld after
adversarial review exposed lexical scope, reference and floating-point-state
hazards. Unknown native syntax remains unknown, not a clean bill of health.

The existing CPU ERI geometry optimization is independently tracked in
[PR #1667](https://github.com/jinzhezenggroup/generativeqc/pull/1667), with its
own measured census and endpoint qualification. Its result is not claimed as a
new scanner discovery. Generated C++ inside Python templates, native simplex
support and the #1574 MP2 relaxed-weight proof are still outside this slice.
The before/after regression fixtures verify that moving fresh scratch outside a
loop and sharing a repeated source block removes the corresponding finding.

## Reproduce

```sh
python3 tools/audit_native_work.py --format json > scan.json
python3 -m pytest tests/python/test_native_work_audit.py \
  tests/python/test_work_audit_python.py tests/python/test_native_complexity_audit.py
```

Running at another commit intentionally produces a different receipt. To compare
semantic site identities, use `--compare` with the decompressed prior receipt.
All findings remain advisory; this does not close #1628–#1631.
