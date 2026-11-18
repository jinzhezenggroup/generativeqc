# Decision: explicitly qualify positive Full moments for packaged stationary CPU ERIs

Status: implemented
Date: 2026-11-13

## Problem

The factorized host first/second and physical-owner matrix sources already use
the shared positive Full moment evaluator, but packaged stationary CPU first
derivatives still used the generic 64-node moment quadrature. An actual native
integral-gradient prefix on the original saved 81-AO, six-atom spherical basis
spent about 11.16 of 14.72 seconds inside packaged ERI leaves. This is a distinct
consumer from the previously optimized nuclear XC maps. Optimizing grid maps
does not remove this cost.

## Decision and invariants

Add an explicit, default-false `positive_full_moments` policy to the weighted
primitive and CPU first-derivative emitters. Only the packaged stationary CPU
s/p/d AOT schedule opts in. The compiler changes the existing geometry template
policy, not derivative algebra, ERI weights, Cartesian components, exponent or
coordinate conventions, dispatch ABI, finite batch admission or transactional
publication. Native ownership remains with the existing geometry and moment
templates. The existing positive-series/downward or erf/upward evaluator keeps
its bounded positive-quadrature fallback and FP64 derivative identity.

Both emitter policies reject non-boolean values. The cached CPU entry uses typed
keys: Python normally considers `True` and `1` equal, so validating only inside
a non-typed cached function could admit integer one after warming the true key.
The weighted emitter refuses positive policy on a non-CPU backend. Generic CPU
and CUDA default source generation is unchanged. LR/SR evaluation retains its
existing numerical path, domain and bounds; this change does not qualify new
range-separated consumers or device execution.

## Independent qualification and provenance

The prototype uses the actual `cpu_stationary_integral_gradient`, with original
saved native C reconstructed into D, explicit zero W and exact exchange 0.2.
Zero W is appropriate only to this ERI component diagnostic; it is not a
complete stationary gradient. Normal prepared-value construction remains in
each run. Transparent dispatch interposition stops by controlled exception
after exactly 8,000,000 completed primitive records. No partial gradient is
published. All six interleaved runs visit the same 263 kinds in 33 shards, with
identical work histograms and sampled inputs. No native/reference SCF runs.

The private prototype changes only the geometry template policy. Across three
pairs, the complete interrupted apply changes from 14.708–14.726 to
7.074–7.108 seconds and leaf time from 11.158–11.164 to 3.530–3.533 seconds.
This is not a full-gradient, Hessian or optimizer speedup. Independent libcint
primitive derivatives cover 526 actual-record cases, with maximum error
1.1379786002407855e-15. All 8416 captured records agree with the frozen baseline
within 7.105427357601002e-15. Signed batches, invalid geometry/batch admission
and output canaries preserve the fixed 3e-11 absolute/relative gates.

Permanent production tests cover all 313 admitted ERI Cartesian kinds, not only
the observed prefix: three independent libcint geometries per kind, signed
batches, late-record failure, canaries and invalid admission. They also exercise
strict policy typing after cache warming and refusal of unqualified CUDA opt-in.
The native integral-gradient tests retain the independent Jet-based assembled
gradient control, asymmetric weights, symmetry folding and resource gates.

The prototype belongs to source
`26f199f94962c7da7faeb4d5a7c642b19a8e803d47a584ed3bfcc782f7f9ab34`,
core SHA256 `78c5349dd295a252c1a6ce3f4555adebc943f130286d37594c674d20eb7bf0c9`.
Its sealed archive SHA256 is
`5854fa20c4ecd30c635b2c51deb80c3e881e2805697aee2b8d028fc04efb6afe`.
Historical validators and profilers originally link a mutable build path. After
a rebuild, explicitly use the frozen old core rather than mistaking the new
production implementation for the old baseline. Recovered original generated
shards are verified against their original hashes; historical emitter bindings
are checked against the frozen source inventory, never edited current files.

The production core binds source
`dcbd294e1f6f1ef08869582c859a75f8e16e0e7ae111b0b5a55c659233fb9515`,
SHA256 `5741816b45359b70793ad3295a488d8e9010e6891080f414c58929a0fd3cf536`.
Raw production receipts belong to
`build/qualification/host-stationary-positive-source-20261113`.

The production comparison uses only transparent capture, with no private
candidate router. Three interleaved pairs explicitly load the frozen old/new
cores and check each returned source identity. Median interrupted apply is
14.700595648 versus 7.122436179 seconds; leaf time is 11.126738764 versus
3.535497363 seconds. Prepared-value medians remain separately reported at
4.399808553 and 4.336321857 seconds. All work histograms and captured inputs
match. Direct production validation repeats the 8416-record parity, 526 libcint,
263 signed-batch and 2104-invalid-case-per-core gates, with the same maximum
errors as the prototype.

One-off migration audits cover all 46 shards and 362 requests: generic CPU and
CUDA default sources are byte-identical; packaged CPU source differs only at
the 313 ERI geometry policy calls. Mathematical graphs and all S/T/V code are
unchanged. This is a retained opt-in audit, not a permanent generated-source
fingerprint test. An initial 150-second audit cap was insufficient for repeated
CPU/CUDA emission; its timeout remains recorded, followed by a finite successful
audit with per-shard progress and exact recovery of the original sampled sources.
No expensive molecular endpoint was rerun for that harness correction.

## Rejected alternatives and consequences

- A global geometry-default switch would silently change unqualified generic,
  CUDA and range-separated consumers. Keep the opt-in local and explicit.
- A second Boys/derivative algebra is unnecessary; reuse the qualified evaluator
  and preserve its bounded fallback.
- Do not loosen scientific/optimizer tolerances or infer complete endpoints from
  a prefix. Both public ECP saddle guards remain closed.

The change reduces work inside a primitive, not the number of submitted
primitive records, source preparations, response actions or Hessian columns.
Full raw-H/response/unscaled-mode and stationary/TS acceptance remain separate
requirements. Revisit other consumers only with independent numerical/domain
gates and source-bound complete-consumer evidence.
