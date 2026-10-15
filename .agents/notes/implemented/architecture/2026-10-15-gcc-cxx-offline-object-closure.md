# Decision: explicit native GCC C++20 offline object closure

Status: implemented
Date: 2026-10-15

## Problem

The existing CPU collector admits GCC C and explicit Clang C. A G++ object has
different driver/frontend, language, standard-library configuration and ABI
inputs. Removing the C suffix check alone cannot establish a C++ closure.

## Decision

Extend the existing collector with explicit `language="c++"`, canonical G++, a
`.cpp` source and exactly one C++20 standard flag. Reuse CpuToolchainManifest,
CacheClosure and the immutable object store. Keep the C branch's defaults and
payload/key construction unchanged. No new cache owner or runtime selector.

Observe actual cc1plus/as selection and the two-job object plan. Validate the
temporary assembly handoff before normalizing that one nondeterministic name;
retain other job arguments. Bind specs, full macro/ABI output, include search,
headers and producer-asserted opaque configuration to the existing closure.
Require regular ELF tools, bracket driver selection, repeat observations and
retain the existing timeout, source/path/cwd/environment and manifest checks.
Reject neighboring PCH candidates and unsupported flags rather than promising
coverage of modules, PCH, alternate ABI options, cross compilation or linking.

## Rejected alternatives

- Broadening the GCC banner/suffix gate would omit cc1plus and language identity.
- A separate C++ cache would duplicate ownership and publication invariants.
- Hashing the raw dry-run plan would make random temporary filenames invalidate
  every collection; ignoring the plan would miss actual frontend selection.
- Inferring completeness from dependencies/ldd would omit absent headers,
  implicit configuration, loaded-library and assembler-only inputs.

## Invariants

Only the producer asserts an immutable complete snapshot. Fresh pre/post keys
must match before publishing the newly compiled object. The entire returned seed
is passed through a verified compiler-cache launcher. C/C++ cannot share a key
merely because their source bytes match. No production formulas, source emission,
scientific policy, installed packaging, CUDA qualification or parent-issue closure.

## Evidence

Protocol tests and native Linux qualification are in
`tests/python/test_bulk_aot_gxx_closure.py`; current handoff and limitations are
documented in `docs/developer/libxc_cpu_aot_closure.md`. Qualification receipts
bind raw object/store/library identities and Decimal rows. Measured results are
recorded separately after source-matched execution; this decision alone does not
assert that validation or delivery has passed.

## Consequences and revisit conditions

This is a narrow opt-in object tool, not an automatic generated-source consumer.
Broader language/ABI/link/toolchain support needs its own dependency model and
independent real compiler/store evidence before admission.

Refs #1123. Existing GCC C, Clang C and NVCC slices retain their own domains.
