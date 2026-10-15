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

### Source-matched Linux qualification

At `d5635169c33963e22e4e9bc2cc147d91e2b3bdf8`, the CPU-only Notebook
`i1123-gxx-1015` ran Python 3.11.16, canonical
`/usr/bin/x86_64-linux-gnu-g++-11` (Ubuntu 11.4.0) and verified sccache 0.16.0.
The selected private environment image advertises additional GCC 12 tools, but
its canonical `/usr/bin/g++` resolves to GCC 11; this evidence is GCC 11.4, not
GCC 12.3. No system compiler selection was changed.

All 11,049 archived regular files were checked against committed Git blob bytes
before transfer and against their sizes/SHA-256 on Linux. The archive used
`git -c core.autocrlf=false archive`. The two repaired files came from a second
raw Git archive with the updated full input manifest. The executed production
and test blob identities are respectively
`e7747b007a44e8b4e8f143eb64354a44063419db` and
`d3d3a1bc1035ba085d66ce4eeeb0a3f1ef2c711b` (verify against the source commit).

The source-matched GCC C/C++ closure, cache/store, publication, boundary and alias
suite passed **137 tests, zero skips**. The C++ standard-header fixture independently
matched the object compiler's `-MD` dependency file, bracketed compilation with
equal reusable keys, published/retrieved the object, deleted the original, and
linked the stored object for execution against four independent Decimal rows.
This separate execution link is not a link-closure cache claim.

- Stored object SHA-256:
  `4edf91198a6b16973455181b4e80ee7b751b79f912d013b1f21e5f17782f7551`.
- Executed shared library SHA-256:
  `9fc31cbb8fa7d53246ff555e1e99f9e79b115da1e213615aebcb2ff0ff4c279b`.
- Ordinary unchanged replay increased shared-cache hits from 0 to 1; refreshing
  the opaque manifest salt increased misses from 1 to 2. These counters belong
  to the recorded foreground commands, not unrelated private JIT cache slots.
- Changed assembler-only `.incbin` payloads produced different raw object hashes
  `67d52f342a1ea6062d0a81954fc8c47b18f0f0030413ecfb976cf3a6507a6286` and
  `cb5a305f6f675e91b6d407180a5dd2b262632231076699c03c90755e75158826`.
  These prove fresh objects; cacheable replay/seed hashing is demonstrated by
  the separate ordinary-source control, not inferred from `.incbin` alone.
- A one-off comparison against upstream baseline
  `e88dee9d2f242f76bb8f712aaff2cd0cab103e41` observed identical GCC C recipes,
  complete closure payloads and keys under three ordered flag combinations.
  It is retained evidence, not a permanent source fingerprint test.

The initial source `7e6bea0ed6f672198ca486a42ee6a6cef2353650` produced
**133 passed / 4 failed**. Actual G++ diagnostics spell the PATH-selected
assembler as `as`; requiring an absolute diagnostic token rejected that valid
native job. The repair resolves only this known token under `CPU_ENVIRONMENT`
and requires agreement with the independently queried canonical assembler.
Unexpected relative commands still fail closed. Initial failure logs remain.
One CLI invocation returned client exit 1 despite remote explicit exit 0;
downloaded test/runner logs, raw stored/library hashes, receipt rows and cache
counter receipts independently confirm the remote result. A later remote
compatibility/receipt-archive command returned client and remote exit 0.

Raw evidence remains under the task-owned project directory
`/inspire/qb-ilm/project/chemicalreaction/czxs25220150/issue1123-gxx-20261015`.
The locally retrieved ignored `.cache/gxx-evidence.tar.gz` SHA-256 is
`d28e7e39c774863f26085a1a3ac33bf2213061f4d539393e82256ddf74fcbcb5`;
its object/library bytes were rehashed after download against the execution
receipt. The sole zero-GPU 1 CPU/4 GiB Notebook had a live 0 points/hour quote,
requested priority 10 and received 6/HIGH. The 40-minute platform timer was
read back while running; it was explicitly stopped and verified STOPPED after
receipt recovery, within the authorized 45-minute cumulative window. No GPU,
installed bundle, production runtime or scientific endpoint qualification.

### Review correction: cache-command environment

Review of `aadcef12d` identified that the ordinary cache-counter commands
merged the entire parent environment back into `CPU_ENVIRONMENT`. Unlike the
collector and controlled object producer, these commands could inherit compiler
or loader overrides. The test now admits only the controlled compiler mapping
plus HOME, cache directories and the existing sccache server address. Statistics
queries use the same restricted mapping. The real Linux test injects invalid
GCC_EXEC_PREFIX, COMPILER_PATH, CPATH, CPLUS_INCLUDE_PATH, LIBRARY_PATH and
LD_PRELOAD values into the parent to exercise this boundary.

This correction changes the test producer, not the production collector. The
137-test run and archive above remain measurements of their original snapshot;
they do not claim execution of this later environment-poison control. Current
PR Linux CI must execute the corrected test at the new exact head before
delivery. No Notebook was restarted to repeat unchanged production qualification.

### CI correction: persistent-cache test inputs

Run `38700565333`, core-c job `116313767233`, reported 22 passing G++ tests,
one failure and zero skips at `480eebe26`. The failure occurred after successful
compilation/replay: both fixed ordinary-control payloads were cache hits, leaving
the expected changed-salt miss counter at zero. A restored persistent cache can
already contain both fixed input states; a warm hit is not a collector failure.
The failed JUnit record is retained in ignored
`.cache/pr2519-corec-debug.zip`. No production collector change follows from it.

The control now prefixes the actual opaque file contents with fresh per-run
bytes. Its first two states remain byte-identical, while its third state really
changes the manifest-covered input. This avoids assuming a cold persistent cache
without clearing it, weakening it, or inventing a source/toolchain identity.
For ccache, a test-specific `CCACHE_STATSLOG` and `--show-log-stats` attribute
the counters to these commands even under parallel CI, as described in the
[ccache 4.14 manual](https://ccache.dev/manual/4.14.html#_cache_statistics).
The namespace and counter source are retained in the test receipt. Existing
production flags, salt derivation, object identity and immutable store are
unchanged. This later test repair needs fresh exact-head CI and independent
delta approval; the earlier approval and measurements are not promoted to it.

## Consequences and revisit conditions

This is a narrow opt-in object tool, not an automatic generated-source consumer.
Broader language/ABI/link/toolchain support needs its own dependency model and
independent real compiler/store evidence before admission.

Refs #1123. Existing GCC C, Clang C and NVCC slices retain their own domains.
