# Advisory source-work audit

Run the standard-library-only inventory from the repository root:

```sh
python3 tools/audit_native_work.py --format json > work-audit.json
python3 tools/audit_native_work.py --summary-only
python3 tools/audit_native_work.py --compare baseline.json --format json > comparison.json
```

The default scan covers native files under `src/` and `include/`, and Python
ASTs under `python/`; vendored `src/xtb/native/` is excluded. Repeat `--path` to
restrict the inputs or scan generated native output. Python-embedded C++ strings
are not executable Python statements and are not parsed as native code. NumPy
loop findings are included in the same advisory JSON/CI artifact, alongside
the existing exact-zero materialization findings.

## What the inventory means

- **Loop allocation sites:** visibly sized vector construction, first growth of
  fresh local vector/aggregate storage, and named allocation APIs in loops.
  Storage created outside a loop and first resized inside it is omitted from
  the per-loop allocation inventory. Default and zero-sized vectors, known
  moved storage, iterator overloads and already-reserved capacity are omitted.
- **Python NumPy loop allocations:** source-bound calls to `numpy.empty`,
  `zeros`, `ones`, `full`, their `*_like` variants, `concatenate`, or
  `stack` inside a `for`/`async for`/`while` body or repeated `while`
  condition. Direct module aliases and `from numpy import ...` are recognized
  only when the visible binding is unshadowed; conditional or later local
  imports are not trusted. A loop's one-time iterable and `else` block are
  excluded from that loop's repeated region; an enclosing loop can still make
  them candidates. `asarray` and `reshape` can return views and are not
  treated as guaranteed backing allocations. These are only **potential
  executed sites**, not execution counts, byte totals, or proof that a function
  is prepared replay. Phase hints from variable names are advisory.
- **Native allocation/release roles:** lexical `realloc` is a possible host
  capacity change; `free`/`cudaFree` are releases, not new allocations.
  `cudaMallocHost` and `cudaHostAlloc` allocate pinned **host** storage, not
  device storage. None of these API sightings proves positive size, runtime
  reachability, or new backing storage per iteration.
- **Loop-callee sites:** an allocation/transfer/synchronization site reachable
  through a bounded, unambiguous same-file symbol chain from a lexical loop.
  Every chain includes caller and callee locations. Conditional execution and
  actual overload resolution across headers still require review.
- **Repeated producer loops:** identical closed arithmetic loop blocks with one
  written buffer and at least two read buffers. These are syntactic candidates;
  intervening mutation, alias identity and resource lifetime are not proved.
- **Structured zero materialization:** a Python AST analysis reports a fresh
  NumPy zero tensor and the union/bound of its admitted local writes. Supported
  forms include literal slices, Cartesian `ix_` blocks, `diag_indices` and
  range-indexed diagonals, and `triu_indices`/`tril_indices` triangles. Repeated
  unknown index names or attributes are omitted because their runtime values
  may be slices, new axes or scalar booleans that select the full tensor.
  Escaping aliases, unsupported mutation and unknown calls fail closed. A
  symbolic restricted write domain need not prove a strict storage reduction.
- **CUDA boundaries:** direct named transfer/synchronization sites and bounded
  same-file helper paths are an inventory. They do not establish device
  residency, transfer direction pairing, runtime bytes, or redundant waits.

Native parsing is a deliberately limited lexical analysis, not a complete C++
AST. Macros, templates, virtual/function-pointer dispatch, cross-file overloads,
user-defined allocator semantics and unparsed scopes can be missed. The audit
does not prove scalar-call hoisting or loop-invariance. It is incomplete by design and must not be interpreted as a zero-allocation or
zero-transfer guarantee when no finding is returned.

## Receipts and review

The JSON records the source commit/tree, scanned-source digest, scanner digest,
source dirty state, input roots, file counts, location, evidence, confidence,
disposition and next action. Static sites and call paths are never runtime event
counts. Fingerprints include semantic evidence and a deterministic occurrence
suffix so identical sites are not collapsed; line-only motion remains stable.
Comparison reports added/removed/unchanged semantic sites, not measured work
ratios. Moving code into another function intentionally changes its identity.

CI publishes the full JSON artifact. Findings do not fail CI and are not added
to the strict native matrix-chain gate. The ordinary Python test shards execute
the positive/negative scanner regression suite. Before introducing a hard gate,
qualify a named endpoint, scientific/resource identity, path role and independent
semantic counter. Setup, final publication, compatibility and oracle paths must
remain distinguishable rather than silently exempted by filename.

For a candidate, inspect the source and callers, state the explicit input and
ownership assumptions, identify the live consumer, and record whether the
finding is actionable, intentional or unresolved. Require complete endpoint
measurements before claiming a speedup. Changing scratch lifetime or removing
synchronization is outside the scanner's authority.

This is a first advisory slice of #1628, #1629, #1630 and #1631. It does not close
their runtime-counter, prepared-replay assertion, native exact-support,
TensorIR-integration or promotion-ratchet acceptance criteria. In particular,
the native MP2 relaxed-weight representation from #1574 is not yet automatically
proved by this analyzer.
