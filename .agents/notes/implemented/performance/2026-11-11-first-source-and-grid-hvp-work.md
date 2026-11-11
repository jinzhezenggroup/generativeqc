# Explicit first-integral outputs and structural grid HVP work

Date: 2026-11-11

## Decision and ownership

The canonical Hermite/Boys emitter now accepts explicit `full`, `first` and
`directional` output selections. They select mathematical roots before scalar
CSE, rather than intercepting generated source or matching target spelling.
The original full gradient/JVP remains the default, including for CUDA.

The bounded CPU adapter compiles each requested output lazily, after validating
the complete input tile. Its typed, record-local geometry binding is shared
across component selections, never copied between unrelated geometry types or
reused across records. The native record consumer admits operation 3 with two
doubles per component (value and scalar first direction). Operations 1 and 2
retain their raw-first and weighted-HVP packing. Zero motion never selects an
operation. The selected output participates in generated-source, native-artifact
and operation identities. Native output/counters remain transactional on bad
capacity, nonfinite input and late arithmetic failure. The existing conservative
worst-case numeric lease also bounds the smaller first/directional publications.

The molecular first-source consumer now admits one RKS density or two alpha/beta
densities before constructing a provider. Core and overlap derivatives are
shared. ERIs are streamed once per canonical quartet/operator/component page;
the common Coulomb density is the spin sum and each exchange density keeps its
own same-spin coefficient. Full-range sources consume scalar directions;
one-electron and range-separated sources retain bounded raw-first lowering.
Successful primitive records/component evaluations and source kinds propagate
to HVP diagnostics and sum across full-Hessian columns. Counts describe source
work, not the number of downstream spin Fock scatters.

## Grid work

The grid HVP retains the right AO motion already returned by `direction`, instead
of rebuilding it inside every left-coordinate/kind consumer. Weight-only left
motion has identically zero left/mixed AO and feature partials, regardless of
the right density response. That branch still evaluates and contracts explicit
first and mixed Becke weights, including both nonlocal quadrature legs.

The compiler's mixed feature program differentiates the primal AO inputs on the
left, with density fixed, then differentiates all live inputs on the right.
Density/right-AO/mixed-AO cross terms remain. `density_mixed`, the ECP projector
consumer of the same graph, migrates its feed contract with this change. General
first feature directions still support density tangents. No runtime identity,
shape heuristic, numerical-zero screening or alternate scientific algebra is
introduced; byte-budget admission and point-kernel eviction remain in force.

## Qualification and boundaries

Independent Libcint tests cover signed multi-exponent contractions, ssss,
d/p/s/p and ffff component pages, raw gradients, scalar directions, weighted
HVPs, translation, zero directions, capacity, late-failure publication and
owner recovery. A staged compiler-only package exercises raw and directional
outputs with public runtime and reference imports prohibited. Independent
fixed-density moving-integral Fock tests retain distinct spins and Coulomb-off
behavior, and require source work not to grow when adding the second spin.

Grid tests cover nested multistep feature differences with right density motion,
projected rho and structural zeros, and explicit weight decomposition for one
and two spins with and without nonlocal correlation. Complete coupled/public
molecular regressions and larger Ni endpoint qualification remain separate
receipts; no isolated source timing constitutes an endpoint speedup.

This integrates the mechanisms proposed in
`../../proposed/2026-11-11-factorized-directional-first-and-shared-spin-rhs.md`
and `../../proposed/2026-11-11-composite-grid-hvp-structural-zero-work.md`.
Their earlier experimental Ni acceptance is historical evidence only, not a
qualification of these production bytes. In particular this change does not
promote public Ni, unrestricted/ECP full Hessians, D3 composition or TS support,
and does not close issues #2844–#2848.

The integrated working tree passes 114 tests in four non-overlapping suites:
31 first-source/factorized/installed-compiler tests (278.04s), 29 grid/point-kernel
tests (9.76s), 34 coupled/public/lifecycle/Cs-ECP/WB97M-V endpoint tests
(2517.45s), and 20 standalone/public D3 regression tests (274.25s). The endpoint
suite includes both deliberately long nightly Cs-ECP directions at their
unchanged independent gates; it does not qualify mixed Ni/I/Cs or unrestricted
ECP totals. Six existing JUnit `record_property` warnings occur in the endpoint
and D3 suites. Logs/XML use `/tmp/gqc-production-first-source-final`,
`/tmp/gqc-production-grid-work-final`, `/tmp/gqc-production-directional-grid-endpoints`
and `/tmp/gqc-production-directional-d3-regression`, respectively. Final Ruff,
clang-format, public-contract classification, 603-module compiler dependency,
configured ty 0.0.82, changed-file precommit and warnings-as-errors Sphinx checks
pass. These are correctness regressions, not paired performance claims.

## Qualification-harness failure, not an accepted Ni action

The first final-byte Ni trial retained the matched 123-iteration native state
and both independently checked spin responses, but died with signal 11 at
900.80 seconds, exactly when an added `faulthandler.dump_traceback_later(900)`
diagnostic fired. Its receipt remains `native-hvp-start`, not an accepted action;
`/tmp/gqc-ni-production-directional-grid-final/process-failure.json` records the
failure, and `failed-harness-script.tar.gz` preserves the exact original script.

An isolated compiler-only reproduction uses Anaconda Python 3.13.9 and calls
`emit_factorized_second(output="directional")` with a 0.02-second repeating
traceback timer. It executes no native integral kernel and still stops under
GDB with SIGSEGV in `dump_frame`, called by `faulthandler_thread` on the watchdog
thread. The main thread is doing ordinary Graph arithmetic. Exact commands and
backtraces are in `/tmp/gqc-faulthandler-emitter-isolation.log` and
`/tmp/gqc-faulthandler-emitter-isolation-time.txt`. The same compiler emission without
that timer exits normally under GDB in 8.11 seconds and emits 3,951,648 bytes,
SHA256 `9cd9a70ff22000d74107965e6c8b613eb1ee6568ee5c96108791967b1a95c5e4`.
Control logs are `/tmp/gqc-emitter-no-watchdog-control{.log,-time.txt}`.

Consequently the next scientific trial removes only this added diagnostic,
retains the independent state/action gates and the external finite 3,600-second
process timeout, and writes a distinct receipt directory:
`/tmp/gqc-ni-production-directional-grid-no-watchdog`. This does not establish
acceptance of the unfinished HVP or rule out a separate numerical failure.
Do not reintroduce that asynchronous watchdog on this interpreter merely to
observe a long compile; a diagnostic must not become a new source of failures.
