# Native structured materialization source receipt

Source commit: `d2593148661f0b491b6254a59779017c21ac74f4`.
Source tree: `00d3a5eeb26dd78bbb4d8e94f7eb645781dca7bb`.
The scanned files are unchanged from that source commit. The scanner was new
and uncommitted during receipt generation; `scan.json` binds its exact bytes
and both imported scanner modules with SHA-256. This is a local Windows source
receipt, without native compilation or CPU/GPU execution.

Reproduce from the repository root:

```sh
python tools/audit_structured_materialization.py \
  --path src/posthf/mp2_gradient.cpp \
  --path src/posthf/mp2_gradient.hpp \
  --path src/posthf/mp2_force.cpp \
  --path src/posthf/mp2_derivative_common.cpp \
  --output scan.json
python -m unittest tests.python.test_native_structured_materialization
```

The actual scan inventories four files and 18 zero-initialization candidates.
All 18 remain unknown: this receipt does not automatically certify MP2 support.
The runnable regressions certify closed native examples and include independent
small-shape address enumeration for occupied/virtual, dense and triangular
domains, alongside mutation/alias/control-flow/overload counterexamples.

## Source-matched manual triage

| Concern | Current source evidence | Conclusion / missing gate |
| --- | --- | --- |
| Dense producer | `mp2_gradient.cpp:176-198`, `initial_orbital_weights`: fresh `OrbitalRhs result`, `two_electron.assign(fourth_power(n), 0.0)`, diagonal/Fock-like and `iajb` writes | Visible dense-oracle producer candidate. Header/default constructor and checked-helper semantics are not automatically proved. |
| Dense consumer | `mp2_gradient.cpp:323-364`, `canonical_orbital_rhs`, and `:422-458`, `canonical_lagrangian_weights`, consume and mutate the dense representation | Producer-local support is not a complete relaxed-weight lifetime proof. |
| Current production producer | `mp2_force.cpp:131-144` and `:220-235` select the streamed RHS and Lagrangian producers | Historical #1574 production dense-weight description must not be reused as a current defect claim. |
| Structured ABI | `mp2_gradient.hpp:28-70`, `FactorizedTwoElectronWeights`, `OrbitalRhs`, `LagrangianWeights`; `mp2_gradient.cpp:477-483` moves Fock/`iajb` factors | The native structured production contract already exists. No new ABI is implemented here. |
| Derivative consumer | `mp2_derivative_common.cpp:126-139` validates dense/factorized alternatives; `:170-209` branches on representation | A whole-consumer proof needs type, helper and cross-function analysis. |
| Oracle gate | Header describes oracle/legacy storage; native tests call dense APIs | This audit did not establish a hard native maximum-size gate. Neither comments nor small fixtures establish one. |
| Common IR | Standalone scanner; existing common audit is unchanged | Shared work-audit/TensorIR integration and full MP2 exact-support proof remain #1631 scope. |

These are manual source observations attached to exact file byte identities,
not runtime reachability, memory counters, timings or scientific acceptance.
The PR uses **Refs #1631** and leaves the parent issue open.

## Local validation

The focused suite plus consumed native-work/complexity/Python-audit regressions
passed: **156 tests and 34 subtests**. The new tool has 16 unittest cases and
can run without pytest or a native build. The default CLI also completed an
actual 606-file scan (123 unknown inventory candidates), retained as an ignored
local qualification artifact rather than adding a second large tracked receipt.

Relevant lint, formatting, source-boundary and retention checks passed. The
Windows full pre-commit run encountered existing publication bytes expanded
from LF to CRLF and two default-GBK reads. The two encoding checks passed with
`PYTHONUTF8=1`; publication compaction is additionally checked on exact Git-index
bytes without altering existing evidence. Required PR CI remains a separate gate.
