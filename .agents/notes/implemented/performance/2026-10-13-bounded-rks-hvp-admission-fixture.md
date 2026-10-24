# Decision: separate RKS AO admission from exhaustive high-angular compilation

Status: implemented
Date: 2026-10-13

## Problem

Nightly core-c [run 38343058044, job 115118167657](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38343058044/job/115118167657)
failed in `test_rks_hvp_admits_resource_bounded_domain_above_12_aos` after
4,937.627543 seconds. Its last generated ERI second-derivative C++ compilation
hit the existing 300-second total compiler budget. The other core-c results were
8,412 passed and 1,138 skipped. The failure artifact is
[11790238281](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38343058044/artifacts/11790238281).

The compiler timeout applies to one translation unit, not the complete test.
The prior 4,637.6 seconds include SCF/response, derivative graph generation,
earlier compilations and execution. The hosted diagnostics record only test
start/end, so they do not establish the split among those phases or the angular
class of the final failed translation unit. Do not describe the entire 82-minute
test as a single compiler invocation.

The old fixture used one s and two Cartesian d shells to reach 13 AOs. The
second-integral provider visits all ordered angular classes, even with zero
weights. Its 64-AO-component / six-coordinate tiling produces 96 distinct ERI
compile tiles across s/d classes. dddd alone has 1,296 Cartesian components,
21 AO chunks and two coordinate tiles. This is incidental compilation work for
an AO-count/resource-admission assertion.

## Decision and retained gates

- Use one s and four p shells, still exactly 13 AOs, for the real native RKS HVP
  admission case. Strict SCF, the normalized translation direction, finite and
  translation-zero HVP gate (`atol=2e-7`, `rtol=0`), complete source coverage,
  64 MiB integral budgets and provider/workspace bounds remain unchanged
- The s/p angular inventory has 34 ERI compile tiles; pppp has 81 components and
  two AO chunks. This is test workload selection, not a production optimization
- Retain all-d numerical coverage with a separate compiled dddd HVP over three
  weighted components and four coordinate outputs. Components 0, 310 and 1295
  include xx/xx/xx/xx and zz/zz/zz/zz extremes plus xy/xz/yy/yz. Output 11 checks recovery
  of the last-center z response. Distinct exponents, asymmetric centers and
  signed nonunit weights yield an explicitly nonzero independent answer
- Freeze that answer from analytic PySCF/libcint in
  `tests/reference_data/second_derivatives/dddd_hvp.json`. It includes all inputs,
  PySCF version, the libcint binary hash and generator source provenance. Normal
  execution does not import or call PySCF for this new test. Keep the existing
  `5e-11` absolute / `2e-11` relative native gates and a `5e-11` translation gate
- Retain the old shell-quartet-dominated workspace branch with a cheap resource
  fixture: 13 AOs, shell sizes `(1,6,6)`, 165,888 bytes accepted and one byte less
  refused before CPKS. The new physical s/p fixture has a 21,632-byte bound
- Existing independent d/p and through-f second-derivative tests are unchanged

## Evidence

Base is master `bcf4e746444328109c37d7743bd1c319420f9a6e`, tree
`8bbb9140bb1a15a9ccc5663209c90730c828ad03`. A source-matched CPU Release build
passes using GCC 14.2 and verified ccache 4.14.1. The artifact-cold complete
13-AO candidate passes in 147.648 seconds locally. One-off instrumentation
observed 79 compiler calls totaling 104.721 seconds, including first-derivative
and response work; the longest call was 18.121 seconds. Exactly 50 compiled
second-derivative tiles cover the 34 ERI and 16 one-electron cases. The remainder includes graph generation and execution.
Compiler caching remained enabled; no cache was cleared or identity overridden.

This local successful endpoint and the hosted failed endpoint use different
runners and workloads: they are not a matched scientific speedup experiment.
The unchanged supporting directional, resource and independent second-derivative
suites passed 68 tests with 17 intentional GPU/slow skips. The bounded dddd
native test was separately checked against the live analytic oracle before
freezing the reference; it passed in 51.71 seconds, with its GPU case skipped.
The ordinary pytest admission case separately passed in 163.18 seconds. The
frozen dddd gate and all 13 resource-control cases passed with PySCF absent
from the test environment (14 passed in 48.42 seconds). An independent reviewer
recomputed all four frozen values exactly and found a translation residual at
most 1.11e-16. No GPU qualification is claimed.

Final focused validation on the original base: 71 passed and 18 intentionally
skipped in 260.28 seconds across the four directional, second-derivative and
resource suites. The admission case took 161.28 seconds and the frozen dddd
native case took 44.49 seconds. All repository pre-commit aggregate gates pass.

Before publication, master advanced to `f999a19e1b082155670f0a19d85761733e746ccc`
(tree `1e9e3eee8c8c1ea197b135290cce5205e6c8d880`), changing automatic initial
guesses through #2370. The five-file candidate rebased without conflict. A fresh
source-matched cached CPU Release build and the 13-AO admission/resource rerun
passed: 14 tests in 162.84 seconds, with the admission case taking 161.88 seconds.
The new library SHA-256 is
`6a272d5f5546535a673d615c38c563d3c2c675b4875e356ef310787781631f9c`.
The original 71-test receipt remains attributed to the original base; it is not
relabeled as a full-suite result on the later initial-guess source.

To recheck the frozen independent reference explicitly (not part of ordinary CI):

```python
import json
from pathlib import Path
import numpy as np
from tools.generativeqc_validation.second_derivatives import libcint_eri_hessian

p = Path("tests/reference_data/second_derivatives/dddd_hvp.json")
r = json.loads(p.read_text())
h = libcint_eri_hessian(tuple(r["angular"]), r["exponents"], r["centers"])
m = h.reshape(12, 12, -1)[:, :, r["component_indices"]] @ np.array(r["weights"])
v = (m @ np.array(r["direction"]).ravel())[r["output_indices"]]
np.testing.assert_allclose(v, r["expected_hvp"], atol=5e-11, rtol=2e-11)
```

## Rejected alternatives and limitations

Do not raise compiler timeouts, delete the >12-AO assertion, loosen numerical
thresholds, or replace real HVP execution with a mock. A first s/p exponent trial
`1.2,0.55,0.25,0.1` did not satisfy unchanged strict SCF; the committed
`2.0,0.8,0.35,0.15` choice does. No convergence tolerance was changed.

This split does not qualify exhaustive 64-component dddd compilation or full
d-shell RKS/XC integration; the old fixture's d-shell occupied admixture is not
numerically equivalent to p-only virtuals. The focused independent dddd gate
protects integral arithmetic, and the synthetic budget case protects storage
admission. Production high-angular graph-generation/compiler cost remains
unaddressed. Revisit it as a separately authorized production performance task
with complete endpoint evidence, rather than rebuilding an unbounded CI fixture.
