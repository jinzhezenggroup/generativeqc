# Decision: Use native capacity for budgeted dense CPU KS forces

Status: implemented
Date: 2026-11-23

## Decision

Use the CLI's dense Direct force fleet inventory in Python when the linked
strict-FP64 named program and its correction reservation are available. This
bounds retained dense ERI values, force outputs and the maximum serialized
phase workspace, not total process memory. Explicit corrections and cold-start
policies still compose their own resource requests. Native admission checks
the force capability of every prepared item, including for implicit calculator
budgets. A positive capacity result cannot fund compatibility derivatives.

Keep the existing conservative floor for libraries without the force query
or its derivative package, and for force upgrades of batches prepared against
an energy-only envelope. Distinguish either case from malformed or refused
native inventory, which is authoritative. Energy-only replay under a native
force plan does not retire its reservation. Reuse the validated plan for
ordinary coordinate replay; revalidate when the model or its controls change.

## Evidence

On base `cae2959e1`, all 152 CPU CTests, 251 selected installed pytest cases
(16 skipped), the additional installed CLI cutoff case and strict Sphinx
passed. The installed SDK and CLI force contract runs with Python and the
compiler off the runtime search path and uses independent PySCF references
from the test harness. The native test compares dense and packed provider
bytes and verifies transactional failure. A matched PBE0 water CLI and Python
plan both reserve 67,281,232 host bytes and reject one byte less before SCF.
Python also covers explicit HF guessing, implicit and scalar preparation,
ragged replay, corrected UKS, unavailable and malformed queries, and refusal
of compatibility derivatives under the smaller bound.

Seven interleaved process pairs per case, after one warmup pair and on one
EPYC 7K62 core, compare the previous PBE-D3(BJ) install with the new install.
Timing includes calculator construction, estimate, preparation, cold/warm/moved
energy and forces, and close. Energies, forces, SCF iterations, Fock builds,
warm-start choices and sampled capacities are identical across variants.

| Endpoint | Old → native host plan (bytes) | Old → native median (s) | Cold / warm / moved Fock builds |
| --- | ---: | ---: | ---: |
| Ragged PBE0 RKS (He, water) | 268,746,672 → 67,421,800 | 0.17273 → 0.16608 | 16 / 8 / 14 |
| B3LYP-D3(BJ) UKS (HeH) | 268,576,588 → 67,250,332 | 0.12261 → 0.11105 | 8 / 4 / 6 |
| B3LYP-D3(BJ) RKS (CsH, scalar ECP) | 275,496,180 → 126,356,556 | 6.80129 → 6.77065 | 16 / 4 / 11 |

The smaller reservation does not imply a reduction in observed allocation or
RSS. Raw endpoint and compiler-cache receipts are under
`/data/jzzeng/gqc-2227-sdk-JsefEq0x/cpu-dense-force-budget`. This is scoped
CPU evidence; it does not admit a new force model or CUDA provider.

## References

- [Packed Python force plans](2026-11-23-python-packed-eri-force-plans.md)
- [CLI host force admission](2026-11-23-cli-cpu-ks-force-host-admission.md)
