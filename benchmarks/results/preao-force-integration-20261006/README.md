# Repaired-source pre-AO force integration

This is an additional, source-scoped integration campaign, not a replacement for
the frozen [initial portfolio publication](../preao-force-portfolio-20261006/README.md).
It measures clean revision `1886ca21637a244c544927d128bafaa9b28ab295`, including
the second master merge and the reviewed indexed-layout/installed-asset repairs.
The rebuilt core SHA-256 is
`1e56e3afee967f91b3a0188bb283753b442bb36fd6f13d3888707a194aa64044`.

## Scope

- At 24 atoms: five interleaved AB/BA `current-default`/`auto` campaigns, each
  retaining cold, warm, moved and moved-warm complete E+F, actual Fock histories,
  force AO/jet/projection/gather counters and independent reference gates.
  These are 40 clean timing calls; two separate intrusive campaigns add eight
  calls. Excluded prewarms are not timing samples.
- At 3 and 48 atoms: one four-phase numerical/work smoke pair per size, 16 calls
  total. These check occupancy fallback and exact sampled-policy/work preservation;
  they are **not new five-round no-regression performance qualification**.
- The control is the incumbent-only registry on the **same rebuilt candidate
  source/binary**, not a separately built unmodified master. The independent
  references are copied byte-for-byte from the primary publication, with exact
  protocol equality checked. Native SCF remains sampled in both variants.
- Only n1, finite Slurm `main` allocations, and RTX 5090 are measured. GPU names
  are provenance: default admission has no product, SM-window or atom/AO/grid-size
  whitelist. Capability, resource headroom, occupancy and continuous work cost
  remain the guards; numerical cutoffs do not change.

## Exact reproduction

`bundle.json.xz` stores all 19 original UTF-8 JSON byte streams as strings, each
with its SHA-256: three references, ten timing campaigns, two intrusive campaigns
and four smoke campaigns. XZ uses one dictionary across records to fit the
existing PR evidence budget; it does not reduce precision or discard samples.

```bash
PYTHONPATH=python:. python benchmarks/results/preao-force-integration-20261006/reproduce.py \
  benchmarks/results/preao-force-integration-20261006/bundle.json.xz \
  --output .artifacts/preao/reproduced-integration.json
```

The helper materializes hash-checked original bytes and invokes the tracked
portfolio verifier. `qualification.json.gz` is a compact summary omitting only
duplicated raw/profile/Fock-history payloads; every original remains in the bundle.
`support.json.xz` retains complete before/after source checks, scheduler/device,
analytic/device/sanitizer gates, host/build/ccache receipts and checked AOT
binary/contract identities. Build both the required libraries **and their
`generativeqc_stationary_*_manifest` targets** after compiler-input changes.

The failed allocation 6240 is preserved in support: library-only targets left
stale AOT manifests and the fail-closed loader rejected them. No endpoint samples
from that attempt count toward qualification; identity checks were not bypassed.

`evidence.json.gz` accepts independent numerical gates. The user-authorized warm
objective is checked separately; generic all-phase performance remains `not-run`.
No significant cold gain, uncached deployment compile delta, observed global
allocator peak or timing on other GPU products is invented. Separate native
KS/XC inventory sharing and P0-D/P0-E remain deferred; #1893 stays open.
