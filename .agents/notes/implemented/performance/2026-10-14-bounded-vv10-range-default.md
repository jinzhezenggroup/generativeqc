# Decision: bounded automatic range residency for initialized VV10

Status: implemented (qualification candidate)
Date: 2026-10-14

Full range-value storage plus exact warm reuse at integration source
`68f4c5f70acbe9f4c2e54a9030c694183787922c` completes all 36 original-protocol
calls at 3/12/48 atoms. At 48 atoms the cold/warm/moved/moved-warm medians are
253.488/11.408/212.392/11.392 seconds. The 12-call populations retain the 48×16×32
grid, FP64, 100-step limit, 1800-second deadline and independent 1e-8/1e-7 gates.
A partial mask is slower for small changing-density points, so it is not the
automatic bucket policy.

Select full screened SPD radial values automatically only for restricted strict
VV10 native initialization with at least 65536 grid points, the existing reuse
threshold and at most 2 GiB of optional values after complete admission. A moved
geometry running the same bounded initializer has repeated range-K consumers
and may prepare a new source. A warm call only retains an already audited source.
No warm call constructs new quartic storage. Selection strings, buckets and
capacity remain part of the retained source policy; changes restore the separately
certified incumbent energy baseline. Explicit zero selects either fallback.

The narrower 2 GiB automatic domain excludes the unresolved large inventory:
96 atoms require 14,985,773,728 full-cache bytes, exceeding the measured allowance.
Its 12.44 GB partial experiment still times out after six passing calls. It remains
an explicit experiment, with no automatic expansion or claim that #2426 is solved.
Capacity refusal and unsupported methods retain ordinary physical evaluation.
Original source/omega/screening, resource lifetime, energy and force acceptance
are unchanged. The newly automatic source needs its own full-protocol qualification
before its measurements can replace the published default curves.
