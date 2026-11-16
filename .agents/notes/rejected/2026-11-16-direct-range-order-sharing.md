# Decision: reject the first shared per-order bounded range dispatcher

Status: rejected
Date: 2026-11-16

## Hypothesis

The runtime and AOT bounded range dispatchers retain distinct maximum-angular
order 8 and 12 admission domains. Their large per-order scientific bodies look
like candidates for sharing across those owners. The isolated prototype moves
orders 4--12 behind shared noinline per-order helpers, leaving orders 0--3 inline.
Runtime range/omega values remain runtime inputs; AOT range and omega identities
remain template constants inside their helpers. The existing scientific worker
still force-inlines within each helper.

This follows the ownership audit in
`2026-11-16-direct-shared-density-preparation.md`; it does not include that
already-rejected density-preparation extraction.

## Decision and complete-object evidence

Do not adopt this prototype. The full-object measurement does not demonstrate a
material reduction in compiler work or working set. Production Direct headers
remain unchanged, and no scientific method, architecture, angular admission,
bounded fallback or acceptance gate is removed.

Both complete `sm_80` fallback objects compile with NVCC 12.9.86 and verified
sccache 0.16.0 using `-O3 -DNDEBUG -std=c++20 -rdc=true`, PIC and the production
`--disable-optimizer-constants` setting. Both requests have zero cache hits and
four CUDA-stage misses. The baseline is the published shared-arithmetic owner,
not the original unshared implementation.

| Complete fallback owner | Published baseline | Candidate |
| --- | ---: | ---: |
| Embedded PTX bytes, same extraction method | 141,121,029 | 141,504,238 |
| Object bytes | 66,620,976 | 66,068,144 |
| Reaped compiler-worker maximum RSS, KiB | 5,579,972 | 5,538,956 |

Embedded PTX grows 0.27%; the object shrinks 0.83% and the single observed worker
maximum falls 0.74%. These are not a material working-set win. The separate NVVM
input PTX comparison, 149,129,566 to 149,532,122 bytes, also grows 0.27%; do not
mix that representation with embedded PTX. Both retain the same 108 kernel entry
identities. The candidate adds 54 shared high-order helper bodies rather than
removing the underlying scientific work. Symbol-level repetition alone does
not establish that this particular extraction will reduce complete-object work.

The candidate completes in 1,138 client seconds and its owned foreground cache
server exits normally after 18:58.77. Other host tests and compiler-related work
overlap the observations; no cold-build timing speedup or aggregate container
memory improvement is claimed. Neither PTX-only frontend measurements nor the
sccache client's RSS substitute for the complete compiler-worker maximum above.

## Contract evidence and limits

The executable comparison includes the actual dispatcher header and mock
scientific callees. All 748 output records agree, exercising both spins, the
8/12 admission domains, orders 0--13, runtime full/short/long range, zero and
signed-zero omega, 0.3 and its adjacent representable value, and custom omega
4.7. AOT long-range 300 and short-range 500 identities preserve forwarding of
batch, pointer and scalar arguments. This establishes the sampled forwarding
and admission behavior, not scientific numerical or physical-GPU correctness.

The compiler-resource gate already rejects the candidate. Do not spend or claim
device-link, other-architecture, independent numerical, GPU or complete Direct/HF
endpoint qualification for it. Those gates remain necessary for a future
candidate and for any separate unqualified production change.

Local ignored evidence is under `/tmp/gqc-wheel-range-dispatch/`: frozen shadow
headers, contract binaries/outputs, PTX ownership report, complete object,
compiler-worker timing, cache statistics and `compiled-report.json`.
The report's SHA256 is
`cf28de0c8b72db957e436aab0dd73e73c46035957d1d2a98a000a717f7553148`.
These are investigation artifacts, not permanent implementation-spelling tests.

## Revisit when

Revisit only with a materially different decomposition that removes substantial
complete-owner work or has a measured resource advantage. Preserve low-order
performance, both bounded admission domains and AOT constants; qualify every
affected owner together to avoid inconsistent weak device definitions. Require
independent numerical and appropriate complete-endpoint runtime evidence before
promoting it. Do not repeat this same extraction based only on large function
names or a cleaner dispatcher source.
