# Automatic small-CTA canonical Direct sources

The default order-3/4/5 owners use 32-thread CTAs and 1/3/6 component slots,
covering every s/p/d/f domain of up to 27/81/162 components. Orders 6--9 keep
256-thread owners. Persistent grids are bounded by the dense shell-pair
product and 4096, with clamped factors and no D2H count read. There is no
method, basis, molecule-size whitelist or opt-in.

## Source and measurement scope

- Measured upstream: `01eb694578dab5ba96175fbedcad67e1db6bac37`.
- Matched baseline: `f6b9220ade1b8e90189c8562f28a501431cb9600`, that upstream
  plus the then-pending automatic order-5-through-9 work. It is **not raw
  master**. That work subsequently merged as #2217.
- Qualified latest integration base: `dec25f5922f8a63c56ecd9f6f89eb345595f3755`.
  Compiler/kernel/test source hashes match the measured bounded-grid arm;
  four native suites, eight Libcint cases, low/high-order exact work, all 56
  source actions and independent complete three/twelve-atom physical gates
  pass again. The previous `2b68d10a9f43175d796500d7853ed90245f24315`
  integration remains a separate historical receipt. Fresh integration
  samples are qualification, **not matched throughput arms**; the frozen
  measurements below are not relabeled as latest-master measurements.
- All device execution uses finite Slurm jobs on n1's RTX 5090, preserving
  assigned visibility. The two complete twelve-atom arms and GPU4PySCF control
  share one job/device. Runtime job, visibility, GPU and library/source hashes
  are retained in [the compact receipt](evidence.json.gz).
- Release/sm_120 compilation uses verified ccache 4.5.1 and the existing shared
  cache. Compiler commands and before/after statistics remain in the local raw
  evidence. Caching is not substituted for source or device qualification.

The fixture is the README **water proxy**, not an OMol25-distribution sample.
It uses the complete spherical def2-TZVPD H/O basis, including diffuse/f
functions, exact Direct WB97M-V/RKS FP64, a 48x16x32 per-atom Becke-3 grid,
E/density/screening tolerances 1e-12/1e-10/1e-12, max100 and VV10 rho cutoff
1e-8. The same four-GiB force budget applies to every native arm.

The complete clock covers preparation, SCF, physical forces, synchronization
and host publication. Imports/context/Calculator construction precede it;
teardown and serialization follow it. Compiler/artifact caches are preserved.
Every sample has a fresh owner, not a warm solver/density. Three-atom
allocation/JIT-first-use samples are retained but excluded from medians.

## Complete endpoint acceptance

| Endpoint | Automatic 5--9 baseline | Small CTA + bounded grid | Work |
| --- | ---: | ---: | --- |
| 3 atoms, three alternating repeats, median | 11.706517 s | 11.917956 s | 15 iterations/Focks |
| 12 atoms, ABBA, two samples/arm, median | 247.464159 s | 194.720287 s | 21 iterations/Focks |

The complete twelve-atom endpoint improves **21.31%**. The three-atom endpoint
regresses **1.81%**; this is not a universal speedup. Its first-density source
total grows from 0.245797 to 0.266688 seconds, consistent with about 0.3
additional SCF seconds across fifteen Focks. The bounded grid removes idle
launch work but does not eliminate that small-domain amortization tradeoff.

All physical E/F gates pass at 1e-8 Eh / 1e-7 Eh/Bohr. Exact accepted errors,
residuals, every endpoint sample and phase times are in the receipt. Twelve-atom
AO work (excluding only discovery **time**), force work and force traffic
match across all four samples. Discovery remains inside complete timing.

The independent GPU4PySCF full-Fock control completes in **182.512815 s**, with
45 iterations / 46 Focks. Native remains about **1.067x slower** on this
endpoint. This is **not a stock-default comparison**: the reference explicitly
rebuilds the full Fock using its own engine-local seed. The inherited protocol
field `reference_fock_policy=incremental` is superseded by the recorded
`reference_full_fock=true` override. No native/reference density is borrowed.

## Source diagnostics, not complete endpoints

All 56 first-density twelve-atom CUDA-event source actions improve from
**7.401796 to 4.889548 seconds**. Event/resource collection is intrusive and
stops before public E/F; do not substitute it for complete timing.

The paired 96-atom diagnostic selects only actions **4/5/8/32/33/36**, the
representative full/LR order-3/4/5 blocks. Its baseline versus fixed-grid
32-thread prototype totals improve from **608.579418 to 391.283691 seconds**
(35.71%). The final bounded-grid replay is a separate spot check, not a matched
timing arm: every selected large domain still launches 4096 blocks, and all
nine requested raw J/K matrices agree with baseline within **2.680e-12**.
Unused generic LR-J ABI buffers are not comparison channels.

Order three alone regresses from **134.210102 to 144.350219 seconds** in that
paired diagnostic; orders four/five carry the aggregate improvement. Further
order-three class/primitive/sparsity scheduling needs its own measured gate.
Baseline outer counts are AO quartets and candidate counts are shell quartets;
their ratio is **not a same-unit scientific work reduction**.

**Complete 96-atom cold E+F timing and independent physical acceptance remain
null.** Other source classes are deliberately omitted in the selected
diagnostic, which exits before publishing a physical E/F result.

## Qualification and retention

- Four native CUDA suites pass, including screening, positive J/K, internal
  full/SR/LR, RHF/UHF, spherical projection and provider/ledger/MD-J fallbacks.
- Eight independent Libcint cases pass: Cartesian/spherical, RHF/UHF and two
  bases. Public ranged requests are K-only; ranged public J is not added.
- Complete low-order shell compositions, largest pppp/dppp domains, slot tails,
  signed/coincident/empty/repeated-pair cases and exact primitive work pass
  against retained component and independent host-orbit oracles. Legacy
  order-5-through-12 and high-order qualification remain intact.
- Memcheck, initcheck, synccheck and racecheck pass with zero errors/hazards,
  forcing one persistent CTA and retaining actual order-3/4/5 launches.
- 33 host Direct generation/dispatch tests pass. The integrated scientific
  compiler inventory has 512 modules and zero dependency errors.

The compact receipt is gzip-compressed deterministic JSON. Raw journals,
matrices, scripts, cache statistics, compiler commands, exact patches and
library/source hashes stay under the existing local root
`/data/jzzeng/wb97m-canonical-low-orders-20261011`; no Release or external
publication is created. Initial diagnostic include-inventory/static-cudart
preflight failures are retained alongside their corrected executions.

A final dispatch-comment clarification changes the embedded source identity,
not scientific code: of 489 compiled objects, only the build-identity metadata
object changes; every other object matches byte-for-byte. The receipt keeps
both library identities, the object bridge and the submitted source manifest.
Four native suites and independent complete three/twelve-atom physical gates
pass again on the submitted library. These remain qualification samples, not
matched throughput arms.
The local `raw-evidence.tar.gz` contains inputs and raw results, excluding the
generated compact receipt to avoid a checksum cycle. Its SHA256, byte size,
member count and gzip-integrity gate are recorded in the compact receipt.

The [decision note](../../../.agents/notes/implemented/performance/2026-10-11-canonical-small-cta-scheduling.md)
records the rejected active-rectangle initialization and width alternatives.
The [current contract](../../../docs/developer/direct_pair_recurrence.md)
defines production ownership, defaults and bounded fallback semantics.
