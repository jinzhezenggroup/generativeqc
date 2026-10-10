# Decision: automatically admit indexed canonical order-five J/K reuse

Status: implemented
Date: 2026-10-10

## Problem

The [initial investigation](../../proposed/2026-10-10-wb97mv-tzvpd-cold96-source-reuse.md)
identified repeated primitive geometry, Hermite and Coulomb preparation across
Cartesian components. The [qualification implementation](2026-10-10-indexed-canonical-pair-materialization.md)
removed that repetition, but required the legacy HF materialization switch.
That opt-in is neither an appropriate public policy nor a safe global default:
the older dense HF/through-f schedule has retained negative endpoint evidence.

## Decision

Automatically prepare and use the indexed canonical order-five consumer when
its immutable primitive-pair cache, canonical index and remaining provider
budget are available. There is no new environment variable, public option or
method/basis/atom-count whitelist. An unset legacy HF switch, and explicitly
setting that switch to zero, both admit this canonical route. The legacy HF
and native dddd opt-ins are unchanged.

This supersedes only the qualification-only/default-disabled decision in the
previous note. Its historical source identities, opt-in measurements and
incomplete larger oracle/endpoints remain unchanged.

## Resource priority and fallback

Prepare incumbent generated/canonical owners and optional MD-J storage before
the new shell index and bit-preserving bounds transpose. SPD full-range J can
use MD-J while range K uses the canonical source; allowing optional K metadata
to take the last available bytes first would evict the already-established J
acceleration. Automatic admission must not introduce that budget cliff.

The complete additional index, keys, bounds view and sort/scan workspace are
charged, including preparation scratch. Budget rejection and runtime allocation
failure roll back only this optional lease. They cannot discard its borrowed
geometry cache, incumbent matrices, MD-J owner or canonical source. Execution
does not allocate or reread policy. Fixed screening, compensation, resident
ERI replay, other orders, combined full-J/range-K and paired RSH keep their
existing consumers.

## Scientific invariants

- Share one primitive-pair recurrence across all components of a shell quartet;
  order five has at most 162 components, fitting one 256-lane packet.
- Preserve the original component-level Schwarz product test. Shell maxima
  provide only a conservative outer index, never replacement scientific gates.
- Transpose original bounds bit-for-bit for the generated helper's layout;
  separately computed reciprocal bounds need not be bitwise equal.
- Publish independent positive J/K through the authoritative generated scatter.
  Preserve HF defaults, full/SR/LR radial identity, RHF/UHF density semantics,
  public spherical projection and every lane's publication/retirement barriers.
- Do not divide cached coefficient products to reconstruct primitive inputs,
  relax thresholds, reduce the basis/grid, or introduce production CPU/oracle
  work. Forces and mixed-precision policy are unchanged.

## Evidence and limits

The [source-matched default qualification](../../../../benchmarks/results/wb97mv-canonical-default-20261010/README.md)
records the latest-master base, binary identities, complete matched cold E+F
timing and semantic counts. Native gates cover absent/zero legacy selectors,
independent CPU full/SR/LR matrices, RHF/UHF masks, signed/diffuse/screened
contractions, public Cartesian/spherical layouts, batches and moved geometry.
Exact-budget and real-ledger allocation-denial fixtures protect rollback and
incumbent MD-J priority. Sanitizers exercise persistent-CTA task reuse.

On the source-identical fetched master, the complete 12-atom cold pair improves
582.201867 -> 531.607555 seconds (8.69%), with 21 iterations/21 Focks on both
arms. Physical forces take 38.585711/39.080256 seconds; the reduction is in SCF,
not force cache setup. Native/native E/F differences are 1.137e-13 Eh /
5.234e-11 Eh/Bohr. Three independent-gated 3-atom repetitions per arm improve
the complete median 12.749601 -> 12.508062 seconds (1.89%), all with 15
iterations/15 Focks. First-use samples are retained separately, not attributed
as a value-kernel win. Default tracing records 90 materialized launches without
the legacy switch. All four sanitizers report zero errors/hazards.

The historical selected 96-atom result is not a complete physical E+F result.
The unchanged independent 12-atom reference did not converge in 100 iterations;
native/native larger comparisons are consistency checks, not independent oracle
passes. Small complete independent E/F and source-level matrix gates therefore
remain explicit, rather than substituting a native self-comparison as an oracle.

## Rejected alternatives

- Set the legacy HF switch globally to one: that promotes a different, dense
  traversal with negative endpoint evidence.
- Add a new canonical opt-in or a WB97M-V/96-atom special case: capability and
  resource admission already determine legality, including other methods and
  system sizes.
- Admit this lease before MD-J: optional K reuse must not evict established J.
- Raise the recurrence order without adding complete multi-packet ownership:
  orders six and above can exceed one packet and silently omit components.

## Next route

The [bounded multi-packet decision](2026-10-10-bounded-canonical-multi-packet-reuse.md)
extends this admission's angular domain to orders six/seven while retaining its
original invariants and fallbacks; the order-five measurements here remain
historical evidence of this specific source revision.

First qualify shared full/LR materialized recurrence with independent positive
source matrices and unchanged masks. Then consider orders six/seven with
explicit multi-packet ownership, recurrence/work counters and register/resource
gates. Require complete matched endpoints, moved/batched/tight-budget behavior
and independent numerical acceptance before promoting those additional routes.
