# Decision: do not promote register-specialized component Rys axes

Status: rejected after balanced A100 and5090 complete endpoint measurements
Date: 2026-11-08

## Problem

The portable value-only DPPP/PPPP component worker reconstructs an addressed
TRR table for every axis, root, primitive and component. Its noinline volatile
array helper introduces local-memory traffic even when a component requests
only a small dependency closure. The HF96 prepared diagnostic228328 attributes
1.935/1.705s to the DPPP/PPPP streaming kernels. These are collected kernel
sums, not an amount that can be subtracted from complete cold latency.

The existing independently qualified SM90 shared-axis companion does not own
the A100 streaming-K route. Its architecture and ownership restrictions must
not be silently removed or its measurements transferred to another path.

## Candidate

For the two value-only angular signatures, emit a switch over legal one-axis
angular states. Each branch reuses `build_rys_axis_program` and the existing
scalar recurrence-instruction lowering. Only the requested state's dependency
closure executes; all intermediate indices are static. This is one algebra
implementation with a different storage/scheduling owner, not handwritten ERI.

Force-support, shared-axis, other shell signatures and their fallbacks remain
unchanged. There is no runtime GPU-product test, relaxed convergence/screening,
resident cache, extra storage budget, CPU production work or new opt-in. Branch
divergence/register-pressure/code-size effects are reasons to measure, not
assumed improvements. Unrestricted instantiations require independent coverage
before shipping, even though their scatter owner is unchanged.

## Evidence and acceptance

Campaign `.artifacts/hf-register-axis-v1-20261108/` freezes clean committed
control source50ca688e7f9dc3d51c05f9aa336fab14226372fcbe8f970536062753a87ff6e7
and candidate480ed4bcbc19030cdb1acf980efa96ec32aef2ce0c1a7d667f5155c9ab154759.
Unrelated adaptive-delta dirty paths are explicitly excluded. Both images must
be relinked from these exact closures; no old timing is a new-image control.

- 103 direct-value tests, including independent contracted Libcint comparisons,
  pass;41 shared-XY/weighted-Z tests pass.
- Compiler structure:587 modules, zero dependency errors; ruff lint passes.
- Formatter requests only grouping of the assembled source string. The frozen
 pilot intentionally retains its recorded bytes; any qualified final change
 must reconcile formatting and native identity explicitly, not relabel an ELF.
- First compilation stopped at identity audit because a reused owned source
 tree retained files deleted by current master. Preserve its failed receipt.
 The repair verifies each removed file against the original archive and
 restores the original owned tree and runtime at process exit.

## Endpoint decision

Control ELF2977049443dd116779ad583f67da20c902fcede2fdb0ae721d2995d8aee147ff
and candidate64e4b3e68e6abd4286933facf523c916501e2bd77152711a0aeeb1eb034c9305
were independently linked and authenticated. CPU Slurm16777 restores the
borrowed source/runtime;16778 validates both complete delta reconstructions.
Finite Slurm5090 job16779 and exclusive GPU22 LSF228682 both exit0; CPU setup
228680 also succeeds. Each GPU measures both sizes in four-population ABBA
order, with complete cold energy+analytic forces+host export and three warm
calls per population. No new reference solve or CPU oracle runs in production.

|Card/case|Control median cold/s|Candidate median cold/s|Reduction|
|---|---:|---:|---:|
|A100 HF96|16.153318|15.939340|1.3247%|
|A100 HF48|6.098310|6.039661|0.9617%|
|5090 HF96|16.903445|16.847528|0.3308%|
|5090 HF48|6.286097|6.284813|0.0204%|

These tiny point changes do not close the previously measured GPU4PySCF gap;
there is no new same-allocation GPU4 timing and no all64 recount. Reject the
candidate rather than shipping an unhelpful opt-in/default. The three owned
production/test edits are removed; archived candidate sources remain exact.
The earlier adaptive-delta experiment is neither changed nor promoted here.

All120 native sample pairs per case/card pass E1e-8/F1e-7, including every
cold/warm sample, not a selected subset. Maxima are A10096 about4.55e-12/
6.46e-13 and48 about2.28e-12/3.95e-13. Every balanced pair's three warm guards
passes3% on both cards. Cold work remains15 target iterations/Focks for96,
10 for48; each warm has one target Fock. Quartet work counters remain invalid,
not zero. Both96 cohorts additionally pass the authenticated independent
tighter GPU4 oracle228320, max E~1.4e-9/F~1.55e-9. No independent48,
changed-geometry, unrestricted, or final-default qualification is asserted.

CPU-only resource audit16780 binds the actual two ELFs. For SM80 RHF streaming
DPPP, reported registers rise96 to130 and stack96B to0; PPPP registers80 to128
and stack136B to0; shared stays680B. LOCAL0 is a static descriptor, not proof
of zero linked spill/local traffic or actual occupancy. Reduced stack storage
does not establish a whole-endpoint win. Kernel/callee dynamic attribution is
still needed before blaming a particular axis/root routine.

Original A100 archive SHA306fd25dc6f49742a74f9030f4dde332a8e782096458a021499b039622d3bd56;
5090 SHA bfaec411fa7b45a3bd272a21dfbd3d4e8fbed830747b9f5dddd2a47eef9123f1.
Allraw populations, source/ELF checks, analysis failures and submission receipts
remain in the campaign. Metadata-only acquisition first failed on node-local
LSF-log globs; a multi-source SCP waited at another password prompt. Preserve
those receipts, acquire remaining metadata as one new archive, and never
repeat the already successful GPU jobs for a transport failure.

## References

- Issue2976 comment6714615009 records the hypothesis and host gates.
- The prior public finalized-DF guess and fixed/coarse delta rejection notes
 remain valid and are not superseded by this exact compiler experiment.
