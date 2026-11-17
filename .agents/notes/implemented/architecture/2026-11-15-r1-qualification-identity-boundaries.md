# Decision: requalify native force transport across provenance changes

Status: implemented
Date: 2026-11-15

## Problem

Publishing the already implemented size-tagged KS transport descriptors changes
an installed ABI dependency. The existing Libxc endpoint catalog correctly
expires its prior qualification even though no force equation changes. During
R1 qualification, the stacked base also gained 42 transitive generator inputs
and stronger safety-receipt authentication. An earlier successful rerun does
not authenticate an input missing from its embedded native census.

## Decision

Keep dependency invalidation conservative. Preserve the original campaigns,
source inventories, exact previous summaries and negative observations. Rebuild
with the expanded generator inventory and rerun independent domain, CPU and
CUDA acceptance rather than adding missing historical hashes from the current
checkout or acknowledging a scientific demotion to make CI green.

Run resource/lifetime checks in the same finite Slurm allocation and on the
same build ELF as their independent campaign. The receipt must name an actually
measured force context, retain both minimum/below-budget spin cases and ragged
recovery, and report the actual nonconvergence outcome. A hash authenticates
bytes, not the truth of a success assertion.

Distinguish a build ELF from its installed ELF. Installation can rewrite RPATH
and therefore change SHA-256 while preserving the native source identity.
Installed SDK/CLI/Python parity receipts bind the installed ELF; the official
catalog and build-linked native tests bind the build ELF. Never describe those
different hashes as identical binaries or use one as evidence for the other.

Keep paired endpoint timing isolated from qualification and profiling. Frozen
frontends, source patches, binary hashes and runner hashes identify each actual
cohort. Later documentation or provenance-only changes are not new timing
measurements. Explicitly audit their input delta before reusing a scientific
non-regression result; do not silently relabel an older cohort as a new-head run.

## Rejected alternatives and invariants

Do not exempt public headers or unknown generators from dependency checks.
Do not synthesize a matching Slurm environment, change the nonconvergence
functional name in a receipt, or borrow a separately measured safety job.
Do not treat equal source identities as equal ELF identities. No force equation,
precision policy, native capability bit, physical-domain boundary or numerical
tolerance changes as part of this provenance repair.

Do not edit a running campaign shell script. Bash may already have parsed its
functions and keep executing an old checkout selection even when the on-disk
script now names frozen inputs. Restart from immutable frontend copies and
record each block's actual working directory and before/after source checks.

## Evidence

Source-bound campaigns and complete endpoint samples are retained under
`benchmarks/results/r1-native-dft-forces-20261115/`. The generated qualification
report remains the current authority; this note is rationale, not admission.

## Storage boundary

The existing aggregate benchmark budget had only 114,146 bytes of headroom,
insufficient even for a fresh independently authenticated catalog campaign.
Do not raise that budget, drop paired samples or move numerical observations
behind a network fetch. Two GFN2 whole-checkout hash streams consumed 1,259,176
bytes and differed in only three lines. A lossless three-line delta would save
628,968 bytes, but that still leaves insufficient room for complete positive
and negative R1 timing populations. Keep their original compressed/decoded
identities and line counts in an authenticated snapshot and recover the exact
streams from the existing published ancestor with the standard restoration
tool. Both dirty source patches and all 120 independent numerical observations
remain offline. Normal numerical replay checks the report/source bindings and
explicitly reports that the whole census was not loaded; the opt-in historical
census audit additionally checks original compressed/decoded bytes. This is the
existing separation of scientific replay from bulky provenance auditing, not a
new codec, GFN2 requalification, relaxed numerical gate or storage-budget waiver.

## Squash integration

The parent PR merged as `c99e6543e1fe0f4ea6832d58381d30e002e04c67` while
R1 CI was running. Its complete tree equals the measured parent
`df42f93bbace0f16fb5b3dec77552384c9da9101`, with tree identity
`3cbd9c6234725daabd1c373c550bd07dac2b83ab`. Rebase only the R1 commit onto
that identical tree rather than duplicating the already merged parent history.
All production inputs remain byte-identical; this is not a new timed campaign.

Keep original measurement base revisions and patches. Add a separately named
reconstruction base for the two affected source snapshots, pointing at the
published squash ancestor. Verify the exact 15,504-file immutable frontend from
that ancestor and retain the new source-recovery receipt. Do not rewrite original
run headers or rely on an obsolete PR-head object being fetched by a fresh clone.

## Revisit when

A future qualification schema explicitly binds independently scheduled safety
receipts, or native build and install identity cease to differ. Such a change
needs executable validation and must not rewrite the observations retained here.
