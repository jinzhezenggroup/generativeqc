# Decision: append RHF response controls after endpoint DIIS history

Status: implemented
Date: 2026-10-04

## Problem

PR #1832 added optional orbital-screening, J/K-profile and nuclear-response
selectors starting at positional argument eight. Master independently assigned
argument eight to CCSD DIIS history. A merge that simply selected either layout
would silently alter the meaning of a valid command from the other branch.

## Decision

Preserve master's existing argument-eight `DIIS_HISTORY` contract (default six;
zero or two through twenty), then append the three response controls at positions
nine through eleven. Update the live extracted endpoint fixture and current-state
documentation. Historical benchmark receipts and their source-bound commands stay
byte-exact and describe their original revisions.

Require a complete integer parse for the DIIS token. In particular, `0.0`,
`0e-12` and `2e-12` cannot silently become zero or two. An integer `0` in argument
eight always disables DIIS; it is not inferred to mean zero screening. Existing
master calls do not need changes. Response-branch calls must insert an explicit
DIIS history before their response controls.

## Rejected alternatives

- Guessing an argument's meaning from its numeric value or argument count:
  integer zero has two plausible interpretations and optional suffixes overlap
- Moving DIIS after response controls: breaks the established master CLI
- Editing frozen receipts to the new syntax: misrepresents the actual run

## Invariants and evidence

The matrix/Lambda defaults remain enabled, orbital screening defaults to zero,
J/K profiling defaults off and the nuclear schedule defaults to two-pass
symmetric polarization. The numerical production response implementation is
unchanged by this resolution. The host fixture in
`tests/python/test_df_lambda_default_policy.py` compiles the actual endpoint
selector code and descriptor assignment. It covers every valid DIIS history,
partial optional suffixes, all three nuclear selectors, default/explicit scalar
controls, malformed selectors, old fractional screening tokens and arity errors.

## Revisit when

A separately designed named-option CLI can provide an explicit versioned
migration. Do not add heuristic positional compatibility.

## Later landed Q-batch composition

Master subsequently assigned argument nine to `CCSD_Q_BATCH_LIMIT` (default
eight). The integrated endpoint preserves DIIS at eight and CCSD batch at nine,
and appends response controls at ten through twelve. Both batch tokens and DIIS
require complete unsigned integers; screening requires a complete finite
nonnegative numeric token. No numeric-value or argument-count guessing is used.
The function signature likewise retains the CCSD batch selector before the
response options. The live extracted-signature fixture protects defaults, all
DIIS histories, explicit batches, partial suffixes and malformed numeric inputs.
Frozen source-bound receipts are unchanged. This extends the compatibility
decision above; the nine-through-eleven layout described there was the preceding
branch revision.
