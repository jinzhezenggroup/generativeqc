# Preserve explicit force requests in schedule trials

Status: implemented; source-rendering compatibility only
Date: 2026-10-16

## Problem

The explicit force request repair moved the force policy to shared scalar
bindings after the ordinary/composite resource split. The existing trial
renderer expected an ordinary-branch tuple assignment and rejected the real
repository source. The required core-c contract failed; retrying CI or removing
that contract would not repair the incompatible transformation.

## Decision

Authenticate the current keyword request and adjacent shared scalar bindings,
then change the trial default only for an ordinary call with no explicit tile.
Both policy and execution use that same guarded value. Composite requests and
explicit caller values retain their original policy. Legacy literal tuple
bindings remain supported separately; unknown, duplicate or ambiguous bindings
are rejected. Other resource/point trial controls are unchanged.

## Evidence and limitations

The actual repository contract and six executable combinations of composite /
ordinary with absent / explicit requests fail before repair. Host tests cover
the actual source, both supported forms, execution/policy agreement, preserved
composite behavior and rejection of unknown bindings. This is a source-rendering
repair; it does not qualify GPU work, timings, dynamic traffic or a new SDK.

Refs #2072 and PR #2250. The production request repair is documented in
`../performance/2026-10-16-fixed-force-request-qualification.md`.
