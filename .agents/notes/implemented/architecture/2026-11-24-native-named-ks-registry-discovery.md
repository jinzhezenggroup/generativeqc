# Decision: Query native named KS registry carriers directly

Status: implemented
Date: 2026-11-24

The backend-neutral Python registry query previously reconstructed a compiler
graph even when the installed library already owned the exact named program.
Publish the existing public registry projection alongside the preparation
carrier. Query it without preparing a context or invoking the compiler resolver.
Keep selection and capability lookup on the same loaded library; a native error
or malformed carrier cannot trigger an alternate lookup.

Corrected programs can use a different registry row from the preparation row:
PBE-D4's full-graph predicate, r2SCAN-D4's ingredients and the basis-bound
r2SCAN-3c composite must preserve their previous public projections. Reuse the
existing graph predicate and electronic ingredient mapping at generation time,
not a second table of corrected method names. A basis-bound composite is only
advertised under its manifest spellings, not a Libxc electronic spelling.
Zero selection leaves other compiler-owned selectors on their explicit routes.
Automatic Libxc and contextual `Calculator` admission are unchanged. No
registry row proves endpoint support.

The parity test enumerates the generated named catalog and compares its
represented selections to the previous compiler projection;
failure injection proves a native hit or error cannot fall through to that
projection. Refs #2227.
