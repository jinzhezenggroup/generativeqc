# Decision: retain the spent PBE0 pilot and separate input identity from product closure

Status: implemented
Date: 2026-10-26

The first frozen-source H100 Direct PBE0 water48 pilot consumed one Standard/cold
scientific request and returned no complete E/F. Fast and GPU4PySCF were not run;
the finite stage is spent, with no automatic retry or unused two-call balance.
The retained source was `cbd4d8074de35105717c8aac7cd1f86babbe90ec`, not the
publication checkout's current master. This finding does not establish a current
master numerical defect or an RTX 5090 quota issue.

The source/environment input readbacks and physical device identity matched
before/after. The generated force consumer nevertheless required a separately
packaged `pbe0_rks_spd` DSO/manifest next to the core library. The loader failed
before a complete return; a subsequent two-path read-only stat found both exact
pathnames absent. The archived HF build requested only the core `generativeqc`
target. CMake defines the Python stationary companion and its manifest as
independent products. Core/native profile generation and a successful whole-file
input identity are therefore insufficient evidence that the selected Python force
product was produced and distributed.

The publication preserves the failure and source/package boundary without changing
runtime, loader, SDK, gates, cache identities or measured source. The original
full85 publication remains in Git history and frozen local backups. This
publication restores76 collected terminal originals plus eight exact supplements,
omitting only the9,451,229-byte
whole-environment manifest as explicitly named identity-only metadata. Its SHA,
mode, domains, counts, actual deployed consumption and equal recorded before/after
receipts remain bound; the full census cannot be recovered by the current reader.
The historical preflight consumed the full list. Offline replay of these retained
receipts is not a new whole-environment validation. All other84 raw originals,
including scientific request inputs, failure logs and exact companion-path stat,
remain byte-identical. The two census XZ fragments and their unused storage
descriptor are absent from the current public tree. Cache binaries
and compressed transport envelopes retain their separate recovery limits.

Rejected interpretations include a silent JIT fallback for a declared missing
package, renaming/copying old native-cache artifacts into the companion locator,
inferring numerical/precision/OOM failure, or filling absent E/F/time/work with0.
The artifact's semantic key and matching binary/plan/target/weight provenance are
unavailable because no manifest was loaded. A future engineering or scientific
stage requires a separate source-matched product-closure disposition and finite
authorization; this evidence PR supplies neither.

See [the retained report](../../../benchmarks/results/h100-pbe0-direct-df-20261024/README.md),
[issue #2731](https://github.com/jinzhezenggroup/generativeqc/issues/2731), and
[the production-owner request](https://github.com/jinzhezenggroup/generativeqc/issues/2731#issuecomment-6408960658).
The parent scientific scope, DF, water96, formal statistics and delivery remain
incomplete. This note records a publication boundary, not an implemented runtime fix.
