# Frozen Johnson migration oracle

This test-only fixture freezes the complete CPU common-mixer consumer at
`3b97c234eb18f5e6f354842b458a4e56c1619f95`, before shared Johnson ownership.
Numerical sources listed in `manifest.json` were copied byte-for-byte using
`git show`. The status-only `runtime/types.hpp` shim retains the exact int32
typedef and enum from that header; its original and excerpt hashes are recorded.
The four generated CPU fragments were emitted by the Python compiler and
`tools/generate_ordered_history_native.py` extracted with `git archive` from
that same commit, never by the current compiler. SHA-256 hashes guard the
snapshot. No installed xTB headers, current production mixer, or current
code generation is used to build this oracle.

The Python differential test runs identical ragged multi-field inputs and
failure injections through separately loaded legacy and shared-owner
libraries, comparing every exposed value and metadata field and the exact
failure diagnostic. It also checks every persistent byte and storage offset,
peer-local batch failure, initialization and binding rejection, selective
restart, counter exhaustion, and recovery. The independent NumPy chronological
oracle continues to check the algorithm separately from this migration
compatibility gate.

The shared owner intentionally tightens invalid admission for unaligned FP64
fields and descriptors placed inside their own numerical storage. Those
legacy-accepted malformed cases are checked by the standalone method-free
owner contract test, together with transaction rollback and allocation checks;
they are not requirements to reproduce unsafe legacy behavior.

Do not refresh this fixture to make a changed implementation pass. An
intentional numerical or contract change needs its own explicit acceptance
criteria and independently reviewed reference update.
