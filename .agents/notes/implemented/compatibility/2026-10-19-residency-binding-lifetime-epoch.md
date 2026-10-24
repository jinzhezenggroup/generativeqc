# Decision: distinguish canonical observer binding lifetimes

Status: implemented
Date: 2026-10-19

## Problem

The private synchronous residency observer previously identified its owner by a
callback/context pair. Detaching and rebinding the same pair can occur between
source begin/end records without changing either address. A future diagnostic
phase consumer cannot infer uninterrupted binding from those addresses alone.
Existing graph/source generation IDs describe different lifetimes and must not
be repurposed for this check.

## Decision

Maintain a separate thread-local, nonwrapping binding epoch. Only successful
canonical bind/unbind operations advance it. Failed replacement, foreign-context
detach, null callback/error destinations and mutation during dispatch leave it
unchanged. A non-null callback may still use a null context. The
observer object layout, callback signature, existing scalar layouts and graph
identity remain unchanged.

The epoch saturates instead of wrapping. A sticky exhaustion state and the
admission helper make saturation unusable for new diagnostic lifetime proofs.
Legacy bind/unbind, callback delivery and error accounting continue to operate;
epoch exhaustion does not introduce a new scientific or legacy API failure.

This provides a primitive for a separately opted-in CC diagnostic. It does not
itself add CC phase emission, collector registration or complete source coverage.
A phase consumer must pin the admitted epoch, reject changes and preserve an
incomplete result when its original binding disappears. It must not retain a
borrowed callback/context/code pointer and later invoke it after detach.

## Compatibility and proof domain

The epoch proves only mutations through the same new native's canonical binding
implementation. Old inline clients or mixed old/new weak definitions and DSOs
can bypass the new counter. They are outside this admission domain; this change
does not retroactively qualify installed historical binaries. An actual capture
must establish its library, registration path and collector identity.

The extra TLS object and stores on successful binding operations have a cost.
There are no added explicit heap, clock, I/O or CUDA operations in these helpers;
that is not a claim of zero whole-library or TLS first-touch overhead.

## Evidence

Five executable CPU cases compile separate bridge, emitter and external-client
translation units. They cover successful versus failed mutation, same-address
detach/rebind, submitting-thread isolation, reentrant rejection and callback
exception shielding, and saturation with continuing legacy delivery. The tests
assert behavior without production-body hashes or spelling constraints.

An isolated CPU prototype additionally exercises the actual canonical API-body
delegation in a minimal shared library with an external client. Explicit CC
permit candidates reject ABA and changed owners. Those prototype results are
not an installed scientific build, NVTX losslessness, device qualification or
complete endpoint attribution. Historical energy/capture evidence keeps its
original source/library identity and limitations.

## Revisit when

A diagnostic needs other-thread coverage, supports old inline consumers, changes
the canonical binding path, or introduces a separate lifecycle registration ABI.
Each requires explicit lifetime/identity qualification rather than assuming this
thread-local epoch covers it.
