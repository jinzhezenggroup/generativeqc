# Decision: keep method capability resolution on the selected native library

Status: implemented
Date: 2026-10-16

## Problem

Calculator first selected a library for its requested backend/device/profile,
then called public cached `method_capabilities(name)`. That independent discovery
loaded the default library or reused a prior registry result. Consequently the
capability view could describe a different native artifact, and already-resolved
compiler compositions were resolved a second time only to rediscover a carrier.
Successful global discovery is not provenance for a selected execution owner.

## Decision

Share one thin registry presentation helper. Public discovery retains its cached,
backend-neutral contract. A Calculator instead supplies its already-selected
library, public display name and preflight carrier ID directly. The helper uses
the existing installed `generativeqc_method_from_name` owner to verify the
canonical native carrier and the existing capability getter to obtain metadata.
It refuses selector/record inconsistencies and propagates selected native errors
without a default-library, cached-result or Python-table retry.

Keep numeric-ID compatibility only when the older library lacks the optional
named-selector ABI. Python alias/composition preflight still owns public syntax;
the native carrier is not permission to execute an arbitrary MethodIR or broader
backend/property domain. Automatic Libxc retains its energy-only registry mask,
and existing context/basis/precision/DF/correction/second-order admission remains
unchanged. No new native registry, ABI or scientific formula is introduced.

## Rejected alternatives

Invalidating the public cache on every Calculator would not bind discovery to the
chosen library and would weaken a useful public contract. Loading another default
library after a selected-native failure would silently change execution provenance.
Moving loader selection ahead of Python metadata validation would also change
early-error/device-initialization behavior. None is necessary to reuse the existing
native selector and registry owners at the established load boundary.

## Validation boundary

Executable tests distinguish selected/native registry publications from warmed
public discovery; exercise every public built-in alias against the actual compiled
registry; retain old numeric compatibility; and inject selector/query errors and
malformed ABI/method/family publications without alternate discovery. These do not
pin function bodies, AST shapes or exact internal call counts. Matching-source
library/SDK, complete endpoint and independent molecular qualification are still
required for production acceptance; this scoped ownership change does not complete
#2227 R5/R6 or remove the Python compiler/explicit model resolver.
