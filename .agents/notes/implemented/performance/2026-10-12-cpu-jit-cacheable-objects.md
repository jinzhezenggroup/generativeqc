# Decision: cache CPU JIT translation units before linking

Status: implemented
Date: 2026-10-12

## Problem

The CPU adapter sent `c++ -shared source.cpp ...` to the verified compiler
launcher. This invokes the launcher but cannot cache compilation: ccache reports
`called_for_link`. The completed
[force-shard job](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38258968631/job/114844759085)
reported 166/166 uncacheable calls despite passing its tests.

## Decision and boundaries

For normal C++ source suffixes and an explicitly phase-safe argument subset,
compile each original translation unit with `-c`, then link the ordered objects
and libraries with the same driver and flags. The admitted optional arguments
are the runtime's `-ffp-contract=off` and joined/separate `-I`, `-D`, `-U` forms.
Do not turn this into an approximate general-purpose compiler argument parser.

Opaque arguments, response files, language switches, extra inputs (including
Path operands), and side-output options retain the exact previous combined
command and output location. This fallback retains its previous output/failure
semantics and is not claimed cacheable.

The split path verifies its launcher once and shares one elapsed-time budget
across discovery, all object compilations, and the final link. Every subprocess
still uses the existing finite process-group owner, including private sccache
workers. Object and final-library temporaries belong exclusively to this call.
Only a successful complete library is atomically published; compilation, link,
deadline, or publication failures preserve an existing caller-owned output.
Output-parent failures remain failed CompileResult outcomes.

The sources, include order, library order, compiler/provider identity, generated
mathematics, and runtime artifact keys are unchanged. Binary byte identity is not
a general promise of changing compiler execution phases; existing artifact
integrity checks still authenticate the produced binary.

## Source-path correctness

A formerly uncacheable combined ccache call forwards the original source paths.
A cacheable object call can instead rewrite them through CCACHE_BASEDIR, changing
`__FILE__` in sources and included headers. The split CPU path opts into a copied
child environment with an empty CCACHE_BASEDIR, preserving the original path
spelling. No global environment, persistent cache configuration, other compiler
caller, or shared sccache server is changed. No sloppiness, source rewriting, or
invented provenance is used to obtain hits.

## Evidence and limits

Executable tests cover repeated real ccache object hits, changed headers/macros,
ordered static libraries, multi-TU execution, exact path-macro parity, opaque
language/response/side-output/Path operands, early and later compilation failure,
link and publication failure, discovery and cumulative timeout, descendant
cleanup, and survival of unrelated work. Existing runtime identity/integrity and
Tensor/CC numerical tests also qualify the path.

A bounded local workload executes a two-program Tensor bundle, CC Lambda
transpose, Hamiltonian pullback, triples VJP, and a four-program Lambda bundle.
With stable generated-source roots and separately retained finished artifacts,
three final warm repeats each report 9 direct object hits and 5 expected
uncacheable final links. The old combined path reports only 5 called-for-link
results. The complete fixture medians are 17.845 s before and 11.986 s after;
this includes lowering, independent interpreter comparisons, and execution.
All five artifact keys/identities and these five fixture binary hashes match.
These are local fixture observations, not a hosted CI speedup or scientific CPU
execution performance claim; cold compilation still pays compile/link work.

Fresh source-root trials correctly miss all 9 object entries. Moving source paths
can affect genuine compiler inputs, so no cross-directory cache-hit promise is
made. Restore the object cache alongside authentic stable generated-source paths;
do not change source identity merely to obtain reuse.

Real ccache 4.14.1 was exercised without clearing its shared cache. The installed
sccache 0.16.0 passed version discovery but could not start its Unix socket in the
local sandbox (Operation not permitted). Existing mocked process-group/lease
lifecycle tests pass; this is not real sccache execution qualification.

## Revisit when

Extend the admitted option subset only with executable phase-equivalence tests.
A future private multi-command sccache session may amortize server startup, but
must preserve one aggregate deadline, exclusive leases, and unrelated owners.
