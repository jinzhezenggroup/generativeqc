# Decision: eliminate primitive-local dynamic Jets from packaged ERI channels

Status: implemented
Date: 2026-11-13

## Problem and measured diagnosis

After positive Full moments were qualified for packaged stationary CPU ERIs,
the actual eight-million-record interrupted integral-gradient apply took
7.122436179 seconds, of which 3.535497363 seconds belonged to the packaged
leaves. This establishes substantial consumer overhead, not its cause.

A bounded glibc malloc interposer on the calling thread observes the same
actual consumer, original saved C->D, exact original 81-AO/six-atom basis/ECP,
RKS/exchange0.2, explicit zero W and exactly eight million completed ERI
primitive records. The work histogram and sampled inputs match the previous
production profile. After the first ERI, it observes 56,177,483 successful malloc
calls requesting 5,387,555,103 bytes in aggregate, including 56,073,500 calls
of 96 bytes. Before the first ERI there are 933,758 calls/98,440,544 bytes and
only33 96-byte calls. These are allocation traffic, not live or peak memory.
The counter observes malloc, not every allocation API; its interposed timing
is not production performance evidence.

The private packaged helper wrapped every fixed twelve-slot derivative in a
dynamic Jet. Scalar multiplication then constructed a constant Jet and a
product Jet; each channel addition constructed another Jet. This repeats
twelve-double vector allocation for every primitive and active channel even
though neither derivative algebra nor dynamic coordinate dimension is needed.
The native resource bound was finite but did not prevent this allocation work.

## Decision and ownership

Only `src/integrals/s_integrals.cpp` changes production behavior. The packaged
helper restores original shell/axis slots into a fixed twelve-double array,
including the existing through-f generic primitive owner. The shared channel
loop retains normalization, coefficients, channel/primitive iteration order,
zero-weight handling and semantic work counters. Packaged accumulation writes
the already-sized channel Jet directly through the existing compiler-generated
scalar-AD `Multiply` and `Add` operations. No new scientific algebra or scalar
chain rule is authored. The explicit reference recurrence and its original
dynamic-Jet arithmetic remain intact.

The former packaged Jet had a zero primal, not an integral value. Preserve that
zero primal and the generated multiplication's zero left tangent, rather than
simplifying the operation to a bare multiply/add. Removing a nominally zero
term changes signed-zero/exception behavior, and removing the zero-primal
finiteness check can silently admit overflowing coefficient products at an
all-zero physical gradient. Permanent coverage explicitly requires rejection
without publishing a partial work count in that case.

This changes temporary storage/lifetime, not FP64 precision, recurrence work,
an artifact ABI, the generated primitive dispatcher or compiler source.
Conservative resource allowances and explicit reference fallback/unsupported
class refusal are unchanged. There is no shared numeric cache, identity lease,
optional storage precondition or global allocator change in production.

## Independent qualification

Permanent native controls add multi-primitive signed contracted channels through
d/f shells, Cartesian and spherical representations, separated/coincident and
diffuse/tight geometry, zero-channel isolation and transactional overflow. The
existing native independent recurrence, finite-difference, full assembled
stationary gradient, explicit unsupported-class and resource gates remain.

A source-bound probe executes36 complete shell contractions (six angular
classes through two f shells, two representations, three geometries, two signed
primitives per shell, three channels). Both packaged and reference outputs,
including all output bits and semantic work, match the frozen old core exactly.
Maximum packaged-versus-independent-recurrence error is
2.6645352591003757e-15, within the unchanged2e-10 absolute/relative gate. No
native/reference SCF runs in these component probes. The stationary-prefix
diagnostics use zero W only for the ERI component; they publish no full gradient.

The baseline is source
`dcbd294e1f6f1ef08869582c859a75f8e16e0e7ae111b0b5a55c659233fb9515`,
core SHA256 `5741816b45359b70793ad3295a488d8e9010e6891080f414c58929a0fd3cf536`.
The production change binds source
`b3a5ae19e3bd83c41f382c988d097fe3a9913798018fe7f502002d9aa4a8856a`,
core SHA256 `057ac20ee7fc8dddfb6b123555031e23867ad943b09a56606d3163e369be3326`.
Raw evidence belongs to
`build/qualification/host-stationary-inplace-source-20261113`;
the pre-change allocation diagnosis belongs to
`build/qualification/host-stationary-allocation-prefix-20261113`.

The production qualification passes1298 Python cases and six native CTest
groups, including the new contracted/overflow controls and independent assembled
stationary gradients. A focused ASan/UBSan build instruments the actual complete
`s_integrals.cpp` owner and native MP2 contract test, links other dependencies
from the matching frozen core, and passes with leak detection and halt-on-error.
This is not a claim that every dependency in the shared core is instrumented.
Six documentation tests (with six additional subtests) and a Sphinx
warnings-as-errors build pass.

Three unmodified-allocator interleaved baseline/production timing pairs use
explicit frozen cores, identical 8,000,000 primitive records and captured
inputs. Median interrupted apply is7.144593900 versus5.710356391 seconds;
packaged leaf scopes are3.542104241 versus3.492982375 seconds. Normal prepared
values are separately4.335823085 versus4.385702519 seconds. No partial gradient
is published, and this remains a component prefix rather than a complete
stationary gradient/Hessian or optimizer timing.

The separate interposed production allocation probe observes177,483 calls and
11,555,103 requested bytes after the first ERI, including73,500 96-byte calls.
The before-first-ERI counts are unchanged. This removes56 million malloc calls
and5,376,000,000 requested bytes on the same work, without changing live storage
admission or inferring a peak-memory reduction. Exact counts are retained
one-off evidence, not a permanent CI implementation constraint. The diagnostic
archive SHA256 is
`b50232ba8a1d0b70fbdae79941c39933ced39f4a77515ec973a9283ca876b829`.

## Rejected alternatives and revisit conditions

- Do not replace the compiler-generated scalar-AD arithmetic with hand-written
  derivative rules or drop nominal zero terms for speed.
- Hoisting coefficient arithmetic, changing summation order, geometry caching
  and parallel reduction are separate proposals requiring their own evidence;
  they are not needed to remove primitive-local heap storage.
- Do not pin exact malloc counts in permanent CI. Those counters are one-off
  performance evidence; numerical, error and ownership behavior are the contract.

This does not complete a full Hessian/response/mode/optimizer endpoint, nor open
either public ECP saddle guard. Complete endpoint qualification, mixed Ni/I/Cs,
real stationary cases, practical size and the requested native Ni TS acceptance
remain separate requirements. Revisit other source/response phases from actual
measured work, rather than assuming this storage change removes their cost.
