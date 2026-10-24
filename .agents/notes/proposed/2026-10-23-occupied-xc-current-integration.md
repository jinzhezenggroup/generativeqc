# Candidate: integrate the existing occupied-XC proposal with current cold SCF

Status: current integration qualified; retained opt-in, no default promotion
Date: 2026-10-23

## Source and scope

Reuse the compiler/native implementation and executable tests from PR #2577,
source `b1851abc43856f777a0f99e8150bc87b2d8e6b66`, on the current raw-J and
charge-partition base. The original PR's historical grids, timings and numerical
receipts remain attached to their original source; none is a result for this
integration. The retained upstream patch in local artifacts records the import.

The existing proposal lends the exact canonical frame that formed accepted D
to strict-FP64 meta-GGA XC. Project four orbital jets when the occupied rank
is smaller than the actual selected AO domain, using dead density panels.
Potential assembly keeps its original AO jets. Unsupported/mixed/nonlocal,
response, imported and held densities retain matrix contraction. The selector
`GENERATIVEQC_CUDA_KS_OCCUPIED_XC=1` remains explicit; unset remains off.

## Integration decisions

Preserve the current explicit FP64 FMA and normalized FP32 tail arithmetic in
the shared product body, including its inactive-source resource specializations.
Preserve current FP32 admission and nonlocal density-feature phase controls.

Newer cold SCF can run an early final-state proof that overwrites tmp1/tmp2.
Therefore every completion decision retires the previous occupied-XC lease;
only accepting a proposal without that scratch-consuming proof can republish
it. Held stationary confirmations and rejected probes retain the dense fallback.
Extend the executable owner fault tests to check retirement and later recovery.

The old PR's source-spelling mutation tests are not imported: the current
compiled XC handoff probes already check full versus delta density and remain
in the integration suite. No numerical or lifetime coverage is removed.

## Required evidence

Current-source independent dense-feature, native owner, alias/stale-generation,
resource, sanitizer and complete energy/force checks precede promotion claims.
Cold/warm/moved results and executed occupied-frame counts must retain this
integration's own source and package identities. PBE's cheaper one-jet density
product is outside this four-jet candidate and must retain its current path.

## Current-source qualification

At 66a8a98bf5e81d17a71225f9fcd091022d64e4bf, five alternating fresh-process
r2SCAN-3c pairs improve complete cold median 43.990828 to 41.718511 seconds
for DF (5.165%) and 49.427546 to 47.710711 seconds for Direct (3.473%). Both
exceed the 2% variability floor. Warm remains on the matrix fallback and has
no qualified gain. All 168 pilot, paired, diagnostic and moved calls pass
independent gates. The largest moved force difference is 1.316e-5 Eh/Bohr
(Fast Direct). These gains do not yet reach GPU4's complete cold endpoint.

1260 host tests, 18 independent GPU cases and eight owner cases under each
of memcheck/synccheck/racecheck pass. Trace receipts prove eight/nine occupied
frames in r2SCAN/3c and no occupied frame in held warm confirmation. The exact
source inventory, original failed integration assertions, full numerical
receipts and executable replay are retained under
`benchmarks/results/occupied-xc-cold-20261023/`.
