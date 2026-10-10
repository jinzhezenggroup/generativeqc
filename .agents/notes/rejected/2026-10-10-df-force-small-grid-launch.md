# Trial: smaller point blocks and wider Becke pair blocks after fitted AO masks

Status: rejected for default promotion; numerical diagnostic passes
Date: 2026-10-10

## Motivation and frozen source

After the qualified fitted AO bitmask change, a separate force-only profile
attributes 1.807984 s to stationary point evaluation and 1.333425 s to primal
Becke pairs. A 256-point tile launches only two 128-thread point CTAs, while
the Becke pair domain launches two CTAs per pair. Test schedule changes rather
than introducing another XC formula or reducing the mathematical pair domain.
See the sibling implemented note
`../implemented/performance/2026-10-10-df-force-active-ao.md` for the independently
qualified policy and full complete-endpoint population.

Copy the frozen master-464df951f-plus-bitmask source and runnable artifacts to a
separate source/build directory. Change only the native stationary template:
point evaluation blocks 128 -> 32 threads; primal and all reverse pair blocks
128 -> 256 threads. Rebuild its PBE0-RKS SPD AOT wrapper and matching manifest
through verified ccache 4.5.1. Retain the unchanged main library and primitive
science, every AO jet/point/pair, coefficient, force channel and budget. This
trial is not part of the production policy patch or its timing population.

The first configure uses an incorrect `Python_EXECUTABLE` CMake variable and
selects a Python without NumPy. It fails before GPU execution. Reconfigure with
`Python3_EXECUTABLE`, the correct source-local Python path and checkout-root
ccache normalization; retain both the failure and corrected build receipts.

## Single diagnostic endpoint

The same fresh-owner 96-atom FP64 PBE0-DF unpruned 48x16x32 cold protocol runs
through a finite Slurm GPU job. Complete E+F is 86.064101 s, force is 15.564293 s,
preparation is 18.248663 s, and native work remains 24 iterations/24 Focks with
physical residual 5.972529e-13. The qualified policy-only population has median
86.942887 s complete / 16.440103 s force. Thus the single trial's apparent
complete-endpoint gain is only about 1.01%, not a repeated population.

The scientific full-vector gate passes, but that alone does not qualify a
schedule default. Do not start a large qualification campaign or promote this
combined schedule on a single below-2% endpoint observation. It also does not
establish which of the two schedule changes accounts for the small gain, nor
qualification for primitive/normalized adjoints, smaller capability descriptors,
other shapes, or other devices. In particular a production 256-thread pair
schedule would need its own compiler-owned capability/admission checks instead
of silently relying on a descriptor that promises only 128 threads.

## Retention and revisit

The source delta, configure/build/cache logs and complete probe receipt remain
under local `.artifacts/pbe0-parity/` and n1's
`/data/jzzeng/qc-pbe0-df-cold-20261009-4e20f7a7b/results/`; the trial checkout is
`parity-point-trial` under that root. Do not bind its timing to the qualified
policy-only main source or silently install its wrapper in a published build.

Revisit only with a causal schedule alternative expected to materially improve
the complete endpoint and preserve every capability/fallback gate. Non-force
preparation/SCF, snapshot construction, repeated host validation/source lookup,
and compiler lowering of the exact point model remain separate avenues. The
remaining reference gap cannot be solved by claiming this one kernel-schedule
probe reaches parity.
