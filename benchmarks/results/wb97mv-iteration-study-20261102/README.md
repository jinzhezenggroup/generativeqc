# A100 WB97M-V iteration investigation

Source: `d243aaf78c82f870d8110b40eeddc2950489cfbe866f586d070487f1e695562f`.
The same SM80 library, README water-96/def2-SVP fitted target, stock grids,
strict FP64, energy tolerance `1e-10`, density tolerance `1e-8`, DIIS eight,
and independent energy/force gates are used throughout. The measured library
hash and exact drivers are authenticated inside the capsule.

## Why 28 iterations

The default cold solve contains nine semilocal/RSH proposals, fourteen frozen
VV10 proposals, and five complete-target evaluations. Four of those target
evaluations perform a VV10 pair sweep and the closing one reuses an exact key.
Seven preliminary native atomic Hückel Focks are counted separately. GPU4PySCF's
retained default takes ten SCF cycles and eleven effective-potential builds;
these counters do not describe equal work. Its default energy/orbital-gradient
controls (`1e-9` / `sqrt(1e-9)`) also differ from native physical stopping gates.

The adaptive frozen-potential threshold was tuned for Direct calculations.
History's `physical_residual_max` is the maximum over spin-channel RMS values;
the solver also checks a separate AO maximum. The historical diagnostic cannot
be substituted for that maximum-norm gate.

## Controlled diagnostic

Successful exclusive A100 allocation **224502**, finite 15 minutes, fixes
`GENERATIVEQC_KS_RANGE_RESIDENT_VALUES=1` and
`GENERATIVEQC_KS_RESIDENT_MAX_BYTES=17179869184` for both variants. Only the
VV10 provisional schedule changes. Neither variant removes VV10 or loosens
final accuracy. Each has one complete cold energy-plus-force observation and
one frozen warm observation; this is a diagnostic pair, not statistical promotion.

| Schedule | Cold iterations | VV10 pair sweeps | Cold E+F (s) | Warm E+F (s) |
| --- | ---: | ---: | ---: | ---: |
| Default provisional schedule | 28 | 4 | 115.948489 | 14.508478 |
| Full target every iteration | 16 | 16 | 167.804866 | 14.513994 |

Reducing the iteration count this way increases cold latency by **44.72%**.
Both pass the independent dense-grid observable gates. Between these variants,
total energy differs by `4.55e-13 Ha`; maximum force difference is below
`3.59e-10 Ha/Bohr`. The final physical target agrees despite different work.
This result retains the current default; it does not establish optimal inner
stopping thresholds or resolve the remaining cold gap against GPU4PySCF.

## Failed and incomplete attempts

Allocation 224499 failed before numerical work because the task transfer server
rejected an unlisted shell-script URL. The retry uses its allowed Python wrapper.
Allocation 224501 completed default and ordinary-full-target observations, then
failed on the explicitly unsupported MINAO-with-DF request. The final
`default-minao` and reference callback variants were never reached. All receipts,
logs and controller proofs are retained. Its completed numerical observations
pass the independent gates but its failed allocation does not rank timings.
The ordinary schedule opt-out also disables automatic LR source residency;
224502 therefore controls that confounding policy explicitly. Do not use the
uncontrolled diagnostic as the isolated cost of extra VV10 evaluations.

## Replay

```bash
PYTHONPATH=.:python python benchmarks/results/wb97mv-iteration-study-20261102/report.py > /tmp/wb-study.json
PYTHONPATH=.:python python -O benchmarks/results/wb97mv-iteration-study-20261102/report.py > /tmp/wb-study-optimized.json
cmp /tmp/wb-study.json /tmp/wb-study-optimized.json
```

The exact-FP64 capsule reconstructs every original selected campaign file byte
for byte. The reader recomputes source/library/driver and scheduler checks,
independent and paired observables, and iteration work counts. Failed attempts
remain visible and do not authorize a performance/default promotion. No
production source or convergence policy changes in this investigation.
