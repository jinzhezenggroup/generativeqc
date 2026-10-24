# Standard and Fast DF-CCSD(T)/CBS

Full DF: RI-JK reference/response; shared JKFIT auxiliaries for correlation.

One API-cold complete E+F observation per policy and molecule; same binary and GPU within each pair.
Fast differences are against Standard, not independent oracle errors. Constituent budgets also apply.

| Alkane | Standard (s) | Fast (s) | Max composite ΔE (Eh) | Max force difference (Eh/Bohr) | RMS force difference (Eh/Bohr) | Fast control |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C1 | 5.091 | 4.512 | 5.424e-11 | 4.075e-10 | 1.568e-10 | passed |
| C2 | 10.665 | 9.727 | 6.295e-10 | 2.252e-10 | 8.357e-11 | passed |
| C3 | 30.342 | 25.397 | 6.533e-10 | 2.576e-10 | 1.231e-10 | passed |
| C4 | 86.148 | 67.539 | 7.925e-10 | 3.033e-10 | 1.129e-10 | passed |
| C5 | 222.235 | 171.701 | 1.241e-09 | 6.697e-10 | 2.080e-10 | passed |
| C6 | 650.381 | 483.648 | 1.732e-09 | 8.016e-10 | 2.144e-10 | passed |
| C7 | 1447.813 | 1093.350 | 2.375e-09 | 1.251e-09 | 3.484e-10 | passed |

## Fast arithmetic

FP32 counts cover auxiliary CCSD preconvergence; final checks and response use FP64.
Unknown means the older receipt did not record precision telemetry.

| Alkane | Executed arithmetic | FP32 refused | FP32 evaluations | FP32 GEMMs | Cast elements |
| --- | --- | --- | ---: | ---: | ---: |
| C1 | mixed | False | 6 | 684 | 412902 |
| C2 | mixed | False | 6 | 1026 | 3126494 |
| C3 | mixed | False | 6 | 1482 | 11889942 |
| C4 | mixed | False | 6 | 1824 | 32290446 |
| C5 | mixed | False | 6 | 2280 | 71758406 |
| C6 | mixed | False | 6 | 2736 | 139567422 |
| C7 | mixed | False | 6 | 3078 | 246834294 |

## Lambda work

FP32 auxiliary warmup is followed by a fresh FP64 solve and independent audit.
Actions count the shared transpose operator, excluding the separate audit; phase time includes both.
Q is the actual auxiliary batch. Numeric capacity is not measured peak VRAM.

| Alkane | Lambda Standard/Fast (s) | Standard FP64 actions | Fast FP32/FP64 actions | Fast Q | Fast FP32 GEMMs | Fast numeric capacity (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 0.405/0.336 | 18 | 10/6 | 32 | 1620 | 8.160 |
| C2 | 2.312/1.575 | 21 | 11/6 | 32 | 2673 | 8.640 |
| C3 | 11.156/7.123 | 22 | 11/7 | 32 | 3861 | 10.305 |
| C4 | 40.108/24.733 | 23 | 12/7 | 32 | 5184 | 14.334 |
| C5 | 108.895/67.050 | 23 | 12/7 | 32 | 6480 | 22.314 |
| C6 | 339.732/206.039 | 23 | 12/7 | 32 | 7776 | 28.324 |
| C7 | 798.776/513.805 | 24 | 12/8 | 16 | 17496 | 31.162 |

## Standard diagnostics

### Complete endpoints

Full DF: RI-JK reference/response; shared JKFIT auxiliaries for correlation.

One fresh-calculator cold observation per molecule unless noted. Seconds include
both native basis groups and returned host forces; they are not RHF-only times.
‡ = independent accuracy unchecked. Checked means constituent + composite energy
and one normalized force projection at two finite-difference steps, not every force component.

| Alkane | QM7 index | AO DZ/TZ | Auxiliary DZ/TZ | Complete E+F (s) | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| C1H4 | 0 | 34/86 | 162/199 | 5.091 | unchecked ‡ |
| C2H6 | 1 | 58/144 | 278/338 | 10.665 | checked (directional) |
| C3H8 | 8 | 82/202 | 394/477 | 30.342 | unchecked ‡ |
| C4H10 | 48 | 106/260 | 510/616 | 86.148 | unchecked ‡ |
| C5H12 | 183 | 130/318 | 626/755 | 222.235 | unchecked ‡ |
| C6H14 | 293 | 154/376 | 742/894 | 650.381 | unchecked ‡ |
| C7H16 | 5604 | 178/434 | 858/1033 | 1447.813 | unchecked ‡ |

### Observed work

DZ SCF iteration counts are unavailable: the frozen companion ABI returns
the primary CC convergence counter, not the HF counter. They are not inferred.
TZ has an MP2 primary whose convergence counter records its HF reference.
Response iterations are separate for DZ CCSD(T), DZ MP2 and TZ MP2.
Group times are non-overlapping; individual companion timings must not be added.

| Alkane | SCF iterations DZ/TZ | CCSD iterations | Response iterations CC/DZ-MP2/TZ-MP2 | Group seconds DZ/TZ | Max ΔE (Eh) | Max directional error (Eh/Bohr) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | —/15 | 15 | 10/12/12 | 2.668/2.422 | — | — |
| C2 | —/18 | 18 | 11/13/14 | 5.950/4.715 | 3.965e-12 | 5.454e-09 |
| C3 | —/21 | 19 | 13/16/16 | 21.024/9.317 | — | — |
| C4 | —/23 | 20 | 14/17/17 | 69.794/16.353 | — | — |
| C5 | —/25 | 19 | 15/17/18 | 195.271/26.963 | — | — |
| C6 | —/27 | 21 | 15/17/18 | 605.685/44.694 | — | — |
| C7 | —/27 | 21 | 15/17/18 | 1369.031/78.780 | — | — |


## Fast diagnostics

### Complete endpoints

Full DF: RI-JK reference/response; shared JKFIT auxiliaries for correlation.

One fresh-calculator cold observation per molecule unless noted. Seconds include
both native basis groups and returned host forces; they are not RHF-only times.
‡ = independent accuracy unchecked. Checked means constituent + composite energy
and one normalized force projection at two finite-difference steps, not every force component.

| Alkane | QM7 index | AO DZ/TZ | Auxiliary DZ/TZ | Complete E+F (s) | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| C1H4 | 0 | 34/86 | 162/199 | 4.512 | unchecked ‡ |
| C2H6 | 1 | 58/144 | 278/338 | 9.727 | checked (directional) |
| C3H8 | 8 | 82/202 | 394/477 | 25.397 | unchecked ‡ |
| C4H10 | 48 | 106/260 | 510/616 | 67.539 | unchecked ‡ |
| C5H12 | 183 | 130/318 | 626/755 | 171.701 | unchecked ‡ |
| C6H14 | 293 | 154/376 | 742/894 | 483.648 | unchecked ‡ |
| C7H16 | 5604 | 178/434 | 858/1033 | 1093.350 | unchecked ‡ |

### Observed work

DZ SCF iteration counts are unavailable: the frozen companion ABI returns
the primary CC convergence counter, not the HF counter. They are not inferred.
TZ has an MP2 primary whose convergence counter records its HF reference.
Response iterations are separate for DZ CCSD(T), DZ MP2 and TZ MP2.
Group times are non-overlapping; individual companion timings must not be added.

| Alkane | SCF iterations DZ/TZ | CCSD iterations | Response iterations CC/DZ-MP2/TZ-MP2 | Group seconds DZ/TZ | Max ΔE (Eh) | Max directional error (Eh/Bohr) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | —/14 | 12 | 7/8/9 | 2.104/2.407 | — | — |
| C2 | —/16 | 13 | 8/9/10 | 5.071/4.655 | 6.290e-10 | 5.416e-09 |
| C3 | —/18 | 13 | 9/11/11 | 16.182/9.214 | — | — |
| C4 | —/21 | 13 | 9/11/12 | 51.451/16.086 | — | — |
| C5 | —/22 | 13 | 10/12/13 | 145.168/26.531 | — | — |
| C6 | —/24 | 13 | 10/12/13 | 440.786/42.860 | — | — |
| C7 | —/24 | 13 | 10/13/13 | 1017.427/75.921 | — | — |
