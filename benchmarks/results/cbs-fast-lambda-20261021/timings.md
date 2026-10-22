# Standard and Fast DF-CCSD(T)/CBS

Full DF: RI-JK reference/response; shared JKFIT auxiliaries for correlation.

One API-cold complete E+F observation per policy and molecule; same binary and GPU within each pair.
Fast differences are against Standard, not independent oracle errors. Constituent budgets also apply.

| Alkane | Standard (s) | Fast (s) | Max composite ΔE (Eh) | Max force difference (Eh/Bohr) | RMS force difference (Eh/Bohr) | Fast control |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C1 | 4.663 | 4.410 | 5.424e-11 | 4.075e-10 | 1.568e-10 | passed |
| C2 | 12.305 | 11.392 | 6.295e-10 | 2.254e-10 | 8.365e-11 | passed |
| C3 | 36.131 | 31.194 | 6.533e-10 | 2.559e-10 | 1.229e-10 | passed |
| C4 | 99.351 | 80.415 | 7.925e-10 | 3.030e-10 | 1.130e-10 | passed |
| C5 | 247.676 | 196.187 | 1.241e-09 | 6.687e-10 | 2.078e-10 | passed |
| C6 | 710.270 | 539.012 | 1.732e-09 | 8.018e-10 | 2.146e-10 | passed |
| C7 | 1788.217 | 1282.575 | 2.375e-09 | 1.251e-09 | 3.484e-10 | passed |

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
| C1 | 0.409/0.336 | 18 | 10/6 | 32 | 1620 | 8.168 |
| C2 | 2.323/1.567 | 21 | 11/6 | 32 | 2673 | 8.712 |
| C3 | 11.267/7.182 | 22 | 11/7 | 32 | 3861 | 10.603 |
| C4 | 40.374/24.907 | 23 | 12/7 | 32 | 5184 | 15.180 |
| C5 | 109.658/67.310 | 23 | 12/7 | 32 | 6480 | 24.249 |
| C6 | 357.493/219.241 | 23 | 12/7 | 32 | 7776 | 31.049 |
| C7 | 1080.824/644.370 | 24 | 12/8 | 4 | 69660 | 30.747 |

## Standard diagnostics

### Complete endpoints

Full DF: RI-JK reference/response; shared JKFIT auxiliaries for correlation.

One fresh-calculator cold observation per molecule unless noted. Seconds include
both native basis groups and returned host forces; they are not RHF-only times.
‡ = independent accuracy unchecked. Checked means constituent + composite energy
and one normalized force projection at two finite-difference steps, not every force component.

| Alkane | QM7 index | AO DZ/TZ | Auxiliary DZ/TZ | Complete E+F (s) | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| C1H4 | 0 | 34/86 | 162/199 | 4.663 | unchecked ‡ |
| C2H6 | 1 | 58/144 | 278/338 | 12.305 | checked (directional) |
| C3H8 | 8 | 82/202 | 394/477 | 36.131 | unchecked ‡ |
| C4H10 | 48 | 106/260 | 510/616 | 99.351 | unchecked ‡ |
| C5H12 | 183 | 130/318 | 626/755 | 247.676 | unchecked ‡ |
| C6H14 | 293 | 154/376 | 742/894 | 710.270 | unchecked ‡ |
| C7H16 | 5604 | 178/434 | 858/1033 | 1788.217 | unchecked ‡ |

### Observed work

DZ SCF iteration counts are unavailable: the frozen companion ABI returns
the primary CC convergence counter, not the HF counter. They are not inferred.
TZ has an MP2 primary whose convergence counter records its HF reference.
Response iterations are separate for DZ CCSD(T), DZ MP2 and TZ MP2.
Group times are non-overlapping; individual companion timings must not be added.

| Alkane | SCF iterations DZ/TZ | CCSD iterations | Response iterations CC/DZ-MP2/TZ-MP2 | Group seconds DZ/TZ | Max ΔE (Eh) | Max directional error (Eh/Bohr) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | —/15 | 15 | 10/12/12 | 2.481/2.182 | — | — |
| C2 | —/18 | 18 | 11/13/14 | 5.941/6.362 | 3.965e-12 | 5.454e-09 |
| C3 | —/21 | 19 | 13/16/16 | 21.181/14.949 | — | — |
| C4 | —/23 | 20 | 14/17/17 | 70.251/29.098 | — | — |
| C5 | —/25 | 19 | 15/17/18 | 196.920/50.755 | — | — |
| C6 | —/27 | 21 | 15/17/18 | 626.465/83.804 | — | — |
| C7 | —/27 | 21 | 15/17/18 | 1650.860/137.355 | — | — |


## Fast diagnostics

### Complete endpoints

Full DF: RI-JK reference/response; shared JKFIT auxiliaries for correlation.

One fresh-calculator cold observation per molecule unless noted. Seconds include
both native basis groups and returned host forces; they are not RHF-only times.
‡ = independent accuracy unchecked. Checked means constituent + composite energy
and one normalized force projection at two finite-difference steps, not every force component.

| Alkane | QM7 index | AO DZ/TZ | Auxiliary DZ/TZ | Complete E+F (s) | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| C1H4 | 0 | 34/86 | 162/199 | 4.410 | unchecked ‡ |
| C2H6 | 1 | 58/144 | 278/338 | 11.392 | checked (directional) |
| C3H8 | 8 | 82/202 | 394/477 | 31.194 | unchecked ‡ |
| C4H10 | 48 | 106/260 | 510/616 | 80.415 | unchecked ‡ |
| C5H12 | 183 | 130/318 | 626/755 | 196.187 | unchecked ‡ |
| C6H14 | 293 | 154/376 | 742/894 | 539.012 | unchecked ‡ |
| C7H16 | 5604 | 178/434 | 858/1033 | 1282.575 | unchecked ‡ |

### Observed work

DZ SCF iteration counts are unavailable: the frozen companion ABI returns
the primary CC convergence counter, not the HF counter. They are not inferred.
TZ has an MP2 primary whose convergence counter records its HF reference.
Response iterations are separate for DZ CCSD(T), DZ MP2 and TZ MP2.
Group times are non-overlapping; individual companion timings must not be added.

| Alkane | SCF iterations DZ/TZ | CCSD iterations | Response iterations CC/DZ-MP2/TZ-MP2 | Group seconds DZ/TZ | Max ΔE (Eh) | Max directional error (Eh/Bohr) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | —/14 | 12 | 7/8/9 | 2.234/2.175 | — | — |
| C2 | —/16 | 13 | 8/9/10 | 5.072/6.319 | 6.290e-10 | 5.416e-09 |
| C3 | —/18 | 13 | 9/11/11 | 16.361/14.832 | — | — |
| C4 | —/21 | 13 | 9/11/12 | 51.708/28.705 | — | — |
| C5 | —/22 | 13 | 10/12/13 | 145.983/50.202 | — | — |
| C6 | —/24 | 13 | 10/12/13 | 456.486/82.524 | — | — |
| C7 | —/24 | 13 | 10/13/13 | 1147.982/134.591 | — | — |
