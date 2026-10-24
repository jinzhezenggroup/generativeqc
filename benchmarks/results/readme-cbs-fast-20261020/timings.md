# Standard and Fast DF-CCSD(T)/CBS

Full DF: RI-JK reference/response; shared JKFIT auxiliaries for correlation.

One API-cold complete E+F observation per policy and molecule; same binary and GPU within each pair.
Fast differences are against Standard, not independent oracle errors. Constituent budgets also apply.

| Alkane | Standard (s) | Fast (s) | Max composite ΔE (Eh) | Max force difference (Eh/Bohr) | RMS force difference (Eh/Bohr) | Fast control |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C1 | 4.185 | 4.163 | 5.424e-11 | 4.083e-10 | 1.562e-10 | passed |
| C2 | 12.118 | 11.557 | 6.295e-10 | 2.202e-10 | 8.608e-11 | passed |
| C3 | 36.328 | 33.162 | 6.533e-10 | 2.504e-10 | 1.195e-10 | passed |
| C4 | 100.110 | 87.582 | 7.925e-10 | 3.149e-10 | 1.151e-10 | passed |
| C5 | 249.410 | 219.169 | 1.241e-09 | 6.573e-10 | 2.035e-10 | passed |
| C6 | 710.274 | 611.516 | 1.732e-09 | 7.894e-10 | 2.099e-10 | passed |
| C7 | 1788.437 | 1484.929 | 2.375e-09 | 1.255e-09 | 3.496e-10 | passed |

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

## Standard diagnostics

### Complete endpoints

Full DF: RI-JK reference/response; shared JKFIT auxiliaries for correlation.

One fresh-calculator cold observation per molecule unless noted. Seconds include
both native basis groups and returned host forces; they are not RHF-only times.
‡ = independent accuracy unchecked. Checked means constituent + composite energy
and one normalized force projection at two finite-difference steps, not every force component.

| Alkane | QM7 index | AO DZ/TZ | Auxiliary DZ/TZ | Complete E+F (s) | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| C1H4 | 0 | 34/86 | 162/199 | 4.185 | unchecked ‡ |
| C2H6 | 1 | 58/144 | 278/338 | 12.118 | checked (directional) |
| C3H8 | 8 | 82/202 | 394/477 | 36.328 | unchecked ‡ |
| C4H10 | 48 | 106/260 | 510/616 | 100.110 | unchecked ‡ |
| C5H12 | 183 | 130/318 | 626/755 | 249.410 | unchecked ‡ |
| C6H14 | 293 | 154/376 | 742/894 | 710.274 | unchecked ‡ |
| C7H16 | 5604 | 178/434 | 858/1033 | 1788.437 | unchecked ‡ |

### Observed work

DZ SCF iteration counts are unavailable: the frozen companion ABI returns
the primary CC convergence counter, not the HF counter. They are not inferred.
TZ has an MP2 primary whose convergence counter records its HF reference.
Response iterations are separate for DZ CCSD(T), DZ MP2 and TZ MP2.
Group times are non-overlapping; individual companion timings must not be added.

| Alkane | SCF iterations DZ/TZ | CCSD iterations | Response iterations CC/DZ-MP2/TZ-MP2 | Group seconds DZ/TZ | Max ΔE (Eh) | Max directional error (Eh/Bohr) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | —/15 | 15 | 10/12/12 | 1.982/2.202 | — | — |
| C2 | —/18 | 18 | 11/13/14 | 5.715/6.402 | 3.965e-12 | 5.454e-09 |
| C3 | —/21 | 19 | 13/16/16 | 21.248/15.079 | — | — |
| C4 | —/23 | 20 | 14/17/17 | 70.865/29.244 | — | — |
| C5 | —/25 | 19 | 15/17/18 | 198.420/50.989 | — | — |
| C6 | —/27 | 21 | 15/17/18 | 626.463/83.809 | — | — |
| C7 | —/27 | 21 | 15/17/18 | 1651.067/137.368 | — | — |


## Fast diagnostics

### Complete endpoints

Full DF: RI-JK reference/response; shared JKFIT auxiliaries for correlation.

One fresh-calculator cold observation per molecule unless noted. Seconds include
both native basis groups and returned host forces; they are not RHF-only times.
‡ = independent accuracy unchecked. Checked means constituent + composite energy
and one normalized force projection at two finite-difference steps, not every force component.

| Alkane | QM7 index | AO DZ/TZ | Auxiliary DZ/TZ | Complete E+F (s) | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| C1H4 | 0 | 34/86 | 162/199 | 4.163 | unchecked ‡ |
| C2H6 | 1 | 58/144 | 278/338 | 11.557 | checked (directional) |
| C3H8 | 8 | 82/202 | 394/477 | 33.162 | unchecked ‡ |
| C4H10 | 48 | 106/260 | 510/616 | 87.582 | unchecked ‡ |
| C5H12 | 183 | 130/318 | 626/755 | 219.169 | unchecked ‡ |
| C6H14 | 293 | 154/376 | 742/894 | 611.516 | unchecked ‡ |
| C7H16 | 5604 | 178/434 | 858/1033 | 1484.929 | unchecked ‡ |

### Observed work

DZ SCF iteration counts are unavailable: the frozen companion ABI returns
the primary CC convergence counter, not the HF counter. They are not inferred.
TZ has an MP2 primary whose convergence counter records its HF reference.
Response iterations are separate for DZ CCSD(T), DZ MP2 and TZ MP2.
Group times are non-overlapping; individual companion timings must not be added.

| Alkane | SCF iterations DZ/TZ | CCSD iterations | Response iterations CC/DZ-MP2/TZ-MP2 | Group seconds DZ/TZ | Max ΔE (Eh) | Max directional error (Eh/Bohr) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | —/14 | 12 | 7/8/9 | 1.976/2.185 | — | — |
| C2 | —/16 | 13 | 8/9/10 | 5.195/6.361 | 6.290e-10 | 5.416e-09 |
| C3 | —/18 | 13 | 9/11/11 | 18.227/14.934 | — | — |
| C4 | —/21 | 13 | 9/11/12 | 58.670/28.910 | — | — |
| C5 | —/22 | 13 | 10/12/13 | 168.807/50.361 | — | — |
| C6 | —/24 | 13 | 10/12/13 | 528.934/82.581 | — | — |
| C7 | —/24 | 13 | 10/13/13 | 1350.341/134.586 | — | — |
