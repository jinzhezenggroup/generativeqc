# DF-CCSD(T)/CBS complete energy + analytic forces

Correlation-only DF; final RHF uses exact J/K.

One fresh-calculator cold observation per molecule unless noted. Seconds include
both native basis groups and returned host forces; they are not RHF-only times.
‡ = independent accuracy unchecked. Checked means constituent + composite energy
and one normalized force projection at two finite-difference steps, not every force component.

| Alkane | QM7 index | AO DZ/TZ | Auxiliary DZ/TZ | Complete E+F (s) | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| C1H4 | 0 | 34/86 | 112/201 | 14.542 | checked (directional) |
| C2H6 | 1 | 58/144 | 196/342 | 38.492 | checked (directional) |
| C3H8 | 8 | 82/202 | 280/483 | 117.739 | checked (directional) |
| C4H10 | 48 | 106/260 | 364/624 | 309.238 | unchecked ‡ |
| C5H12 | 183 | 130/318 | 448/765 | 2660.194 | unchecked ‡ |
| C6H14 | 293 | 154/376 | 532/906 | 5579.070 | unchecked ‡ |
| C7H16 | 5604 | 178/434 | 616/1047 | 10111.064 | unchecked ‡ |

## Observed work

DZ SCF iteration counts are unavailable: the frozen companion ABI returns
the primary CC convergence counter, not the HF counter. They are not inferred.
TZ has an MP2 primary whose convergence counter records its HF reference.
Response iterations are separate for DZ CCSD(T), DZ MP2 and TZ MP2.
Group times are non-overlapping; individual companion timings must not be added.

| Alkane | SCF iterations DZ/TZ | CCSD iterations | Response iterations CC/DZ-MP2/TZ-MP2 | Group seconds DZ/TZ | Max ΔE (Eh) | Max directional error (Eh/Bohr) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | —/19 | 17 | 10/12/12 | 6.355/8.186 | 3.446e-12 | 5.233e-09 |
| C2 | —/12 | 18 | 12/13/14 | 12.825/25.665 | 4.107e-12 | 4.571e-09 |
| C3 | —/13 | 19 | 14/16/16 | 23.072/94.666 | 5.315e-12 | 3.181e-08 |
| C4 | —/14 | 20 | 14/17/17 | 65.257/243.979 | — | — |
| C5 | —/15 | 19 | 15/17/18 | 178.855/2481.338 | — | — |
| C6 | —/15 | 21 | 15/17/18 | 647.759/4931.308 | — | — |
| C7 | —/14 | 21 | 15/17/18 | 1389.803/8721.260 | — | — |
