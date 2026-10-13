# DF-CCSD(T)/CBS complete energy + analytic forces

Full DF: RI-JK reference/response; shared JKFIT auxiliaries for correlation.

One fresh-calculator cold observation per molecule unless noted. Seconds include
both native basis groups and returned host forces; they are not RHF-only times.
‡ = independent accuracy unchecked. Checked means constituent + composite energy
and one normalized force projection at two finite-difference steps, not every force component.

| Alkane | QM7 index | AO DZ/TZ | Auxiliary DZ/TZ | Complete E+F (s) | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| C1H4 | 0 | 34/86 | 162/199 | 4.448 | unchecked ‡ |
| C2H6 | 1 | 58/144 | 278/338 | 12.143 | checked (directional) |
| C3H8 | 8 | 82/202 | 394/477 | 35.875 | unchecked ‡ |
| C4H10 | 48 | 106/260 | 510/616 | 98.557 | unchecked ‡ |
| C5H12 | 183 | 130/318 | 626/755 | 245.611 | unchecked ‡ |
| C6H14 | 293 | 154/376 | 742/894 | 705.484 | unchecked ‡ |
| C7H16 | 5604 | 178/434 | 858/1033 | 1794.984 | unchecked ‡ |

## Observed work

DZ SCF iteration counts are unavailable: the frozen companion ABI returns
the primary CC convergence counter, not the HF counter. They are not inferred.
TZ has an MP2 primary whose convergence counter records its HF reference.
Response iterations are separate for DZ CCSD(T), DZ MP2 and TZ MP2.
Group times are non-overlapping; individual companion timings must not be added.

| Alkane | SCF iterations DZ/TZ | CCSD iterations | Response iterations CC/DZ-MP2/TZ-MP2 | Group seconds DZ/TZ | Max ΔE (Eh) | Max directional error (Eh/Bohr) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | —/15 | 15 | 10/12/12 | 2.250/2.197 | — | — |
| C2 | —/18 | 18 | 11/13/14 | 5.845/6.297 | 3.965e-12 | 5.454e-09 |
| C3 | —/21 | 19 | 13/16/16 | 21.048/14.826 | — | — |
| C4 | —/23 | 20 | 14/17/17 | 69.743/28.813 | — | — |
| C5 | —/25 | 19 | 15/17/18 | 195.333/50.277 | — | — |
| C6 | —/27 | 21 | 15/17/18 | 618.888/86.595 | — | — |
| C7 | —/27 | 21 | 15/17/18 | 1656.495/138.488 | — | — |
