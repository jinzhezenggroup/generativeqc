# Matched-grid complete energy + analytic forces

Warm values retain all requested fixed-density replays. Columns show median seconds and observed SCF iterations/builds.
Same-grid engine parity and cross-grid convergence are separate gates; see grid-convergence.md.
Exploratory grids: FP64 quadrature accuracy is not certified.

| Mode | Grid | Atoms | Engine | Cold (s) | Warm (s) | Cold work | Warm work |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| none | 48×12×24 | 3 | native | 6.892 | 0.382 | 25/25 | 1/1 |
| none | 48×12×24 | 12 | native | 17.874 | 1.081 | 30/31 | 1/1 |
| none | 48×12×24 | 3 | reference | 23.910 | 2.249 | 32/33 | 1/2 |
| none | 48×12×24 | 12 | reference | 56.058 | 4.275 | 45/46 | 1/2 |
| none | 48×16×32 | 3 | native | 7.439 | 0.343 | 25/26 | 1/1 |
| none | 48×16×32 | 12 | native | 24.314 | 1.429 | 30/31 | 1/1 |
| none | 48×16×32 | 3 | reference | 28.640 | 2.667 | 32/33 | 1/2 |
| none | 48×16×32 | 12 | reference | 109.557 | 8.596 | 45/46 | 1/2 |
| cuda | 48×12×24 | 3 | native | 6.173 | 0.344 | 25/25 | 1/1 |
| cuda | 48×12×24 | 12 | native | 15.975 | 0.971 | 30/30 | 1/1 |
| cuda | 48×12×24 | 3 | reference | 23.587 | 1.808 | 32/33 | 1/2 |
| cuda | 48×12×24 | 12 | reference | 50.187 | 3.620 | 45/46 | 1/2 |
| cuda | 48×16×32 | 3 | native | 7.015 | 0.347 | 25/25 | 1/1 |
| cuda | 48×16×32 | 12 | native | 22.501 | 1.299 | 30/30 | 1/1 |
| cuda | 48×16×32 | 3 | reference | 27.112 | 2.238 | 32/33 | 1/2 |
| cuda | 48×16×32 | 12 | reference | 101.387 | 7.893 | 45/46 | 1/2 |
