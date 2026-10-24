# Complete E+F endpoint timings by protocol and hardware

Seconds: median [minimum, maximum] of three calls from each arm's own immutable cold density.
Cold includes construction/preparation and remains separately recorded in samples.json.
No cross-group pooling, speedup, original-protocol PASS or production promotion.

## legacy-four-owners-old-H100

| Atoms | Policy | Route | Default | SCF reuse off | Force reuse off | Both off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 48 | fast | df | 10.394 [10.351, 10.396] | 11.050 [10.999, 11.051] | 11.477 [11.431, 11.528] | 12.100 [12.098, 12.134] |
| 48 | fast | direct | 7.638 [7.623, 7.644] | 8.291 [8.263, 8.306] | 8.691 [8.691, 8.730] | 9.351 [9.328, 9.364] |
| 48 | standard | df | 10.381 [10.348, 10.393] | 11.027 [10.961, 11.039] | 11.474 [11.441, 11.476] | 12.098 [12.094, 12.098] |
| 48 | standard | direct | 8.332 [8.316, 8.343] | 8.950 [8.935, 8.969] | 9.379 [9.374, 9.403] | 10.000 [9.987, 10.034] |
| 96 | standard | direct | 25.682 [25.554, 25.717] | 28.275 [28.120, 28.279] | 29.905 [29.899, 29.913] | 39.696 [39.666, 39.760] |

## independent-arms-new-H100

| Atoms | Policy | Route | Default | SCF reuse off | Force reuse off | Both off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 96 | fast | df | 33.545 [33.481, 33.564] | 36.059 [35.966, 36.060] | 37.931 [37.925, 37.962] | 40.336 [40.217, 40.449] |
| 96 | fast | direct | 23.709 [23.691, 23.719] | 26.310 [26.296, 26.317] | 28.014 [27.990, 28.084] | 30.566 [30.455, 30.571] |
| 96 | standard | df | 33.664 [33.660, 33.749] | 36.116 [36.065, 36.140] | 37.984 [37.926, 37.989] | 40.344 [40.334, 40.434] |
