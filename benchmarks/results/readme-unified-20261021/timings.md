# Default endpoint timings

Absolute latency; qualification is limited to the sampled static geometries.

| Mode | Method | Atoms | Engine | Cold (s) | Warm median [range] (s) | Status |
|---|---|---:|---|---:|---|---|
| df | HF | 3 | GenerativeQC Fast | 0.6380 | 0.2153 [0.1975, 0.2260] | complete; accuracy failed |
| df | HF | 3 | GPU4PySCF | 3.4448 | 0.2228 [0.2122, 0.2259] | complete; accuracy failed |
| df | HF | 3 | GenerativeQC Standard | 0.5818 | 0.0194 [0.0194, 0.0206] | complete; accuracy failed |
| df | HF | 12 | GenerativeQC Fast | 1.4286 | 0.7048 [0.6977, 0.8142] | complete; accuracy failed |
| df | HF | 12 | GPU4PySCF | 3.9427 | 0.2935 [0.2738, 0.3329] | complete; accuracy failed |
| df | HF | 12 | GenerativeQC Standard | 1.1243 | 0.1510 [0.1505, 0.1516] | complete; accuracy failed |
| df | HF | 48 | GenerativeQC Fast | 26.6307 | 18.7083 [18.6693, 18.7294] | complete; accuracy failed |
| df | HF | 48 | GPU4PySCF | 5.0779 | 1.7020 [1.6949, 1.7195] | complete; accuracy failed |
| df | HF | 48 | GenerativeQC Standard | 15.4824 | 3.9457 [3.9432, 3.9490] | complete; accuracy failed |
| df | HF | 96 | GenerativeQC Fast | 184.4134 | 109.3097 [109.3070, 109.3413] | complete; accuracy failed |
| df | HF | 96 | GPU4PySCF | 19.7194 | 10.0236 [10.0190, 10.0383] | complete; accuracy failed |
| df | HF | 96 | GenerativeQC Standard | 123.2220 | 27.7534 [27.7475, 27.7587] | complete; accuracy failed |
| df | PBE | 3 | GenerativeQC Fast | 4.4013 | 2.8739 [2.7791, 2.8920] | complete; accuracy qualified |
| df | PBE | 3 | GPU4PySCF | 5.8410 | 0.2674 [0.2494, 0.2828] | complete; accuracy qualified |
| df | PBE | 3 | GenerativeQC Standard | 2.8079 | 0.0936 [0.0832, 0.1872] | complete; accuracy qualified |
| df | PBE | 12 | GenerativeQC Fast | 6.1743 | 3.7303 [3.7105, 3.7909] | complete; accuracy qualified |
| df | PBE | 12 | GPU4PySCF | 5.4156 | 0.3715 [0.3534, 0.3743] | complete; accuracy qualified |
| df | PBE | 12 | GenerativeQC Standard | 3.5238 | 0.3010 [0.3004, 0.3103] | complete; accuracy qualified |
| df | PBE | 48 | GenerativeQC Fast | 34.1745 | 25.8973 [25.8578, 25.9452] | complete; accuracy qualified |
| df | PBE | 48 | GPU4PySCF | 6.6043 | 1.4362 [1.4337, 1.4480] | complete; accuracy qualified |
| df | PBE | 48 | GenerativeQC Standard | 17.3202 | 5.6014 [5.5966, 5.6192] | complete; accuracy qualified |
| df | PBE | 96 | GenerativeQC Fast | 136.1732 | 117.9752 [117.9581, 118.0124] | complete; accuracy qualified |
| df | PBE | 96 | GPU4PySCF | 12.6767 | 4.6704 [4.6693, 4.6726] | complete; accuracy qualified |
| df | PBE | 96 | GenerativeQC Standard | 67.5046 | 29.3722 [29.3664, 29.3806] | complete; accuracy qualified |
| df | PBE0 | 3 | GenerativeQC Fast | 4.4767 | 2.7250 [2.6176, 2.8780] | complete; accuracy qualified |
| df | PBE0 | 3 | GPU4PySCF | 5.4748 | 0.3618 [0.3185, 0.4760] | complete; accuracy qualified |
| df | PBE0 | 3 | GenerativeQC Standard | 2.8188 | 0.0937 [0.0924, 0.1619] | complete; accuracy qualified |
| df | PBE0 | 12 | GenerativeQC Fast | 6.4679 | 3.7658 [3.7160, 3.8579] | complete; accuracy qualified |
| df | PBE0 | 12 | GPU4PySCF | 4.5236 | 0.4956 [0.4850, 0.6272] | complete; accuracy qualified |
| df | PBE0 | 12 | GenerativeQC Standard | 3.8182 | 0.4110 [0.4085, 0.4223] | complete; accuracy qualified |
| df | PBE0 | 48 | GenerativeQC Fast | 46.6459 | 35.9925 [35.9350, 36.0790] | complete; accuracy qualified |
| df | PBE0 | 48 | GPU4PySCF | 7.7445 | 2.7360 [2.7325, 2.7740] | complete; accuracy qualified |
| df | PBE0 | 48 | GenerativeQC Standard | 23.3783 | 10.8743 [10.8726, 10.8979] | complete; accuracy qualified |
| df | PBE0 | 96 | GenerativeQC Fast | 209.2864 | 177.0162 [176.9803, 177.0591] | complete; accuracy qualified |
| df | PBE0 | 96 | GPU4PySCF | 26.5060 | 13.6864 [13.6715, 13.7271] | complete; accuracy qualified |
| df | PBE0 | 96 | GenerativeQC Standard | 105.4751 | 56.3720 [56.3408, 56.3735] | complete; accuracy qualified |
| df | B3LYP | 3 | GenerativeQC Fast | 4.2247 | 2.6956 [2.6927, 2.7401] | complete; accuracy qualified |
| df | B3LYP | 3 | GPU4PySCF | 4.2695 | 0.4000 [0.2973, 0.4270] | complete; accuracy qualified |
| df | B3LYP | 3 | GenerativeQC Standard | 2.6742 | 0.0944 [0.0930, 0.1026] | complete; accuracy qualified |
| df | B3LYP | 12 | GenerativeQC Fast | 6.3893 | 3.7491 [3.6977, 3.7880] | complete; accuracy qualified |
| df | B3LYP | 12 | GPU4PySCF | 5.0626 | 0.4874 [0.4746, 0.5192] | complete; accuracy qualified |
| df | B3LYP | 12 | GenerativeQC Standard | 3.7879 | 0.4599 [0.4453, 0.4973] | complete; accuracy qualified |
| df | B3LYP | 48 | GenerativeQC Fast | 46.8520 | 35.7009 [35.6885, 35.7452] | complete; accuracy qualified |
| df | B3LYP | 48 | GPU4PySCF | 7.4931 | 2.7300 [2.7271, 2.7690] | complete; accuracy qualified |
| df | B3LYP | 48 | GenerativeQC Standard | 24.2995 | 11.0574 [11.0570, 11.0632] | complete; accuracy qualified |
| df | B3LYP | 96 | GenerativeQC Fast | 211.8768 | 178.6441 [178.6231, 178.7047] | complete; accuracy qualified |
| df | B3LYP | 96 | GPU4PySCF | 27.0674 | 13.1808 [13.1751, 13.1936] | complete; accuracy qualified |
| df | B3LYP | 96 | GenerativeQC Standard | 107.4601 | 56.5978 [56.5811, 56.6220] | complete; accuracy qualified |
| df | r²SCAN | 3 | GenerativeQC Fast | 4.6741 | 2.9853 [2.9400, 2.9889] | complete; accuracy qualified |
| df | r²SCAN | 3 | GPU4PySCF | 5.1086 | 0.2837 [0.2780, 0.2952] | complete; accuracy qualified |
| df | r²SCAN | 3 | GenerativeQC Standard | 2.9180 | 0.0996 [0.0983, 0.1419] | complete; accuracy qualified |
| df | r²SCAN | 12 | GenerativeQC Fast | 7.0371 | 4.0612 [4.0192, 4.0974] | complete; accuracy qualified |
| df | r²SCAN | 12 | GPU4PySCF | 4.4858 | 0.4158 [0.4064, 0.4255] | complete; accuracy qualified |
| df | r²SCAN | 12 | GenerativeQC Standard | 3.9801 | 0.4000 [0.3988, 0.4024] | complete; accuracy qualified |
| df | r²SCAN | 48 | GenerativeQC Fast | 37.2555 | 26.5267 [26.5179, 26.5389] | complete; accuracy qualified |
| df | r²SCAN | 48 | GPU4PySCF | 7.9188 | 2.1740 [2.1710, 2.2145] | complete; accuracy qualified |
| df | r²SCAN | 48 | GenerativeQC Standard | 18.9679 | 6.1220 [6.1207, 6.1969] | complete; accuracy qualified |
| df | r²SCAN | 96 | GenerativeQC Fast | 142.2454 | 121.2786 [121.2580, 121.2787] | complete; accuracy qualified |
| df | r²SCAN | 96 | GPU4PySCF | 15.4430 | 6.4354 [6.4352, 6.4394] | complete; accuracy qualified |
| df | r²SCAN | 96 | GenerativeQC Standard | 70.3140 | 30.3305 [30.3234, 30.3306] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GenerativeQC Fast | 4.7086 | 2.8649 [2.8321, 3.1818] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GPU4PySCF | 5.7876 | 0.3034 [0.3017, 0.3063] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GenerativeQC Standard | 2.8166 | 0.1161 [0.1154, 0.1244] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GenerativeQC Fast | 8.1051 | 4.9886 [4.9638, 5.0810] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GPU4PySCF | 4.5578 | 0.5258 [0.5257, 0.5278] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GenerativeQC Standard | 4.4643 | 0.7461 [0.7445, 0.7765] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GenerativeQC Fast | 115.1530 | 96.1209 [96.1174, 96.3571] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GPU4PySCF | 10.8943 | 3.9048 [3.9011, 3.9049] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GenerativeQC Standard | 57.9033 | 27.1789 [27.1725, 27.1991] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GenerativeQC Fast | — | — | running; accuracy unverified |
| df | r²SCAN-3c | 96 | GPU4PySCF | 25.5686 | 12.4761 [12.4672, 12.4999] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GenerativeQC Standard | — | — | running; accuracy unverified |
| df | M06-2X | 3 | GenerativeQC Fast | 4.6502 | 3.0701 [2.9165, 3.0940] | complete; accuracy failed |
| df | M06-2X | 3 | GPU4PySCF | 4.3469 | 0.3401 [0.3250, 0.4765] | complete; accuracy failed |
| df | M06-2X | 3 | GenerativeQC Standard | 2.8583 | 0.1046 [0.1018, 0.1099] | complete; accuracy failed |
| df | M06-2X | 12 | GenerativeQC Fast | 7.3081 | 3.9478 [3.9410, 3.9965] | complete; accuracy qualified |
| df | M06-2X | 12 | GPU4PySCF | 4.8944 | 0.5148 [0.5095, 0.5345] | complete; accuracy qualified |
| df | M06-2X | 12 | GenerativeQC Standard | 4.1431 | 0.4727 [0.4697, 0.4738] | complete; accuracy qualified |
| df | M06-2X | 48 | GenerativeQC Fast | 49.5801 | 36.4602 [36.4517, 36.4950] | complete; accuracy qualified |
| df | M06-2X | 48 | GPU4PySCF | 9.0858 | 3.4455 [3.4386, 3.4813] | complete; accuracy qualified |
| df | M06-2X | 48 | GenerativeQC Standard | 25.0317 | 11.0655 [11.0623, 11.0928] | complete; accuracy qualified |
| df | M06-2X | 96 | GenerativeQC Fast | 226.9777 | 186.3021 [186.2729, 186.5508] | complete; accuracy qualified |
| df | M06-2X | 96 | GPU4PySCF | 30.6193 | 15.1516 [15.1476, 15.1659] | complete; accuracy qualified |
| df | M06-2X | 96 | GenerativeQC Standard | 110.3521 | 56.9736 [56.9725, 56.9787] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GenerativeQC Fast | 8.6247 | 3.6405 [3.6354, 3.7542] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GPU4PySCF | 5.6443 | 0.7941 [0.7070, 0.8768] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GenerativeQC Standard | 4.6961 | 0.3356 [0.3352, 0.3362] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GenerativeQC Fast | 25.5759 | 8.5495 [7.9331, 8.6252] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GPU4PySCF | 10.5762 | 2.7583 [2.6867, 2.7605] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GenerativeQC Standard | 12.8359 | 0.9599 [0.9599, 0.9601] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GenerativeQC Fast | 413.9807 | 113.0538 [112.8020, 113.1980] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GPU4PySCF | 91.1768 | 32.3900 [32.3634, 32.3961] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GenerativeQC Standard | 187.3665 | 17.5423 [17.5335, 17.5455] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GenerativeQC Fast | — | — | failed; accuracy unverified |
| df | ωB97M-V | 96 | GPU4PySCF | 388.8684 | 134.2228 [134.2017, 134.2312] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GenerativeQC Standard | — | — | failed; accuracy unverified |
| direct | HF | 3 | GenerativeQC Fast | 1.6433 | 0.0719 [0.0712, 0.0723] | complete; accuracy qualified |
| direct | HF | 3 | GPU4PySCF | 4.9800 | 0.8496 [0.8191, 1.0307] | complete; accuracy qualified |
| direct | HF | 3 | GenerativeQC Standard | 1.8018 | 0.0710 [0.0709, 0.0714] | complete; accuracy qualified |
| direct | HF | 12 | GenerativeQC Fast | 2.0978 | 0.1170 [0.1166, 0.1172] | complete; accuracy qualified |
| direct | HF | 12 | GPU4PySCF | 6.4586 | 1.8281 [1.6668, 2.2131] | complete; accuracy qualified |
| direct | HF | 12 | GenerativeQC Standard | 2.1469 | 0.1172 [0.1149, 0.1177] | complete; accuracy qualified |
| direct | HF | 48 | GenerativeQC Fast | 20.5575 | 5.4884 [5.3240, 5.4905] | complete; accuracy qualified |
| direct | HF | 48 | GPU4PySCF | 7.7503 | 1.9746 [1.9723, 2.0009] | complete; accuracy qualified |
| direct | HF | 48 | GenerativeQC Standard | 8.2118 | 0.7351 [0.7344, 0.7396] | complete; accuracy qualified |
| direct | HF | 96 | GenerativeQC Fast | 69.9823 | 16.9089 [16.8974, 16.9142] | complete; accuracy qualified |
| direct | HF | 96 | GPU4PySCF | 12.0100 | 3.4254 [3.4232, 3.4386] | complete; accuracy qualified |
| direct | HF | 96 | GenerativeQC Standard | 26.5793 | 3.0033 [2.9980, 3.0033] | complete; accuracy qualified |
| direct | PBE | 3 | GenerativeQC Fast | 1.5461 | 0.0586 [0.0583, 0.0586] | complete; accuracy qualified |
| direct | PBE | 3 | GPU4PySCF | 6.2100 | 0.7155 [0.7048, 0.8377] | complete; accuracy qualified |
| direct | PBE | 3 | GenerativeQC Standard | 1.5165 | 0.0565 [0.0564, 0.0572] | complete; accuracy qualified |
| direct | PBE | 12 | GenerativeQC Fast | 2.3479 | 0.2886 [0.2878, 0.2917] | complete; accuracy qualified |
| direct | PBE | 12 | GPU4PySCF | 6.2778 | 1.2013 [0.8825, 1.3626] | complete; accuracy qualified |
| direct | PBE | 12 | GenerativeQC Standard | 2.2414 | 0.2923 [0.2871, 0.2986] | complete; accuracy qualified |
| direct | PBE | 48 | GenerativeQC Fast | 12.1765 | 2.1054 [2.1050, 2.1185] | complete; accuracy qualified |
| direct | PBE | 48 | GPU4PySCF | 9.6066 | 2.3124 [2.3123, 2.3139] | complete; accuracy qualified |
| direct | PBE | 48 | GenerativeQC Standard | 15.2519 | 2.1794 [2.1648, 2.2001] | complete; accuracy qualified |
| direct | PBE | 96 | GenerativeQC Fast | 34.8120 | 6.8337 [6.8196, 6.8396] | complete; accuracy qualified |
| direct | PBE | 96 | GPU4PySCF | 15.9822 | 5.6073 [5.6026, 5.6136] | complete; accuracy qualified |
| direct | PBE | 96 | GenerativeQC Standard | 41.4217 | 6.7890 [6.7718, 6.8156] | complete; accuracy qualified |
| direct | PBE0 | 3 | GenerativeQC Fast | 1.9160 | 0.0915 [0.0912, 0.0916] | complete; accuracy qualified |
| direct | PBE0 | 3 | GPU4PySCF | 10.3579 | 2.0620 [1.7517, 2.0781] | complete; accuracy qualified |
| direct | PBE0 | 3 | GenerativeQC Standard | 1.9140 | 0.0891 [0.0889, 0.0906] | complete; accuracy qualified |
| direct | PBE0 | 12 | GenerativeQC Fast | 2.4868 | 0.3901 [0.3833, 0.3994] | complete; accuracy qualified |
| direct | PBE0 | 12 | GPU4PySCF | 9.5752 | 1.8145 [1.3544, 1.8971] | complete; accuracy qualified |
| direct | PBE0 | 12 | GenerativeQC Standard | 2.8624 | 0.3880 [0.3873, 0.3889] | complete; accuracy qualified |
| direct | PBE0 | 48 | GenerativeQC Fast | 13.3221 | 2.5121 [2.4938, 2.5303] | complete; accuracy qualified |
| direct | PBE0 | 48 | GPU4PySCF | 12.0164 | 3.0397 [3.0345, 3.0678] | complete; accuracy qualified |
| direct | PBE0 | 48 | GenerativeQC Standard | 16.7509 | 2.7334 [2.7329, 2.7376] | complete; accuracy qualified |
| direct | PBE0 | 96 | GenerativeQC Fast | 37.9017 | 7.7351 [7.7217, 7.7464] | complete; accuracy qualified |
| direct | PBE0 | 96 | GPU4PySCF | 19.7250 | 6.6732 [6.6624, 6.6899] | complete; accuracy qualified |
| direct | PBE0 | 96 | GenerativeQC Standard | 48.7120 | 8.2689 [8.2356, 8.2859] | complete; accuracy qualified |
| direct | B3LYP | 3 | GenerativeQC Fast | 1.7907 | 0.0940 [0.0935, 0.0943] | complete; accuracy qualified |
| direct | B3LYP | 3 | GPU4PySCF | 8.2098 | 1.4518 [1.2663, 2.5561] | complete; accuracy qualified |
| direct | B3LYP | 3 | GenerativeQC Standard | 2.0860 | 0.0915 [0.0913, 0.0923] | complete; accuracy qualified |
| direct | B3LYP | 12 | GenerativeQC Fast | 2.6377 | 0.4123 [0.4111, 0.4236] | complete; accuracy qualified |
| direct | B3LYP | 12 | GPU4PySCF | 8.2075 | 1.2691 [1.2106, 1.5624] | complete; accuracy qualified |
| direct | B3LYP | 12 | GenerativeQC Standard | 2.8622 | 0.4176 [0.4161, 0.4190] | complete; accuracy qualified |
| direct | B3LYP | 48 | GenerativeQC Fast | 13.5154 | 2.5827 [2.5554, 2.5950] | complete; accuracy qualified |
| direct | B3LYP | 48 | GPU4PySCF | 11.8376 | 3.0478 [3.0102, 3.1846] | complete; accuracy qualified |
| direct | B3LYP | 48 | GenerativeQC Standard | 17.3755 | 2.7298 [2.7110, 2.7721] | complete; accuracy qualified |
| direct | B3LYP | 96 | GenerativeQC Fast | 38.5212 | 7.9555 [7.9551, 8.0261] | complete; accuracy qualified |
| direct | B3LYP | 96 | GPU4PySCF | 20.9424 | 6.6690 [6.6643, 6.6691] | complete; accuracy qualified |
| direct | B3LYP | 96 | GenerativeQC Standard | 51.1373 | 8.3763 [8.3690, 8.3903] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GenerativeQC Fast | 1.6560 | 0.0941 [0.0940, 0.0945] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GPU4PySCF | 5.9163 | 0.6607 [0.5730, 0.6754] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GenerativeQC Standard | 1.8271 | 0.0899 [0.0898, 0.0975] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GenerativeQC Fast | 2.3401 | 0.3824 [0.3797, 0.3846] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GPU4PySCF | 5.5690 | 0.8937 [0.7746, 1.0440] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GenerativeQC Standard | 2.5679 | 0.3787 [0.3744, 0.3852] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GenerativeQC Fast | 13.0033 | 2.4065 [2.4032, 2.4257] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GPU4PySCF | 10.4052 | 3.0255 [3.0236, 3.0410] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GenerativeQC Standard | 16.6039 | 2.4386 [2.4231, 2.4719] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GenerativeQC Fast | 36.0299 | 7.3118 [7.3097, 7.3230] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GPU4PySCF | 19.0017 | 7.4726 [7.4700, 7.4726] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GenerativeQC Standard | 30.6385 | 7.3819 [7.3737, 7.3958] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GenerativeQC Fast | 1.8142 | 0.1323 [0.1317, 0.1619] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GPU4PySCF | 6.3173 | 1.3841 [1.2430, 1.4408] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GenerativeQC Standard | 1.7342 | 0.1270 [0.1269, 0.1337] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GenerativeQC Fast | 5.4146 | 0.9164 [0.9138, 0.9729] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GPU4PySCF | 6.1928 | 1.4197 [1.4063, 1.4297] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GenerativeQC Standard | 5.7809 | 0.9193 [0.9181, 0.9405] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GenerativeQC Fast | 45.2238 | 5.2528 [5.2247, 5.2539] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GPU4PySCF | 16.8442 | 5.8278 [5.8247, 5.8817] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GenerativeQC Standard | 22.8190 | 5.2815 [5.2792, 5.2974] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GenerativeQC Fast | 271.3032 | 18.7562 [18.6518, 18.8763] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GPU4PySCF | 32.3201 | 14.8875 [14.8670, 14.9116] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GenerativeQC Standard | 81.4477 | 18.7010 [18.6904, 18.7236] | complete; accuracy qualified |
| direct | M06-2X | 3 | GenerativeQC Fast | 1.9220 | 0.1094 [0.1083, 0.1351] | complete; accuracy failed |
| direct | M06-2X | 3 | GPU4PySCF | 9.1003 | 1.5139 [1.4707, 1.5173] | complete; accuracy failed |
| direct | M06-2X | 3 | GenerativeQC Standard | 2.0552 | 0.1073 [0.1072, 0.1074] | complete; accuracy failed |
| direct | M06-2X | 12 | GenerativeQC Fast | 2.7822 | 0.4396 [0.4389, 0.4449] | complete; accuracy qualified |
| direct | M06-2X | 12 | GPU4PySCF | 9.3934 | 1.2927 [1.2722, 1.4467] | complete; accuracy qualified |
| direct | M06-2X | 12 | GenerativeQC Standard | 3.2097 | 0.4373 [0.4372, 0.4808] | complete; accuracy qualified |
| direct | M06-2X | 48 | GenerativeQC Fast | 14.2022 | 2.6726 [2.6477, 2.6779] | complete; accuracy qualified |
| direct | M06-2X | 48 | GPU4PySCF | 13.9341 | 3.7746 [3.7745, 3.7898] | complete; accuracy qualified |
| direct | M06-2X | 48 | GenerativeQC Standard | 18.5313 | 2.8558 [2.8488, 2.9104] | complete; accuracy qualified |
| direct | M06-2X | 96 | GenerativeQC Fast | 40.7855 | 8.1428 [8.1141, 8.1473] | complete; accuracy qualified |
| direct | M06-2X | 96 | GPU4PySCF | 24.3030 | 8.5734 [8.5609, 8.6325] | complete; accuracy qualified |
| direct | M06-2X | 96 | GenerativeQC Standard | 47.2526 | 8.5810 [8.5760, 8.5991] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GenerativeQC Fast | 3.8828 | 0.3625 [0.3624, 0.3644] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GPU4PySCF | 11.9915 | 2.6118 [2.3832, 2.7528] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GenerativeQC Standard | 5.2311 | 0.3636 [0.3607, 0.3648] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GenerativeQC Fast | 7.4017 | 0.8684 [0.8561, 0.8687] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GPU4PySCF | 15.3895 | 3.9059 [3.8994, 3.9419] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GenerativeQC Standard | 11.9439 | 0.8611 [0.8578, 0.8618] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GenerativeQC Fast | 72.2265 | 6.8790 [6.8152, 6.9118] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GPU4PySCF | 96.7583 | 32.4821 [32.4730, 32.5099] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GenerativeQC Standard | 88.0144 | 7.0358 [7.0265, 7.0516] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GenerativeQC Fast | 306.8957 | 40.1857 [40.1668, 40.2170] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GPU4PySCF | 365.0042 | 117.8861 [117.8565, 117.8973] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GenerativeQC Standard | 638.8891 | 45.1798 [45.1608, 45.2034] | complete; accuracy qualified |
