# Default endpoint timings

Absolute latency; qualification is limited to the sampled static geometries.

| Mode | Method | Atoms | Engine | Cold (s) | Warm median [range] (s) | Status |
|---|---|---:|---|---:|---|---|
| df | HF | 3 | GenerativeQC Fast | 0.4879 | 0.0224 [0.0188, 0.0225] | complete; accuracy failed |
| df | HF | 3 | GPU4PySCF | 4.0219 | 0.2554 [0.2411, 0.3046] | complete; accuracy failed |
| df | HF | 3 | GenerativeQC Standard | 0.5288 | 0.0184 [0.0182, 0.0188] | complete; accuracy failed |
| df | HF | 12 | GenerativeQC Fast | 0.9303 | 0.1119 [0.1118, 0.1194] | complete; accuracy failed |
| df | HF | 12 | GPU4PySCF | 3.4906 | 0.2827 [0.2727, 0.2943] | complete; accuracy failed |
| df | HF | 12 | GenerativeQC Standard | 1.0159 | 0.1126 [0.1112, 0.1408] | complete; accuracy failed |
| df | HF | 48 | GenerativeQC Fast | 3.6326 | 0.9578 [0.9568, 0.9920] | complete; accuracy failed |
| df | HF | 48 | GPU4PySCF | 6.3070 | 1.7419 [1.7375, 1.8073] | complete; accuracy failed |
| df | HF | 48 | GenerativeQC Standard | 3.5919 | 0.9369 [0.9364, 0.9506] | complete; accuracy failed |
| df | HF | 96 | GenerativeQC Fast | 22.2910 | 4.4018 [4.4012, 4.4056] | complete; accuracy failed |
| df | HF | 96 | GPU4PySCF | 20.0258 | 10.0878 [10.0715, 10.0885] | complete; accuracy failed |
| df | HF | 96 | GenerativeQC Standard | 21.5478 | 4.2575 [4.2551, 4.2717] | complete; accuracy failed |
| df | PBE | 3 | GenerativeQC Fast | 1.7669 | 0.0656 [0.0622, 0.0990] | complete; accuracy qualified |
| df | PBE | 3 | GPU4PySCF | 4.0288 | 0.2617 [0.2493, 0.2665] | complete; accuracy qualified |
| df | PBE | 3 | GenerativeQC Standard | 1.7816 | 0.0629 [0.0604, 0.0631] | complete; accuracy qualified |
| df | PBE | 12 | GenerativeQC Fast | 2.3243 | 0.1929 [0.1882, 0.2302] | complete; accuracy qualified |
| df | PBE | 12 | GPU4PySCF | 4.3316 | 0.4102 [0.4091, 0.5443] | complete; accuracy qualified |
| df | PBE | 12 | GenerativeQC Standard | 2.3074 | 0.2240 [0.2226, 0.2285] | complete; accuracy qualified |
| df | PBE | 48 | GenerativeQC Fast | 5.6739 | 1.6077 [1.6052, 1.6270] | complete; accuracy qualified |
| df | PBE | 48 | GPU4PySCF | 7.3315 | 1.4653 [1.4619, 1.5049] | complete; accuracy qualified |
| df | PBE | 48 | GenerativeQC Standard | 6.2020 | 1.8498 [1.8459, 1.8824] | complete; accuracy qualified |
| df | PBE | 96 | GenerativeQC Fast | 15.0089 | 5.4779 [5.4717, 5.5120] | complete; accuracy qualified |
| df | PBE | 96 | GPU4PySCF | 12.5023 | 4.5193 [4.5189, 4.5546] | complete; accuracy qualified |
| df | PBE | 96 | GenerativeQC Standard | 16.1642 | 5.9923 [5.9723, 6.0000] | complete; accuracy qualified |
| df | PBE0 | 3 | GenerativeQC Fast | 1.6036 | 0.0731 [0.0710, 0.0770] | complete; accuracy qualified |
| df | PBE0 | 3 | GPU4PySCF | 4.7181 | 0.3850 [0.3458, 0.4579] | complete; accuracy qualified |
| df | PBE0 | 3 | GenerativeQC Standard | 1.7447 | 0.0719 [0.0692, 0.0748] | complete; accuracy qualified |
| df | PBE0 | 12 | GenerativeQC Fast | 2.4229 | 0.2945 [0.2923, 0.3405] | complete; accuracy qualified |
| df | PBE0 | 12 | GPU4PySCF | 4.3926 | 0.6224 [0.5395, 0.6555] | complete; accuracy qualified |
| df | PBE0 | 12 | GenerativeQC Standard | 2.3477 | 0.2887 [0.2880, 0.3209] | complete; accuracy qualified |
| df | PBE0 | 48 | GenerativeQC Fast | 7.6164 | 2.6348 [2.6261, 2.6458] | complete; accuracy qualified |
| df | PBE0 | 48 | GPU4PySCF | 8.8613 | 2.7715 [2.7708, 2.8301] | complete; accuracy qualified |
| df | PBE0 | 48 | GenerativeQC Standard | 7.3897 | 2.5966 [2.5955, 2.6088] | complete; accuracy qualified |
| df | PBE0 | 96 | GenerativeQC Fast | 28.3839 | 9.4853 [9.4763, 9.5208] | complete; accuracy qualified |
| df | PBE0 | 96 | GPU4PySCF | 26.8330 | 13.6955 [13.6844, 13.7275] | complete; accuracy qualified |
| df | PBE0 | 96 | GenerativeQC Standard | 27.7314 | 9.2653 [9.2503, 9.2679] | complete; accuracy qualified |
| df | B3LYP | 3 | GenerativeQC Fast | 1.5767 | 0.0775 [0.0747, 0.0785] | complete; accuracy qualified |
| df | B3LYP | 3 | GPU4PySCF | 4.4503 | 0.3423 [0.3237, 0.3568] | complete; accuracy qualified |
| df | B3LYP | 3 | GenerativeQC Standard | 1.5764 | 0.0815 [0.0730, 0.1100] | complete; accuracy qualified |
| df | B3LYP | 12 | GenerativeQC Fast | 2.6764 | 0.3398 [0.3218, 0.3508] | complete; accuracy qualified |
| df | B3LYP | 12 | GPU4PySCF | 4.6747 | 0.6293 [0.5203, 0.6334] | complete; accuracy qualified |
| df | B3LYP | 12 | GenerativeQC Standard | 2.3945 | 0.3268 [0.3102, 0.3648] | complete; accuracy qualified |
| df | B3LYP | 48 | GenerativeQC Fast | 7.3860 | 2.6410 [2.6383, 2.6898] | complete; accuracy qualified |
| df | B3LYP | 48 | GPU4PySCF | 8.9859 | 2.8306 [2.8199, 2.9132] | complete; accuracy qualified |
| df | B3LYP | 48 | GenerativeQC Standard | 7.5373 | 2.6771 [2.6764, 2.6798] | complete; accuracy qualified |
| df | B3LYP | 96 | GenerativeQC Fast | 28.1609 | 9.4875 [9.4863, 9.5307] | complete; accuracy qualified |
| df | B3LYP | 96 | GPU4PySCF | 27.8817 | 13.5436 [13.5318, 13.6111] | complete; accuracy qualified |
| df | B3LYP | 96 | GenerativeQC Standard | 28.3263 | 9.5630 [9.5541, 9.5685] | complete; accuracy qualified |
| df | r²SCAN | 3 | GenerativeQC Fast | 1.6511 | 0.1005 [0.0909, 0.1118] | complete; accuracy qualified |
| df | r²SCAN | 3 | GPU4PySCF | 4.2316 | 0.2591 [0.2450, 0.2637] | complete; accuracy qualified |
| df | r²SCAN | 3 | GenerativeQC Standard | 1.9505 | 0.0830 [0.0817, 0.0899] | complete; accuracy qualified |
| df | r²SCAN | 12 | GenerativeQC Fast | 2.4685 | 0.2249 [0.2205, 0.2336] | complete; accuracy qualified |
| df | r²SCAN | 12 | GPU4PySCF | 4.4222 | 0.4043 [0.4028, 0.4078] | complete; accuracy qualified |
| df | r²SCAN | 12 | GenerativeQC Standard | 2.4526 | 0.3149 [0.3098, 0.3161] | complete; accuracy qualified |
| df | r²SCAN | 48 | GenerativeQC Fast | 6.3646 | 1.8131 [1.8055, 1.8272] | complete; accuracy qualified |
| df | r²SCAN | 48 | GPU4PySCF | 8.8383 | 2.3000 [2.2506, 2.3078] | complete; accuracy qualified |
| df | r²SCAN | 48 | GenerativeQC Standard | 6.9596 | 2.3509 [2.3461, 2.3758] | complete; accuracy qualified |
| df | r²SCAN | 96 | GenerativeQC Fast | 16.3988 | 5.8962 [5.8841, 5.9232] | complete; accuracy qualified |
| df | r²SCAN | 96 | GPU4PySCF | 15.5892 | 6.5280 [6.4847, 6.6141] | complete; accuracy qualified |
| df | r²SCAN | 96 | GenerativeQC Standard | 18.5051 | 6.9971 [6.9942, 7.0208] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GenerativeQC Fast | 1.6236 | 0.0990 [0.0914, 0.1143] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GPU4PySCF | 4.4199 | 0.2928 [0.2894, 0.3069] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GenerativeQC Standard | 1.7548 | 0.0908 [0.0872, 0.0959] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GenerativeQC Fast | 2.7695 | 0.2897 [0.2877, 0.2929] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GPU4PySCF | 6.3012 | 0.5527 [0.5428, 0.6468] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GenerativeQC Standard | 2.8817 | 0.3851 [0.3804, 0.4116] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GenerativeQC Fast | 10.1201 | 3.6281 [3.6179, 3.6431] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GPU4PySCF | 10.5677 | 3.8579 [3.8524, 3.8900] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GenerativeQC Standard | 11.0009 | 4.2000 [4.1997, 4.2270] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GenerativeQC Fast | 31.4578 | 9.7389 [9.7226, 9.7553] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GPU4PySCF | 25.5697 | 12.4221 [12.4172, 12.4351] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GenerativeQC Standard | 34.1962 | 10.8974 [10.8964, 10.9513] | complete; accuracy qualified |
| df | M06-2X | 3 | GenerativeQC Fast | 1.7370 | 0.0970 [0.0935, 0.0975] | complete; accuracy failed |
| df | M06-2X | 3 | GPU4PySCF | 3.9058 | 0.2983 [0.2962, 0.3070] | complete; accuracy failed |
| df | M06-2X | 3 | GenerativeQC Standard | 1.7340 | 0.0887 [0.0884, 0.0899] | complete; accuracy failed |
| df | M06-2X | 12 | GenerativeQC Fast | 2.6347 | 0.3492 [0.3477, 0.4335] | complete; accuracy qualified |
| df | M06-2X | 12 | GPU4PySCF | 5.7313 | 0.5166 [0.5108, 0.5453] | complete; accuracy qualified |
| df | M06-2X | 12 | GenerativeQC Standard | 2.5606 | 0.3415 [0.3382, 0.3452] | complete; accuracy qualified |
| df | M06-2X | 48 | GenerativeQC Fast | 8.6707 | 2.8541 [2.8500, 2.8601] | complete; accuracy qualified |
| df | M06-2X | 48 | GPU4PySCF | 10.2034 | 3.5797 [3.4967, 3.6490] | complete; accuracy qualified |
| df | M06-2X | 48 | GenerativeQC Standard | 8.5358 | 2.9123 [2.8713, 2.9416] | complete; accuracy qualified |
| df | M06-2X | 96 | GenerativeQC Fast | 32.1643 | 10.1693 [10.1591, 10.1944] | complete; accuracy qualified |
| df | M06-2X | 96 | GPU4PySCF | 30.5548 | 15.1585 [15.1562, 15.2116] | complete; accuracy qualified |
| df | M06-2X | 96 | GenerativeQC Standard | 30.9365 | 9.8050 [9.7951, 9.8668] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GenerativeQC Fast | 3.7166 | 0.3283 [0.3156, 0.3294] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GPU4PySCF | 6.1068 | 0.6409 [0.6341, 0.6754] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GenerativeQC Standard | 3.7873 | 0.3100 [0.3100, 0.3106] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GenerativeQC Fast | 8.1479 | 0.7030 [0.6892, 0.7303] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GPU4PySCF | 11.2598 | 2.6664 [2.6527, 2.6810] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GenerativeQC Standard | 8.0298 | 0.7024 [0.7024, 0.7216] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GenerativeQC Fast | 69.8505 | 5.9215 [5.9182, 5.9269] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GPU4PySCF | 90.7634 | 32.1916 [32.1603, 32.2255] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GenerativeQC Standard | 68.3673 | 5.7909 [5.7891, 5.7953] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GenerativeQC Fast | 279.8099 | 38.4890 [38.4606, 38.5020] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GPU4PySCF | 385.6021 | 133.1264 [133.1219, 133.1555] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GenerativeQC Standard | 280.4947 | 38.3236 [38.2925, 38.3617] | complete; accuracy qualified |
| direct | HF | 3 | GenerativeQC Fast | 1.5695 | 0.0736 [0.0725, 0.0737] | complete; accuracy qualified |
| direct | HF | 3 | GPU4PySCF | 4.0541 | 1.1605 [1.0671, 1.2518] | complete; accuracy qualified |
| direct | HF | 3 | GenerativeQC Standard | 1.3982 | 0.0715 [0.0712, 0.0716] | complete; accuracy qualified |
| direct | HF | 12 | GenerativeQC Fast | 1.5999 | 0.1176 [0.1170, 0.1306] | complete; accuracy qualified |
| direct | HF | 12 | GPU4PySCF | 6.7690 | 1.2961 [1.1594, 1.4226] | complete; accuracy qualified |
| direct | HF | 12 | GenerativeQC Standard | 1.7755 | 0.1192 [0.1192, 0.1372] | complete; accuracy qualified |
| direct | HF | 48 | GenerativeQC Fast | 6.1444 | 0.7536 [0.7529, 0.7574] | complete; accuracy qualified |
| direct | HF | 48 | GPU4PySCF | 8.1836 | 1.9963 [1.9764, 2.4396] | complete; accuracy qualified |
| direct | HF | 48 | GenerativeQC Standard | 6.2826 | 0.7488 [0.7478, 0.7502] | complete; accuracy qualified |
| direct | HF | 96 | GenerativeQC Fast | 19.0792 | 2.8818 [2.8815, 2.8837] | complete; accuracy qualified |
| direct | HF | 96 | GPU4PySCF | 12.8393 | 3.5770 [3.5567, 3.9103] | complete; accuracy qualified |
| direct | HF | 96 | GenerativeQC Standard | 19.3487 | 2.9270 [2.9239, 2.9298] | complete; accuracy qualified |
| direct | PBE | 3 | GenerativeQC Fast | 1.4970 | 0.0491 [0.0488, 0.0492] | complete; accuracy qualified |
| direct | PBE | 3 | GPU4PySCF | 3.6442 | 0.6374 [0.5272, 0.6451] | complete; accuracy qualified |
| direct | PBE | 3 | GenerativeQC Standard | 1.5414 | 0.0499 [0.0489, 0.0710] | complete; accuracy qualified |
| direct | PBE | 12 | GenerativeQC Fast | 1.8642 | 0.2615 [0.2612, 0.2702] | complete; accuracy qualified |
| direct | PBE | 12 | GPU4PySCF | 7.1811 | 0.9775 [0.7241, 0.9812] | complete; accuracy qualified |
| direct | PBE | 12 | GenerativeQC Standard | 1.8066 | 0.2735 [0.2657, 0.2770] | complete; accuracy qualified |
| direct | PBE | 48 | GenerativeQC Fast | 6.1822 | 1.7239 [1.7203, 1.7479] | complete; accuracy qualified |
| direct | PBE | 48 | GPU4PySCF | 11.6272 | 2.4370 [2.3996, 2.5454] | complete; accuracy qualified |
| direct | PBE | 48 | GenerativeQC Standard | 6.5948 | 1.7404 [1.7399, 1.7431] | complete; accuracy qualified |
| direct | PBE | 96 | GenerativeQC Fast | 15.1383 | 5.2281 [5.1983, 5.2469] | complete; accuracy qualified |
| direct | PBE | 96 | GPU4PySCF | 16.0438 | 5.6021 [5.5924, 5.6387] | complete; accuracy qualified |
| direct | PBE | 96 | GenerativeQC Standard | 16.3039 | 5.2337 [5.2260, 5.2495] | complete; accuracy qualified |
| direct | PBE0 | 3 | GenerativeQC Fast | 1.9174 | 0.0775 [0.0774, 0.0783] | complete; accuracy qualified |
| direct | PBE0 | 3 | GPU4PySCF | 8.5777 | 0.9839 [0.9374, 1.5294] | complete; accuracy qualified |
| direct | PBE0 | 3 | GenerativeQC Standard | 1.8659 | 0.0768 [0.0768, 0.0773] | complete; accuracy qualified |
| direct | PBE0 | 12 | GenerativeQC Fast | 2.3412 | 0.3600 [0.3586, 0.3601] | complete; accuracy qualified |
| direct | PBE0 | 12 | GPU4PySCF | 13.3307 | 1.8409 [1.6627, 2.4821] | complete; accuracy qualified |
| direct | PBE0 | 12 | GenerativeQC Standard | 2.8924 | 0.3707 [0.3704, 0.3729] | complete; accuracy qualified |
| direct | PBE0 | 48 | GenerativeQC Fast | 8.0291 | 2.1870 [2.1807, 2.1872] | complete; accuracy qualified |
| direct | PBE0 | 48 | GPU4PySCF | 14.7600 | 3.5391 [3.2114, 3.7046] | complete; accuracy qualified |
| direct | PBE0 | 48 | GenerativeQC Standard | 9.2903 | 2.3569 [2.3556, 2.3720] | complete; accuracy qualified |
| direct | PBE0 | 96 | GenerativeQC Fast | 19.2204 | 6.1536 [6.1504, 6.1954] | complete; accuracy qualified |
| direct | PBE0 | 96 | GPU4PySCF | 20.0565 | 6.8473 [6.8264, 6.8535] | complete; accuracy qualified |
| direct | PBE0 | 96 | GenerativeQC Standard | 20.0898 | 6.6490 [6.6478, 6.6751] | complete; accuracy qualified |
| direct | B3LYP | 3 | GenerativeQC Fast | 1.7790 | 0.0855 [0.0852, 0.0858] | complete; accuracy qualified |
| direct | B3LYP | 3 | GPU4PySCF | 7.5831 | 0.9459 [0.9234, 1.3116] | complete; accuracy qualified |
| direct | B3LYP | 3 | GenerativeQC Standard | 1.9891 | 0.0849 [0.0846, 0.0855] | complete; accuracy qualified |
| direct | B3LYP | 12 | GenerativeQC Fast | 2.5182 | 0.3846 [0.3831, 0.3927] | complete; accuracy qualified |
| direct | B3LYP | 12 | GPU4PySCF | 7.1230 | 1.2403 [1.2398, 1.5150] | complete; accuracy qualified |
| direct | B3LYP | 12 | GenerativeQC Standard | 2.9958 | 0.3958 [0.3951, 0.3994] | complete; accuracy qualified |
| direct | B3LYP | 48 | GenerativeQC Fast | 7.8501 | 2.2228 [2.2183, 2.2229] | complete; accuracy qualified |
| direct | B3LYP | 48 | GPU4PySCF | 13.3816 | 3.0908 [3.0344, 3.1796] | complete; accuracy qualified |
| direct | B3LYP | 48 | GenerativeQC Standard | 9.2603 | 2.3986 [2.3942, 2.4010] | complete; accuracy qualified |
| direct | B3LYP | 96 | GenerativeQC Fast | 19.6125 | 6.4058 [6.3965, 6.4188] | complete; accuracy qualified |
| direct | B3LYP | 96 | GPU4PySCF | 22.2154 | 6.8100 [6.7737, 6.8244] | complete; accuracy qualified |
| direct | B3LYP | 96 | GenerativeQC Standard | 21.0836 | 7.0273 [7.0095, 7.0291] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GenerativeQC Fast | 1.5244 | 0.0841 [0.0841, 0.0844] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GPU4PySCF | 4.8347 | 0.4850 [0.4817, 0.5210] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GenerativeQC Standard | 1.3817 | 0.0823 [0.0823, 0.0829] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GenerativeQC Fast | 2.0609 | 0.3522 [0.3488, 0.3566] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GPU4PySCF | 5.4702 | 0.7523 [0.7328, 0.7536] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GenerativeQC Standard | 1.9823 | 0.3366 [0.3362, 0.3399] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GenerativeQC Fast | 6.7436 | 2.0351 [2.0337, 2.0393] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GPU4PySCF | 12.9348 | 3.1075 [3.0804, 3.4731] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GenerativeQC Standard | 7.2890 | 2.0607 [2.0605, 2.0637] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GenerativeQC Fast | 16.5204 | 5.8060 [5.7986, 5.8224] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GPU4PySCF | 18.5162 | 7.4350 [7.4319, 7.4383] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GenerativeQC Standard | 17.2934 | 5.7747 [5.7568, 5.8234] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GenerativeQC Fast | 1.5075 | 0.1059 [0.1059, 0.1063] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GPU4PySCF | 4.2538 | 0.9800 [0.7480, 1.0177] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GenerativeQC Standard | 1.4094 | 0.1032 [0.1027, 0.1045] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GenerativeQC Fast | 3.6791 | 0.8888 [0.8856, 0.8943] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GPU4PySCF | 8.3438 | 1.5417 [1.3872, 1.5699] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GenerativeQC Standard | 3.8010 | 0.8926 [0.8926, 0.8934] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GenerativeQC Fast | 11.9099 | 4.6545 [4.6383, 4.6568] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GPU4PySCF | 18.5553 | 6.0393 [5.9922, 6.0827] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GenerativeQC Standard | 12.4012 | 4.7341 [4.7226, 4.7451] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GenerativeQC Fast | 36.4598 | 15.6589 [15.6426, 15.7074] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GPU4PySCF | 32.4865 | 14.8723 [14.8648, 14.8900] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GenerativeQC Standard | 37.5327 | 15.5975 [15.5873, 15.6281] | complete; accuracy qualified |
| direct | M06-2X | 3 | GenerativeQC Fast | 1.7827 | 0.1019 [0.1014, 0.1021] | complete; accuracy failed |
| direct | M06-2X | 3 | GPU4PySCF | 6.6810 | 2.2233 [1.0653, 2.4602] | complete; accuracy failed |
| direct | M06-2X | 3 | GenerativeQC Standard | 2.0967 | 0.0981 [0.0979, 0.0982] | complete; accuracy failed |
| direct | M06-2X | 12 | GenerativeQC Fast | 2.5051 | 0.3994 [0.3973, 0.4069] | complete; accuracy qualified |
| direct | M06-2X | 12 | GPU4PySCF | 12.7372 | 1.7570 [1.5809, 2.1738] | complete; accuracy qualified |
| direct | M06-2X | 12 | GenerativeQC Standard | 3.2502 | 0.4169 [0.4146, 0.4169] | complete; accuracy qualified |
| direct | M06-2X | 48 | GenerativeQC Fast | 9.1574 | 2.3837 [2.3738, 2.3861] | complete; accuracy qualified |
| direct | M06-2X | 48 | GPU4PySCF | 17.6367 | 4.1973 [3.8483, 4.2338] | complete; accuracy qualified |
| direct | M06-2X | 48 | GenerativeQC Standard | 10.6079 | 2.5295 [2.5233, 2.5370] | complete; accuracy qualified |
| direct | M06-2X | 96 | GenerativeQC Fast | 21.5600 | 6.6301 [6.6299, 6.6701] | complete; accuracy qualified |
| direct | M06-2X | 96 | GPU4PySCF | 25.3369 | 8.8455 [8.8450, 8.8757] | complete; accuracy qualified |
| direct | M06-2X | 96 | GenerativeQC Standard | 23.9204 | 7.1372 [7.1023, 7.1404] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GenerativeQC Fast | 3.3887 | 0.3464 [0.3462, 0.3473] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GPU4PySCF | 7.9662 | 1.8191 [1.7432, 2.0528] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GenerativeQC Standard | 4.4112 | 0.3387 [0.3387, 0.3388] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GenerativeQC Fast | 6.6935 | 0.8219 [0.8188, 0.8706] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GPU4PySCF | 15.6388 | 4.1065 [3.8973, 4.1449] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GenerativeQC Standard | 10.8704 | 0.8322 [0.8316, 0.8435] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GenerativeQC Fast | 67.2870 | 6.1280 [6.1253, 6.2102] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GPU4PySCF | 96.9044 | 32.3346 [32.3266, 32.3874] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GenerativeQC Standard | 83.6581 | 6.6949 [6.6489, 6.7069] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GenerativeQC Fast | 280.1461 | 27.6339 [27.6009, 27.6963] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GPU4PySCF | 372.0507 | 119.9128 [119.8934, 119.9153] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GenerativeQC Standard | 288.9541 | 29.2097 [29.1376, 29.2422] | complete; accuracy qualified |
