# Default endpoint timings

Absolute latency; qualification is limited to the sampled static geometries.

| Mode | Method | Atoms | Engine | Cold (s) | Warm median [range] (s) | Status |
|---|---|---:|---|---:|---|---|
| df | HF | 3 | GenerativeQC Fast | 0.5339 | 0.0197 [0.0194, 0.0209] | complete; accuracy failed |
| df | HF | 3 | GPU4PySCF | 2.7429 | 0.2201 [0.2179, 0.2258] | complete; accuracy failed |
| df | HF | 3 | GenerativeQC Standard | 0.5204 | 0.0194 [0.0193, 0.0202] | complete; accuracy failed |
| df | HF | 12 | GenerativeQC Fast | 0.9635 | 0.1201 [0.1199, 0.1211] | complete; accuracy failed |
| df | HF | 12 | GPU4PySCF | 3.1116 | 0.2869 [0.2734, 0.2892] | complete; accuracy failed |
| df | HF | 12 | GenerativeQC Standard | 1.0658 | 0.1179 [0.1165, 0.1198] | complete; accuracy failed |
| df | HF | 48 | GenerativeQC Fast | 14.7139 | 1.1371 [1.1365, 1.1446] | complete; accuracy failed |
| df | HF | 48 | GPU4PySCF | 4.9938 | 1.7050 [1.7025, 1.7468] | complete; accuracy failed |
| df | HF | 48 | GenerativeQC Standard | 14.5164 | 1.1237 [1.1231, 1.1300] | complete; accuracy failed |
| df | HF | 96 | GenerativeQC Fast | 28.7000 | 4.9431 [4.9247, 4.9604] | complete; accuracy failed |
| df | HF | 96 | GPU4PySCF | 19.7038 | 9.9477 [9.9347, 9.9507] | complete; accuracy failed |
| df | HF | 96 | GenerativeQC Standard | 28.6168 | 4.9349 [4.9284, 4.9563] | complete; accuracy failed |
| df | PBE | 3 | GenerativeQC Fast | 1.6739 | 0.0609 [0.0571, 0.0648] | complete; accuracy qualified |
| df | PBE | 3 | GPU4PySCF | 3.0382 | 0.2326 [0.2319, 0.2466] | complete; accuracy qualified |
| df | PBE | 3 | GenerativeQC Standard | 1.5844 | 0.0571 [0.0552, 0.0597] | complete; accuracy qualified |
| df | PBE | 12 | GenerativeQC Fast | 2.3097 | 0.1862 [0.1810, 0.1911] | complete; accuracy qualified |
| df | PBE | 12 | GPU4PySCF | 4.0833 | 0.3722 [0.3689, 0.4004] | complete; accuracy qualified |
| df | PBE | 12 | GenerativeQC Standard | 2.3213 | 0.2249 [0.2213, 0.2258] | complete; accuracy qualified |
| df | PBE | 48 | GenerativeQC Fast | 10.0797 | 2.0701 [2.0696, 2.0720] | complete; accuracy qualified |
| df | PBE | 48 | GPU4PySCF | 6.5459 | 1.4846 [1.4538, 1.4951] | complete; accuracy qualified |
| df | PBE | 48 | GenerativeQC Standard | 10.5986 | 2.3470 [2.3376, 2.3494] | complete; accuracy qualified |
| df | PBE | 96 | GenerativeQC Fast | 15.2058 | 5.5747 [5.5660, 5.6656] | complete; accuracy qualified |
| df | PBE | 96 | GPU4PySCF | 12.6852 | 4.5506 [4.5309, 4.5700] | complete; accuracy qualified |
| df | PBE | 96 | GenerativeQC Standard | 16.1449 | 5.9889 [5.9756, 6.0048] | complete; accuracy qualified |
| df | PBE0 | 3 | GenerativeQC Fast | 1.7820 | 0.0768 [0.0724, 0.0896] | complete; accuracy qualified |
| df | PBE0 | 3 | GPU4PySCF | 3.8483 | 0.3029 [0.2946, 0.3111] | complete; accuracy qualified |
| df | PBE0 | 3 | GenerativeQC Standard | 1.6908 | 0.0773 [0.0729, 0.0838] | complete; accuracy qualified |
| df | PBE0 | 12 | GenerativeQC Fast | 2.5199 | 0.2861 [0.2860, 0.2939] | complete; accuracy qualified |
| df | PBE0 | 12 | GPU4PySCF | 3.7504 | 0.4591 [0.4542, 0.4617] | complete; accuracy qualified |
| df | PBE0 | 12 | GenerativeQC Standard | 2.4322 | 0.2861 [0.2811, 0.2870] | complete; accuracy qualified |
| df | PBE0 | 48 | GenerativeQC Fast | 13.2118 | 4.5789 [4.5752, 4.5832] | complete; accuracy qualified |
| df | PBE0 | 48 | GPU4PySCF | 8.0633 | 2.7229 [2.7158, 2.7417] | complete; accuracy qualified |
| df | PBE0 | 48 | GenerativeQC Standard | 13.3428 | 4.6447 [4.6417, 4.6566] | complete; accuracy qualified |
| df | PBE0 | 96 | GenerativeQC Fast | 30.8007 | 11.8547 [11.8318, 11.9142] | complete; accuracy qualified |
| df | PBE0 | 96 | GPU4PySCF | 26.4615 | 13.5651 [13.5557, 13.5986] | complete; accuracy qualified |
| df | PBE0 | 96 | GenerativeQC Standard | 28.4451 | 9.4283 [9.4178, 9.4295] | complete; accuracy qualified |
| df | B3LYP | 3 | GenerativeQC Fast | 1.6735 | 0.0744 [0.0743, 0.0793] | complete; accuracy qualified |
| df | B3LYP | 3 | GPU4PySCF | 3.3458 | 0.2980 [0.2903, 0.3061] | complete; accuracy qualified |
| df | B3LYP | 3 | GenerativeQC Standard | 1.6118 | 0.0746 [0.0739, 0.0784] | complete; accuracy qualified |
| df | B3LYP | 12 | GenerativeQC Fast | 2.4388 | 0.3136 [0.3108, 0.3174] | complete; accuracy qualified |
| df | B3LYP | 12 | GPU4PySCF | 3.7198 | 0.4830 [0.4795, 0.4864] | complete; accuracy qualified |
| df | B3LYP | 12 | GenerativeQC Standard | 2.4412 | 0.3127 [0.3090, 0.3145] | complete; accuracy qualified |
| df | B3LYP | 48 | GenerativeQC Fast | 13.4413 | 4.6966 [4.6914, 4.7202] | complete; accuracy qualified |
| df | B3LYP | 48 | GPU4PySCF | 7.8792 | 2.7054 [2.6997, 2.7939] | complete; accuracy qualified |
| df | B3LYP | 48 | GenerativeQC Standard | 13.5185 | 4.7520 [4.7466, 4.7553] | complete; accuracy qualified |
| df | B3LYP | 96 | GenerativeQC Fast | 28.6173 | 9.6026 [9.5798, 9.6081] | complete; accuracy qualified |
| df | B3LYP | 96 | GPU4PySCF | 27.6120 | 13.4857 [13.4714, 13.5141] | complete; accuracy qualified |
| df | B3LYP | 96 | GenerativeQC Standard | 28.8147 | 9.7215 [9.7122, 9.7501] | complete; accuracy qualified |
| df | r²SCAN | 3 | GenerativeQC Fast | 1.8061 | 0.0882 [0.0881, 0.0932] | complete; accuracy qualified |
| df | r²SCAN | 3 | GPU4PySCF | 3.8433 | 0.2450 [0.2430, 0.2451] | complete; accuracy qualified |
| df | r²SCAN | 3 | GenerativeQC Standard | 1.7878 | 0.0827 [0.0823, 0.0870] | complete; accuracy qualified |
| df | r²SCAN | 12 | GenerativeQC Fast | 2.3620 | 0.2205 [0.2179, 0.2227] | complete; accuracy qualified |
| df | r²SCAN | 12 | GPU4PySCF | 3.6386 | 0.4231 [0.4203, 0.4308] | complete; accuracy qualified |
| df | r²SCAN | 12 | GenerativeQC Standard | 2.4474 | 0.3136 [0.3096, 0.3148] | complete; accuracy qualified |
| df | r²SCAN | 48 | GenerativeQC Fast | 10.7266 | 2.2489 [2.2391, 2.2551] | complete; accuracy qualified |
| df | r²SCAN | 48 | GPU4PySCF | 8.3336 | 2.2544 [2.2470, 2.2554] | complete; accuracy qualified |
| df | r²SCAN | 48 | GenerativeQC Standard | 11.8412 | 2.9024 [2.8922, 2.9199] | complete; accuracy qualified |
| df | r²SCAN | 96 | GenerativeQC Fast | 16.8922 | 6.0408 [6.0210, 6.0698] | complete; accuracy qualified |
| df | r²SCAN | 96 | GPU4PySCF | 15.7472 | 6.5212 [6.5179, 6.5214] | complete; accuracy qualified |
| df | r²SCAN | 96 | GenerativeQC Standard | 18.3314 | 6.9071 [6.9066, 6.9500] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GenerativeQC Fast | 1.6752 | 0.0944 [0.0870, 0.0969] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GPU4PySCF | 3.5374 | 0.2849 [0.2721, 0.3364] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GenerativeQC Standard | 1.6702 | 0.0854 [0.0851, 0.0916] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GenerativeQC Fast | 2.7565 | 0.2846 [0.2843, 0.2863] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GPU4PySCF | 4.1234 | 0.5232 [0.5209, 0.5268] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GenerativeQC Standard | 2.9956 | 0.3898 [0.3855, 0.3904] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GenerativeQC Fast | 28.6881 | 5.4029 [5.3954, 5.4293] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GPU4PySCF | 11.0011 | 3.9612 [3.9533, 4.0462] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GenerativeQC Standard | 31.2111 | 7.4335 [7.4256, 7.4454] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GenerativeQC Fast | 32.5157 | 9.9622 [9.9537, 9.9812] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GPU4PySCF | 26.8006 | 13.0268 [12.9154, 13.1489] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GenerativeQC Standard | 34.2994 | 10.9481 [10.9400, 10.9883] | complete; accuracy qualified |
| df | M06-2X | 3 | GenerativeQC Fast | 1.8295 | 0.0884 [0.0883, 0.0926] | complete; accuracy failed |
| df | M06-2X | 3 | GPU4PySCF | 4.2880 | 0.3098 [0.3097, 0.3307] | complete; accuracy failed |
| df | M06-2X | 3 | GenerativeQC Standard | 1.7241 | 0.0855 [0.0852, 0.0900] | complete; accuracy failed |
| df | M06-2X | 12 | GenerativeQC Fast | 2.5900 | 0.3381 [0.3352, 0.3428] | complete; accuracy qualified |
| df | M06-2X | 12 | GPU4PySCF | 3.8690 | 0.5198 [0.4974, 0.5471] | complete; accuracy qualified |
| df | M06-2X | 12 | GenerativeQC Standard | 2.6764 | 0.3286 [0.3282, 0.3389] | complete; accuracy qualified |
| df | M06-2X | 48 | GenerativeQC Fast | 14.2182 | 4.7975 [4.7966, 4.8133] | complete; accuracy qualified |
| df | M06-2X | 48 | GPU4PySCF | 9.3664 | 3.4797 [3.4781, 3.5407] | complete; accuracy qualified |
| df | M06-2X | 48 | GenerativeQC Standard | 14.3917 | 4.8767 [4.8741, 4.8812] | complete; accuracy qualified |
| df | M06-2X | 96 | GenerativeQC Fast | 31.6577 | 9.9861 [9.9842, 10.0707] | complete; accuracy qualified |
| df | M06-2X | 96 | GPU4PySCF | 31.7230 | 15.4871 [15.4693, 15.5569] | complete; accuracy qualified |
| df | M06-2X | 96 | GenerativeQC Standard | 34.0054 | 12.4129 [12.3941, 12.4238] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GenerativeQC Fast | 3.8165 | 0.3178 [0.3177, 0.3201] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GPU4PySCF | 3.7688 | 0.6679 [0.6332, 0.6734] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GenerativeQC Standard | 3.7021 | 0.3079 [0.3072, 0.3083] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GenerativeQC Fast | 8.1252 | 0.6959 [0.6955, 0.7230] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GPU4PySCF | 10.7528 | 2.6980 [2.6956, 2.7387] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GenerativeQC Standard | 8.0474 | 0.6951 [0.6889, 0.7318] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GenerativeQC Fast | 74.9109 | 7.3610 [7.3551, 7.3698] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GPU4PySCF | 92.0422 | 32.4094 [32.3960, 32.4249] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GenerativeQC Standard | 74.8534 | 7.3580 [7.3557, 7.3670] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GenerativeQC Fast | 312.2191 | 38.3521 [38.3450, 38.4417] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GPU4PySCF | 392.3357 | 135.7226 [135.7071, 135.7712] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GenerativeQC Standard | 317.3515 | 39.0038 [38.9794, 39.0637] | complete; accuracy qualified |
| direct | HF | 3 | GenerativeQC Fast | 1.5362 | 0.0734 [0.0727, 0.0737] | complete; accuracy qualified |
| direct | HF | 3 | GPU4PySCF | 3.2943 | 0.8482 [0.8286, 0.8550] | complete; accuracy qualified |
| direct | HF | 3 | GenerativeQC Standard | 1.4397 | 0.0707 [0.0704, 0.0711] | complete; accuracy qualified |
| direct | HF | 12 | GenerativeQC Fast | 1.6469 | 0.1152 [0.1151, 0.1362] | complete; accuracy qualified |
| direct | HF | 12 | GPU4PySCF | 5.2259 | 1.3359 [1.1298, 1.3909] | complete; accuracy qualified |
| direct | HF | 12 | GenerativeQC Standard | 1.6552 | 0.1177 [0.1172, 0.1309] | complete; accuracy qualified |
| direct | HF | 48 | GenerativeQC Fast | 18.0339 | 1.9105 [1.9086, 2.9728] | complete; accuracy qualified |
| direct | HF | 48 | GPU4PySCF | 8.4555 | 2.0086 [2.0023, 2.2604] | complete; accuracy qualified |
| direct | HF | 48 | GenerativeQC Standard | 6.1589 | 0.7462 [0.7452, 0.7493] | complete; accuracy qualified |
| direct | HF | 96 | GenerativeQC Fast | 62.4238 | 7.2236 [7.2211, 8.6237] | complete; accuracy qualified |
| direct | HF | 96 | GPU4PySCF | 12.2962 | 3.4663 [3.4179, 3.5105] | complete; accuracy qualified |
| direct | HF | 96 | GenerativeQC Standard | 19.3028 | 2.8967 [2.8943, 2.8981] | complete; accuracy qualified |
| direct | PBE | 3 | GenerativeQC Fast | 1.4094 | 0.0492 [0.0492, 0.0495] | complete; accuracy qualified |
| direct | PBE | 3 | GPU4PySCF | 4.1738 | 0.5028 [0.4450, 0.5762] | complete; accuracy qualified |
| direct | PBE | 3 | GenerativeQC Standard | 1.3072 | 0.0478 [0.0478, 0.0482] | complete; accuracy qualified |
| direct | PBE | 12 | GenerativeQC Fast | 1.8214 | 0.2618 [0.2589, 0.2621] | complete; accuracy qualified |
| direct | PBE | 12 | GPU4PySCF | 5.0939 | 0.7364 [0.7149, 0.7928] | complete; accuracy qualified |
| direct | PBE | 12 | GenerativeQC Standard | 1.8368 | 0.2755 [0.2657, 0.2776] | complete; accuracy qualified |
| direct | PBE | 48 | GenerativeQC Fast | 5.8293 | 1.7140 [1.7122, 1.7149] | complete; accuracy qualified |
| direct | PBE | 48 | GPU4PySCF | 9.7857 | 2.3506 [2.3445, 2.4016] | complete; accuracy qualified |
| direct | PBE | 48 | GenerativeQC Standard | 6.4811 | 1.7380 [1.7327, 1.7396] | complete; accuracy qualified |
| direct | PBE | 96 | GenerativeQC Fast | 14.7759 | 5.0898 [5.0638, 5.0968] | complete; accuracy qualified |
| direct | PBE | 96 | GPU4PySCF | 16.2066 | 5.6280 [5.6191, 5.6317] | complete; accuracy qualified |
| direct | PBE | 96 | GenerativeQC Standard | 16.4817 | 5.2865 [5.2795, 5.3341] | complete; accuracy qualified |
| direct | PBE0 | 3 | GenerativeQC Fast | 1.6965 | 0.0789 [0.0781, 0.0798] | complete; accuracy qualified |
| direct | PBE0 | 3 | GPU4PySCF | 6.9027 | 1.1134 [1.0063, 1.1197] | complete; accuracy qualified |
| direct | PBE0 | 3 | GenerativeQC Standard | 1.8740 | 0.0778 [0.0775, 0.0779] | complete; accuracy qualified |
| direct | PBE0 | 12 | GenerativeQC Fast | 2.3182 | 0.3589 [0.3587, 0.3719] | complete; accuracy qualified |
| direct | PBE0 | 12 | GPU4PySCF | 8.2828 | 1.3956 [1.3446, 1.4021] | complete; accuracy qualified |
| direct | PBE0 | 12 | GenerativeQC Standard | 2.9212 | 0.3706 [0.3675, 0.3722] | complete; accuracy qualified |
| direct | PBE0 | 48 | GenerativeQC Fast | 7.7593 | 2.1635 [2.1630, 2.1673] | complete; accuracy qualified |
| direct | PBE0 | 48 | GPU4PySCF | 12.6143 | 3.1153 [2.9881, 3.1803] | complete; accuracy qualified |
| direct | PBE0 | 48 | GenerativeQC Standard | 9.3351 | 2.3935 [2.3810, 2.4001] | complete; accuracy qualified |
| direct | PBE0 | 96 | GenerativeQC Fast | 19.9493 | 6.3900 [6.3868, 6.4532] | complete; accuracy qualified |
| direct | PBE0 | 96 | GPU4PySCF | 20.3168 | 6.7984 [6.7897, 6.8126] | complete; accuracy qualified |
| direct | PBE0 | 96 | GenerativeQC Standard | 20.4351 | 6.7722 [6.7470, 6.7998] | complete; accuracy qualified |
| direct | B3LYP | 3 | GenerativeQC Fast | 1.7097 | 0.0879 [0.0876, 0.0882] | complete; accuracy qualified |
| direct | B3LYP | 3 | GPU4PySCF | 5.4495 | 0.9252 [0.9111, 0.9828] | complete; accuracy qualified |
| direct | B3LYP | 3 | GenerativeQC Standard | 1.8849 | 0.0843 [0.0841, 0.0844] | complete; accuracy qualified |
| direct | B3LYP | 12 | GenerativeQC Fast | 2.3534 | 0.3882 [0.3836, 0.3919] | complete; accuracy qualified |
| direct | B3LYP | 12 | GPU4PySCF | 7.4070 | 1.4940 [1.2695, 1.6085] | complete; accuracy qualified |
| direct | B3LYP | 12 | GenerativeQC Standard | 2.9644 | 0.3965 [0.3953, 0.4126] | complete; accuracy qualified |
| direct | B3LYP | 48 | GenerativeQC Fast | 8.1191 | 2.2923 [2.2854, 2.3027] | complete; accuracy qualified |
| direct | B3LYP | 48 | GPU4PySCF | 12.1098 | 3.0344 [3.0286, 3.1432] | complete; accuracy qualified |
| direct | B3LYP | 48 | GenerativeQC Standard | 9.5400 | 2.4115 [2.4110, 2.4513] | complete; accuracy qualified |
| direct | B3LYP | 96 | GenerativeQC Fast | 19.7249 | 6.3900 [6.3780, 6.4139] | complete; accuracy qualified |
| direct | B3LYP | 96 | GPU4PySCF | 21.8062 | 6.7757 [6.7257, 7.3679] | complete; accuracy qualified |
| direct | B3LYP | 96 | GenerativeQC Standard | 20.5142 | 6.8059 [6.7950, 6.8225] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GenerativeQC Fast | 1.5232 | 0.0852 [0.0851, 0.0856] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GPU4PySCF | 3.6689 | 0.5108 [0.4767, 0.8992] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GenerativeQC Standard | 1.4470 | 0.0824 [0.0810, 0.0849] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GenerativeQC Fast | 1.9624 | 0.3499 [0.3497, 0.3648] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GPU4PySCF | 4.9904 | 0.7993 [0.7530, 0.8709] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GenerativeQC Standard | 1.9869 | 0.3527 [0.3382, 0.3583] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GenerativeQC Fast | 6.7060 | 2.0166 [2.0157, 2.0239] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GPU4PySCF | 10.8561 | 3.0069 [3.0050, 3.0143] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GenerativeQC Standard | 7.0177 | 2.0476 [2.0408, 2.0500] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GenerativeQC Fast | 16.7758 | 5.8772 [5.8438, 5.8849] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GPU4PySCF | 19.1566 | 7.6460 [7.6263, 8.3158] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GenerativeQC Standard | 17.5874 | 5.8549 [5.8548, 5.9079] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GenerativeQC Fast | 1.4409 | 0.1065 [0.1062, 0.1077] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GPU4PySCF | 3.8764 | 0.7656 [0.7454, 0.9544] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GenerativeQC Standard | 1.3954 | 0.1063 [0.1061, 0.1075] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GenerativeQC Fast | 3.6382 | 0.8982 [0.8913, 0.9016] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GPU4PySCF | 6.4045 | 1.4137 [1.3654, 1.8054] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GenerativeQC Standard | 3.7474 | 0.8807 [0.8806, 0.8825] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GenerativeQC Fast | 11.8919 | 4.6034 [4.6034, 4.6166] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GPU4PySCF | 16.9491 | 5.8484 [5.8479, 5.9093] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GenerativeQC Standard | 12.4187 | 4.7204 [4.7196, 4.7360] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GenerativeQC Fast | 35.7372 | 15.3108 [15.2874, 15.3674] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GPU4PySCF | 33.1360 | 15.1064 [15.0985, 15.4616] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GenerativeQC Standard | 37.6138 | 15.6149 [15.6103, 15.6458] | complete; accuracy qualified |
| direct | M06-2X | 3 | GenerativeQC Fast | 1.8861 | 0.1011 [0.1006, 0.1015] | complete; accuracy failed |
| direct | M06-2X | 3 | GPU4PySCF | 6.3329 | 1.1367 [1.0739, 1.1566] | complete; accuracy failed |
| direct | M06-2X | 3 | GenerativeQC Standard | 2.0105 | 0.0999 [0.0992, 0.1001] | complete; accuracy failed |
| direct | M06-2X | 12 | GenerativeQC Fast | 2.4395 | 0.4019 [0.4013, 0.4031] | complete; accuracy qualified |
| direct | M06-2X | 12 | GPU4PySCF | 8.8095 | 1.2884 [1.2805, 1.3150] | complete; accuracy qualified |
| direct | M06-2X | 12 | GenerativeQC Standard | 2.9976 | 0.4130 [0.4093, 0.4202] | complete; accuracy qualified |
| direct | M06-2X | 48 | GenerativeQC Fast | 8.7165 | 2.3469 [2.3444, 2.3583] | complete; accuracy qualified |
| direct | M06-2X | 48 | GPU4PySCF | 13.6854 | 3.9248 [3.7619, 3.9331] | complete; accuracy qualified |
| direct | M06-2X | 48 | GenerativeQC Standard | 10.9003 | 2.5861 [2.5762, 2.5876] | complete; accuracy qualified |
| direct | M06-2X | 96 | GenerativeQC Fast | 21.6121 | 6.5741 [6.5696, 6.5903] | complete; accuracy qualified |
| direct | M06-2X | 96 | GPU4PySCF | 24.9289 | 8.6485 [8.6205, 8.6515] | complete; accuracy qualified |
| direct | M06-2X | 96 | GenerativeQC Standard | 25.1474 | 7.2926 [7.2807, 7.3042] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GenerativeQC Fast | 3.4471 | 0.3420 [0.3420, 0.3422] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GPU4PySCF | 7.8237 | 1.9302 [1.7079, 2.1612] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GenerativeQC Standard | 4.5992 | 0.3472 [0.3466, 0.3479] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GenerativeQC Fast | 6.7392 | 0.8258 [0.8248, 0.8522] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GPU4PySCF | 16.1506 | 4.1823 [4.0890, 4.3433] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GenerativeQC Standard | 10.7946 | 0.8351 [0.8327, 0.8360] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GenerativeQC Fast | 68.0723 | 6.1675 [6.1628, 6.1920] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GPU4PySCF | 97.6426 | 32.5863 [32.5637, 32.6112] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GenerativeQC Standard | 83.9729 | 6.7021 [6.6810, 6.7103] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GenerativeQC Fast | 243.3704 | 27.6787 [27.6704, 27.6880] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GPU4PySCF | 366.9192 | 118.2415 [118.2121, 118.3232] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GenerativeQC Standard | 259.8232 | 28.5253 [28.5221, 28.5957] | complete; accuracy qualified |
