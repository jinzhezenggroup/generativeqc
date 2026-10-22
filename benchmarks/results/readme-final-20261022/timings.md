# Default endpoint timings

Absolute latency; qualification is limited to the sampled static geometries.

| Mode | Method | Atoms | Engine | Cold (s) | Warm median [range] (s) | Status |
|---|---|---:|---|---:|---|---|
| df | HF | 3 | GenerativeQC Fast | 0.8434 | 0.0194 [0.0191, 0.0204] | complete; accuracy failed |
| df | HF | 3 | GPU4PySCF | 4.1535 | 0.2262 [0.2226, 0.2353] | complete; accuracy failed |
| df | HF | 3 | GenerativeQC Standard | 0.5075 | 0.0193 [0.0188, 0.0209] | complete; accuracy failed |
| df | HF | 12 | GenerativeQC Fast | 1.0282 | 0.1189 [0.1187, 0.1300] | complete; accuracy failed |
| df | HF | 12 | GPU4PySCF | 4.9199 | 0.3144 [0.2920, 0.3771] | complete; accuracy failed |
| df | HF | 12 | GenerativeQC Standard | 1.1936 | 0.1216 [0.1212, 0.1515] | complete; accuracy failed |
| df | HF | 48 | GenerativeQC Fast | 13.0272 | 1.1700 [1.1672, 1.1837] | complete; accuracy failed |
| df | HF | 48 | GPU4PySCF | 6.3171 | 1.7275 [1.7131, 1.7317] | complete; accuracy failed |
| df | HF | 48 | GenerativeQC Standard | 12.6918 | 1.1427 [1.1356, 1.1503] | complete; accuracy failed |
| df | HF | 96 | GenerativeQC Fast | 97.9755 | 4.9181 [4.9148, 4.9340] | complete; accuracy failed |
| df | HF | 96 | GPU4PySCF | 20.4482 | 10.2636 [10.2567, 10.2671] | complete; accuracy failed |
| df | HF | 96 | GenerativeQC Standard | 99.0922 | 4.9813 [4.9761, 4.9926] | complete; accuracy failed |
| df | PBE | 3 | GenerativeQC Fast | 2.8117 | 0.0739 [0.0738, 0.0807] | complete; accuracy qualified |
| df | PBE | 3 | GPU4PySCF | 7.4489 | 0.2585 [0.2456, 0.3590] | complete; accuracy qualified |
| df | PBE | 3 | GenerativeQC Standard | 2.5922 | 0.0733 [0.0728, 0.0814] | complete; accuracy qualified |
| df | PBE | 12 | GenerativeQC Fast | 3.3326 | 0.2146 [0.2131, 0.2197] | complete; accuracy qualified |
| df | PBE | 12 | GPU4PySCF | 8.0929 | 0.5711 [0.4119, 0.6328] | complete; accuracy qualified |
| df | PBE | 12 | GenerativeQC Standard | 3.7782 | 0.2651 [0.2616, 0.3562] | complete; accuracy qualified |
| df | PBE | 48 | GenerativeQC Fast | 11.4242 | 2.2850 [2.2850, 2.2942] | complete; accuracy qualified |
| df | PBE | 48 | GPU4PySCF | 7.7555 | 1.5014 [1.5010, 1.5027] | complete; accuracy qualified |
| df | PBE | 48 | GenerativeQC Standard | 13.8468 | 2.6140 [2.6023, 2.6207] | complete; accuracy qualified |
| df | PBE | 96 | GenerativeQC Fast | 38.9447 | 6.6589 [6.6391, 6.6923] | complete; accuracy qualified |
| df | PBE | 96 | GPU4PySCF | 12.9584 | 4.6370 [4.6305, 4.6831] | complete; accuracy qualified |
| df | PBE | 96 | GenerativeQC Standard | 44.4029 | 7.0917 [7.0232, 7.1081] | complete; accuracy qualified |
| df | PBE0 | 3 | GenerativeQC Fast | 2.7529 | 0.1034 [0.0992, 0.1274] | complete; accuracy qualified |
| df | PBE0 | 3 | GPU4PySCF | 7.0339 | 0.3539 [0.3327, 0.3853] | complete; accuracy qualified |
| df | PBE0 | 3 | GenerativeQC Standard | 2.6678 | 0.0876 [0.0861, 0.0923] | complete; accuracy qualified |
| df | PBE0 | 12 | GenerativeQC Fast | 3.7944 | 0.2886 [0.2816, 0.3555] | complete; accuracy qualified |
| df | PBE0 | 12 | GPU4PySCF | 6.5278 | 0.6205 [0.5043, 0.6939] | complete; accuracy qualified |
| df | PBE0 | 12 | GenerativeQC Standard | 3.9962 | 0.3283 [0.3264, 0.3339] | complete; accuracy qualified |
| df | PBE0 | 48 | GenerativeQC Fast | 17.2333 | 4.6254 [4.6128, 4.6294] | complete; accuracy qualified |
| df | PBE0 | 48 | GPU4PySCF | 9.2587 | 2.7552 [2.7423, 2.9350] | complete; accuracy qualified |
| df | PBE0 | 48 | GenerativeQC Standard | 17.4314 | 4.9080 [4.8998, 4.9451] | complete; accuracy qualified |
| df | PBE0 | 96 | GenerativeQC Fast | 60.7926 | 11.2225 [11.1630, 11.4674] | complete; accuracy qualified |
| df | PBE0 | 96 | GPU4PySCF | 26.7263 | 13.6866 [13.6776, 13.7699] | complete; accuracy qualified |
| df | PBE0 | 96 | GenerativeQC Standard | 60.5627 | 11.4161 [11.3506, 11.4387] | complete; accuracy qualified |
| df | B3LYP | 3 | GenerativeQC Fast | 3.0405 | 0.1069 [0.0952, 0.1314] | complete; accuracy qualified |
| df | B3LYP | 3 | GPU4PySCF | 6.2799 | 0.4824 [0.3332, 0.5339] | complete; accuracy qualified |
| df | B3LYP | 3 | GenerativeQC Standard | 2.7258 | 0.1007 [0.0901, 0.1027] | complete; accuracy qualified |
| df | B3LYP | 12 | GenerativeQC Fast | 3.9638 | 0.2868 [0.2810, 0.3117] | complete; accuracy qualified |
| df | B3LYP | 12 | GPU4PySCF | 6.1298 | 0.5152 [0.4999, 0.8461] | complete; accuracy qualified |
| df | B3LYP | 12 | GenerativeQC Standard | 4.0778 | 0.3613 [0.3485, 0.4417] | complete; accuracy qualified |
| df | B3LYP | 48 | GenerativeQC Fast | 17.7747 | 4.6602 [4.6485, 4.6637] | complete; accuracy qualified |
| df | B3LYP | 48 | GPU4PySCF | 8.4559 | 2.7808 [2.6848, 2.8446] | complete; accuracy qualified |
| df | B3LYP | 48 | GenerativeQC Standard | 18.4888 | 5.0931 [5.0689, 5.1048] | complete; accuracy qualified |
| df | B3LYP | 96 | GenerativeQC Fast | 62.3360 | 11.1868 [11.1421, 11.2489] | complete; accuracy qualified |
| df | B3LYP | 96 | GPU4PySCF | 27.0696 | 13.1866 [13.1690, 13.1872] | complete; accuracy qualified |
| df | B3LYP | 96 | GenerativeQC Standard | 63.3634 | 11.7453 [11.7341, 11.7593] | complete; accuracy qualified |
| df | r²SCAN | 3 | GenerativeQC Fast | 2.6109 | 0.1256 [0.1097, 0.1896] | complete; accuracy qualified |
| df | r²SCAN | 3 | GPU4PySCF | 5.6289 | 0.3523 [0.2584, 0.5383] | complete; accuracy qualified |
| df | r²SCAN | 3 | GenerativeQC Standard | 2.9159 | 0.1467 [0.1197, 0.1956] | complete; accuracy qualified |
| df | r²SCAN | 12 | GenerativeQC Fast | 3.4463 | 0.2536 [0.2533, 0.2565] | complete; accuracy qualified |
| df | r²SCAN | 12 | GPU4PySCF | 5.8286 | 0.5017 [0.4567, 0.5967] | complete; accuracy qualified |
| df | r²SCAN | 12 | GenerativeQC Standard | 4.3133 | 0.3547 [0.3535, 0.3554] | complete; accuracy qualified |
| df | r²SCAN | 48 | GenerativeQC Fast | 12.2074 | 2.5035 [2.4969, 2.5151] | complete; accuracy qualified |
| df | r²SCAN | 48 | GPU4PySCF | 8.6731 | 2.2556 [2.2514, 2.2584] | complete; accuracy qualified |
| df | r²SCAN | 48 | GenerativeQC Standard | 14.8510 | 3.0828 [3.0818, 3.1166] | complete; accuracy qualified |
| df | r²SCAN | 96 | GenerativeQC Fast | 41.0105 | 7.0426 [7.0380, 7.0554] | complete; accuracy qualified |
| df | r²SCAN | 96 | GPU4PySCF | 15.9727 | 6.4140 [6.4099, 6.4166] | complete; accuracy qualified |
| df | r²SCAN | 96 | GenerativeQC Standard | 47.9955 | 7.9435 [7.9383, 8.0276] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GenerativeQC Fast | 2.9410 | 0.1194 [0.1190, 0.1270] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GPU4PySCF | 7.1783 | 0.3118 [0.2978, 0.3231] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GenerativeQC Standard | 2.8391 | 0.1213 [0.1194, 0.1281] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GenerativeQC Fast | 3.6528 | 0.3497 [0.3133, 0.4053] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GPU4PySCF | 5.5905 | 0.5931 [0.5806, 0.6137] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GenerativeQC Standard | 4.2686 | 0.4207 [0.4175, 0.4400] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GenerativeQC Fast | 30.8527 | 5.8446 [5.8358, 5.8450] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GPU4PySCF | 10.9368 | 3.8751 [3.8739, 3.8776] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GenerativeQC Standard | 37.4230 | 7.8530 [7.7631, 7.8816] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GenerativeQC Fast | 49.2368 | 14.9745 [14.9399, 15.0506] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GPU4PySCF | 26.2096 | 12.7685 [12.7679, 12.8927] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GenerativeQC Standard | 69.2235 | 15.8573 [15.8531, 15.8619] | complete; accuracy qualified |
| df | M06-2X | 3 | GenerativeQC Fast | 3.0212 | 0.1089 [0.1076, 0.1122] | complete; accuracy failed |
| df | M06-2X | 3 | GPU4PySCF | 6.9362 | 0.4438 [0.3654, 0.5016] | complete; accuracy failed |
| df | M06-2X | 3 | GenerativeQC Standard | 2.8300 | 0.1049 [0.1038, 0.1104] | complete; accuracy failed |
| df | M06-2X | 12 | GenerativeQC Fast | 4.2334 | 0.3370 [0.3357, 0.3476] | complete; accuracy qualified |
| df | M06-2X | 12 | GPU4PySCF | 5.3659 | 0.5573 [0.5303, 0.5969] | complete; accuracy qualified |
| df | M06-2X | 12 | GenerativeQC Standard | 4.3010 | 0.3733 [0.3707, 0.3763] | complete; accuracy qualified |
| df | M06-2X | 48 | GenerativeQC Fast | 18.6963 | 4.8289 [4.8281, 4.8407] | complete; accuracy qualified |
| df | M06-2X | 48 | GPU4PySCF | 8.9107 | 3.4431 [3.4370, 3.4998] | complete; accuracy qualified |
| df | M06-2X | 48 | GenerativeQC Standard | 19.2521 | 5.1813 [5.1759, 5.1826] | complete; accuracy qualified |
| df | M06-2X | 96 | GenerativeQC Fast | 65.8698 | 11.6539 [11.5987, 11.7112] | complete; accuracy qualified |
| df | M06-2X | 96 | GPU4PySCF | 31.1953 | 15.3301 [15.3283, 15.3498] | complete; accuracy qualified |
| df | M06-2X | 96 | GenerativeQC Standard | 64.9455 | 11.8508 [11.8433, 11.8604] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GenerativeQC Fast | 4.7133 | 0.3305 [0.3298, 0.3319] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GPU4PySCF | 7.1399 | 0.8762 [0.6658, 0.8997] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GenerativeQC Standard | 4.7329 | 0.3346 [0.3338, 0.3348] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GenerativeQC Fast | 13.1652 | 0.8763 [0.8747, 0.8911] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GPU4PySCF | 10.6907 | 2.7447 [2.7322, 2.7858] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GenerativeQC Standard | 13.1731 | 0.9021 [0.8948, 0.9071] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GenerativeQC Fast | 185.4777 | 11.9075 [11.9010, 11.9133] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GPU4PySCF | 90.6781 | 32.2217 [32.2078, 32.3274] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GenerativeQC Standard | 187.1891 | 11.9968 [11.9804, 12.0251] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GenerativeQC Fast | 711.5107 | 110.2682 [110.2454, 110.4658] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GPU4PySCF | 383.7765 | 132.6962 [132.6679, 132.7015] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GenerativeQC Standard | 720.5540 | 111.8716 [111.8036, 111.9943] | complete; accuracy qualified |
| direct | HF | 3 | GenerativeQC Fast | 1.7480 | 0.0717 [0.0706, 0.0718] | complete; accuracy qualified |
| direct | HF | 3 | GPU4PySCF | 6.6541 | 0.9529 [0.8046, 1.0675] | complete; accuracy qualified |
| direct | HF | 3 | GenerativeQC Standard | 1.7297 | 0.0721 [0.0712, 0.0727] | complete; accuracy qualified |
| direct | HF | 12 | GenerativeQC Fast | 2.5120 | 0.1184 [0.1175, 0.1188] | complete; accuracy qualified |
| direct | HF | 12 | GPU4PySCF | 10.4121 | 1.9435 [1.8341, 3.0350] | complete; accuracy qualified |
| direct | HF | 12 | GenerativeQC Standard | 2.2185 | 0.1177 [0.1171, 0.1179] | complete; accuracy qualified |
| direct | HF | 48 | GenerativeQC Fast | 8.3813 | 0.7556 [0.7536, 0.7577] | complete; accuracy qualified |
| direct | HF | 48 | GPU4PySCF | 8.5213 | 2.3877 [2.1640, 2.6960] | complete; accuracy qualified |
| direct | HF | 48 | GenerativeQC Standard | 8.2121 | 0.7440 [0.7412, 0.7482] | complete; accuracy qualified |
| direct | HF | 96 | GenerativeQC Fast | 26.1719 | 2.9999 [2.9978, 3.0037] | complete; accuracy qualified |
| direct | HF | 96 | GPU4PySCF | 11.6582 | 3.3007 [3.3004, 3.3428] | complete; accuracy qualified |
| direct | HF | 96 | GenerativeQC Standard | 25.9193 | 2.9283 [2.9261, 2.9309] | complete; accuracy qualified |
| direct | PBE | 3 | GenerativeQC Fast | 1.4843 | 0.0590 [0.0584, 0.0597] | complete; accuracy qualified |
| direct | PBE | 3 | GPU4PySCF | 6.6547 | 0.9863 [0.6459, 1.7168] | complete; accuracy qualified |
| direct | PBE | 3 | GenerativeQC Standard | 1.7120 | 0.0581 [0.0573, 0.0586] | complete; accuracy qualified |
| direct | PBE | 12 | GenerativeQC Fast | 2.1286 | 0.2870 [0.2866, 0.2894] | complete; accuracy qualified |
| direct | PBE | 12 | GPU4PySCF | 7.9371 | 1.3168 [1.1004, 1.3850] | complete; accuracy qualified |
| direct | PBE | 12 | GenerativeQC Standard | 2.4590 | 0.2944 [0.2890, 0.3061] | complete; accuracy qualified |
| direct | PBE | 48 | GenerativeQC Fast | 7.8859 | 2.0476 [2.0441, 2.0975] | complete; accuracy qualified |
| direct | PBE | 48 | GPU4PySCF | 9.9938 | 2.3394 [2.3353, 2.6159] | complete; accuracy qualified |
| direct | PBE | 48 | GenerativeQC Standard | 10.0547 | 2.0888 [2.0869, 2.1125] | complete; accuracy qualified |
| direct | PBE | 96 | GenerativeQC Fast | 20.3377 | 6.7475 [6.7182, 6.8827] | complete; accuracy qualified |
| direct | PBE | 96 | GPU4PySCF | 16.1293 | 5.5541 [5.5382, 5.5879] | complete; accuracy qualified |
| direct | PBE | 96 | GenerativeQC Standard | 26.3896 | 6.8636 [6.7905, 7.0859] | complete; accuracy qualified |
| direct | PBE0 | 3 | GenerativeQC Fast | 2.0643 | 0.0914 [0.0909, 0.1216] | complete; accuracy qualified |
| direct | PBE0 | 3 | GPU4PySCF | 13.1353 | 2.6950 [1.7507, 3.2602] | complete; accuracy qualified |
| direct | PBE0 | 3 | GenerativeQC Standard | 2.2119 | 0.0892 [0.0887, 0.0904] | complete; accuracy qualified |
| direct | PBE0 | 12 | GenerativeQC Fast | 2.6970 | 0.3817 [0.3795, 0.3834] | complete; accuracy qualified |
| direct | PBE0 | 12 | GPU4PySCF | 11.3653 | 2.2238 [1.8715, 3.1280] | complete; accuracy qualified |
| direct | PBE0 | 12 | GenerativeQC Standard | 4.0586 | 0.3982 [0.3982, 0.4112] | complete; accuracy qualified |
| direct | PBE0 | 48 | GenerativeQC Fast | 10.8649 | 2.4970 [2.4904, 2.5230] | complete; accuracy qualified |
| direct | PBE0 | 48 | GPU4PySCF | 12.1583 | 3.2431 [2.9916, 3.9545] | complete; accuracy qualified |
| direct | PBE0 | 48 | GenerativeQC Standard | 17.6492 | 2.7313 [2.7291, 2.7869] | complete; accuracy qualified |
| direct | PBE0 | 96 | GenerativeQC Fast | 29.2721 | 7.7733 [7.7644, 7.7915] | complete; accuracy qualified |
| direct | PBE0 | 96 | GPU4PySCF | 20.3213 | 6.8926 [6.8910, 6.8994] | complete; accuracy qualified |
| direct | PBE0 | 96 | GenerativeQC Standard | 39.5314 | 8.3745 [8.3651, 8.3954] | complete; accuracy qualified |
| direct | B3LYP | 3 | GenerativeQC Fast | 2.5278 | 0.0946 [0.0944, 0.0951] | complete; accuracy qualified |
| direct | B3LYP | 3 | GPU4PySCF | 12.6668 | 2.6236 [2.4481, 2.6358] | complete; accuracy qualified |
| direct | B3LYP | 3 | GenerativeQC Standard | 2.4168 | 0.0950 [0.0947, 0.1044] | complete; accuracy qualified |
| direct | B3LYP | 12 | GenerativeQC Fast | 2.6434 | 0.4055 [0.4054, 0.4548] | complete; accuracy qualified |
| direct | B3LYP | 12 | GPU4PySCF | 11.7884 | 2.4203 [1.4295, 2.6583] | complete; accuracy qualified |
| direct | B3LYP | 12 | GenerativeQC Standard | 3.9316 | 0.4276 [0.4216, 0.4386] | complete; accuracy qualified |
| direct | B3LYP | 48 | GenerativeQC Fast | 11.1009 | 2.5741 [2.5589, 2.6029] | complete; accuracy qualified |
| direct | B3LYP | 48 | GPU4PySCF | 13.0428 | 3.0608 [3.0133, 3.2854] | complete; accuracy qualified |
| direct | B3LYP | 48 | GenerativeQC Standard | 17.5907 | 2.7486 [2.7340, 2.7612] | complete; accuracy qualified |
| direct | B3LYP | 96 | GenerativeQC Fast | 30.0747 | 8.0292 [8.0205, 8.0723] | complete; accuracy qualified |
| direct | B3LYP | 96 | GPU4PySCF | 20.9973 | 6.6363 [6.6298, 6.6446] | complete; accuracy qualified |
| direct | B3LYP | 96 | GenerativeQC Standard | 40.3336 | 8.5611 [8.5475, 8.5999] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GenerativeQC Fast | 1.6192 | 0.0940 [0.0933, 0.0976] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GPU4PySCF | 6.5247 | 0.8048 [0.5676, 2.2341] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GenerativeQC Standard | 1.8428 | 0.0919 [0.0908, 0.1007] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GenerativeQC Fast | 2.3550 | 0.3793 [0.3761, 0.3892] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GPU4PySCF | 7.0999 | 1.0515 [0.9696, 1.3434] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GenerativeQC Standard | 2.6197 | 0.4041 [0.3917, 0.4291] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GenerativeQC Fast | 8.4069 | 2.3320 [2.3305, 2.3339] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GPU4PySCF | 10.5589 | 2.9914 [2.9853, 3.1624] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GenerativeQC Standard | 10.9322 | 2.3294 [2.3293, 2.3353] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GenerativeQC Fast | 21.9550 | 7.4612 [7.4515, 7.4656] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GPU4PySCF | 19.1945 | 7.6247 [7.6183, 7.6361] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GenerativeQC Standard | 29.0117 | 7.3384 [7.2922, 7.3444] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GenerativeQC Fast | 1.6987 | 0.1331 [0.1326, 0.1334] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GPU4PySCF | 8.1468 | 1.2241 [1.1922, 2.2462] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GenerativeQC Standard | 1.8526 | 0.1297 [0.1283, 0.1297] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GenerativeQC Fast | 4.4941 | 0.8913 [0.8894, 0.9091] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GPU4PySCF | 8.4802 | 2.2993 [1.6455, 2.4043] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GenerativeQC Standard | 5.0853 | 0.9295 [0.9271, 0.9317] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GenerativeQC Fast | 15.1572 | 5.2343 [5.2249, 5.2374] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GPU4PySCF | 17.0351 | 5.8961 [5.8879, 5.9493] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GenerativeQC Standard | 20.8511 | 5.2440 [5.2235, 5.2612] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GenerativeQC Fast | 47.5625 | 18.9234 [18.9145, 18.9344] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GPU4PySCF | 32.3528 | 14.6690 [14.6667, 14.6716] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GenerativeQC Standard | 65.3342 | 19.2781 [19.2676, 19.2782] | complete; accuracy qualified |
| direct | M06-2X | 3 | GenerativeQC Fast | 2.1263 | 0.1108 [0.1100, 0.1110] | complete; accuracy failed |
| direct | M06-2X | 3 | GPU4PySCF | 12.4358 | 2.1019 [1.8870, 4.0239] | complete; accuracy failed |
| direct | M06-2X | 3 | GenerativeQC Standard | 2.5341 | 0.1078 [0.1078, 0.1083] | complete; accuracy failed |
| direct | M06-2X | 12 | GenerativeQC Fast | 2.9347 | 0.4533 [0.4442, 0.4554] | complete; accuracy qualified |
| direct | M06-2X | 12 | GPU4PySCF | 11.0493 | 2.7053 [2.2111, 2.7910] | complete; accuracy qualified |
| direct | M06-2X | 12 | GenerativeQC Standard | 4.3323 | 0.4435 [0.4414, 0.4437] | complete; accuracy qualified |
| direct | M06-2X | 48 | GenerativeQC Fast | 11.7862 | 2.6228 [2.6223, 2.6550] | complete; accuracy qualified |
| direct | M06-2X | 48 | GPU4PySCF | 13.5805 | 3.7251 [3.7243, 3.7382] | complete; accuracy qualified |
| direct | M06-2X | 48 | GenerativeQC Standard | 18.6809 | 2.8577 [2.8427, 2.8898] | complete; accuracy qualified |
| direct | M06-2X | 96 | GenerativeQC Fast | 32.2877 | 8.2934 [8.2556, 8.3012] | complete; accuracy qualified |
| direct | M06-2X | 96 | GPU4PySCF | 25.1299 | 8.8181 [8.8094, 8.8436] | complete; accuracy qualified |
| direct | M06-2X | 96 | GenerativeQC Standard | 42.9366 | 8.8250 [8.8250, 8.8275] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GenerativeQC Fast | 4.3494 | 0.3681 [0.3680, 0.3685] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GPU4PySCF | 12.9748 | 3.0711 [2.7580, 3.8146] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GenerativeQC Standard | 5.4135 | 0.3669 [0.3665, 0.3696] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GenerativeQC Fast | 7.8080 | 0.8727 [0.8605, 0.8873] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GPU4PySCF | 17.3980 | 5.1370 [4.6894, 5.4348] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GenerativeQC Standard | 12.1547 | 0.8626 [0.8576, 0.8632] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GenerativeQC Fast | 70.5654 | 6.5887 [6.5731, 6.6050] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GPU4PySCF | 96.9424 | 32.5741 [32.5718, 32.5760] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GenerativeQC Standard | 86.9449 | 6.8890 [6.8815, 6.9064] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GenerativeQC Fast | 285.8455 | 28.8494 [28.8408, 28.8694] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GPU4PySCF | 369.7620 | 119.3952 [119.3946, 119.4173] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GenerativeQC Standard | 303.7942 | 30.7300 [30.7176, 30.7924] | complete; accuracy qualified |
