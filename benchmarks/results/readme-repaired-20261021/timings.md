# Default endpoint timings

Absolute latency; qualification is limited to the sampled static geometries.

| Mode | Method | Atoms | Engine | Cold (s) | Warm median [range] (s) | Status |
|---|---|---:|---|---:|---|---|
| df | HF | 3 | GenerativeQC Fast | 0.6650 | 0.0196 [0.0192, 0.0205] | complete; accuracy failed |
| df | HF | 3 | GPU4PySCF | 2.9021 | 0.2192 [0.2185, 0.2251] | complete; accuracy failed |
| df | HF | 3 | GenerativeQC Standard | 0.5491 | 0.0195 [0.0193, 0.0205] | complete; accuracy failed |
| df | HF | 12 | GenerativeQC Fast | 0.9217 | 0.1614 [0.1212, 0.2483] | complete; accuracy failed |
| df | HF | 12 | GPU4PySCF | 4.7153 | 0.2936 [0.2862, 0.2995] | complete; accuracy failed |
| df | HF | 12 | GenerativeQC Standard | 0.9450 | 0.1230 [0.1211, 0.1344] | complete; accuracy failed |
| df | HF | 48 | GenerativeQC Fast | 12.9063 | 1.1606 [1.1561, 1.1798] | complete; accuracy failed |
| df | HF | 48 | GPU4PySCF | 5.5278 | 1.7721 [1.7360, 1.8260] | complete; accuracy failed |
| df | HF | 48 | GenerativeQC Standard | 12.7834 | 1.1437 [1.1418, 1.1451] | complete; accuracy failed |
| df | HF | 96 | GenerativeQC Fast | 99.2380 | 4.9242 [4.9203, 4.9386] | complete; accuracy failed |
| df | HF | 96 | GPU4PySCF | 20.6227 | 10.2610 [10.2574, 10.2794] | complete; accuracy failed |
| df | HF | 96 | GenerativeQC Standard | 98.4909 | 4.9244 [4.9195, 4.9374] | complete; accuracy failed |
| df | PBE | 3 | GenerativeQC Fast | 2.7275 | 0.0784 [0.0759, 0.0849] | complete; accuracy qualified |
| df | PBE | 3 | GPU4PySCF | 4.9599 | 0.2533 [0.2461, 0.3546] | complete; accuracy qualified |
| df | PBE | 3 | GenerativeQC Standard | 2.7658 | 0.0775 [0.0723, 0.0892] | complete; accuracy qualified |
| df | PBE | 12 | GenerativeQC Fast | 3.7350 | 0.2614 [0.2602, 0.3068] | complete; accuracy qualified |
| df | PBE | 12 | GPU4PySCF | 4.9740 | 0.3902 [0.3749, 0.4779] | complete; accuracy qualified |
| df | PBE | 12 | GenerativeQC Standard | 3.6983 | 0.2656 [0.2591, 0.2678] | complete; accuracy qualified |
| df | PBE | 48 | GenerativeQC Fast | 14.4359 | 2.6905 [2.6852, 2.7413] | complete; accuracy qualified |
| df | PBE | 48 | GPU4PySCF | 6.6890 | 1.4726 [1.4538, 1.5365] | complete; accuracy qualified |
| df | PBE | 48 | GenerativeQC Standard | 14.6336 | 2.7266 [2.7153, 2.7276] | complete; accuracy qualified |
| df | PBE | 96 | GenerativeQC Fast | 44.5560 | 6.8472 [6.8440, 6.8517] | complete; accuracy qualified |
| df | PBE | 96 | GPU4PySCF | 12.7393 | 4.6108 [4.6057, 4.6128] | complete; accuracy qualified |
| df | PBE | 96 | GenerativeQC Standard | 44.8241 | 6.9185 [6.9152, 6.9320] | complete; accuracy qualified |
| df | PBE0 | 3 | GenerativeQC Fast | 2.6328 | 0.0941 [0.0875, 0.0987] | complete; accuracy qualified |
| df | PBE0 | 3 | GPU4PySCF | 5.0975 | 0.5139 [0.4682, 0.5485] | complete; accuracy qualified |
| df | PBE0 | 3 | GenerativeQC Standard | 2.7332 | 0.0887 [0.0875, 0.0955] | complete; accuracy qualified |
| df | PBE0 | 12 | GenerativeQC Fast | 3.7136 | 0.3181 [0.3160, 0.3188] | complete; accuracy qualified |
| df | PBE0 | 12 | GPU4PySCF | 4.7855 | 0.5194 [0.5049, 0.6644] | complete; accuracy qualified |
| df | PBE0 | 12 | GenerativeQC Standard | 3.6924 | 0.3220 [0.3214, 0.3228] | complete; accuracy qualified |
| df | PBE0 | 48 | GenerativeQC Fast | 17.4366 | 4.9988 [4.9405, 5.0319] | complete; accuracy qualified |
| df | PBE0 | 48 | GPU4PySCF | 7.8593 | 2.7923 [2.7800, 2.7948] | complete; accuracy qualified |
| df | PBE0 | 48 | GenerativeQC Standard | 18.2405 | 5.0751 [5.0624, 5.1263] | complete; accuracy qualified |
| df | PBE0 | 96 | GenerativeQC Fast | 61.5801 | 11.8723 [11.8713, 11.9137] | complete; accuracy qualified |
| df | PBE0 | 96 | GPU4PySCF | 26.1345 | 13.1842 [13.1812, 13.1902] | complete; accuracy qualified |
| df | PBE0 | 96 | GenerativeQC Standard | 60.7151 | 11.7706 [11.7517, 11.7798] | complete; accuracy qualified |
| df | B3LYP | 3 | GenerativeQC Fast | 2.5922 | 0.0926 [0.0920, 0.0968] | complete; accuracy qualified |
| df | B3LYP | 3 | GPU4PySCF | 4.4889 | 0.3421 [0.3209, 0.4263] | complete; accuracy qualified |
| df | B3LYP | 3 | GenerativeQC Standard | 2.7389 | 0.0936 [0.0920, 0.1051] | complete; accuracy qualified |
| df | B3LYP | 12 | GenerativeQC Fast | 3.7693 | 0.3493 [0.3489, 0.3537] | complete; accuracy qualified |
| df | B3LYP | 12 | GPU4PySCF | 4.6982 | 0.4960 [0.4862, 0.5754] | complete; accuracy qualified |
| df | B3LYP | 12 | GenerativeQC Standard | 3.6849 | 0.3467 [0.3452, 0.3476] | complete; accuracy qualified |
| df | B3LYP | 48 | GenerativeQC Fast | 18.2019 | 5.1226 [5.1200, 5.1527] | complete; accuracy qualified |
| df | B3LYP | 48 | GPU4PySCF | 7.9410 | 2.7608 [2.7344, 2.7726] | complete; accuracy qualified |
| df | B3LYP | 48 | GenerativeQC Standard | 18.6136 | 5.2631 [5.2274, 5.3276] | complete; accuracy qualified |
| df | B3LYP | 96 | GenerativeQC Fast | 63.5716 | 12.1440 [12.1189, 12.1474] | complete; accuracy qualified |
| df | B3LYP | 96 | GPU4PySCF | 27.1179 | 13.3685 [13.3392, 13.3745] | complete; accuracy qualified |
| df | B3LYP | 96 | GenerativeQC Standard | 62.1777 | 11.8586 [11.8512, 11.8777] | complete; accuracy qualified |
| df | r²SCAN | 3 | GenerativeQC Fast | 2.7934 | 0.1014 [0.0996, 0.1083] | complete; accuracy qualified |
| df | r²SCAN | 3 | GPU4PySCF | 4.5603 | 0.3822 [0.2555, 0.4479] | complete; accuracy qualified |
| df | r²SCAN | 3 | GenerativeQC Standard | 2.8520 | 0.1012 [0.1002, 0.1113] | complete; accuracy qualified |
| df | r²SCAN | 12 | GenerativeQC Fast | 4.1989 | 0.3539 [0.3528, 0.3561] | complete; accuracy qualified |
| df | r²SCAN | 12 | GPU4PySCF | 4.8625 | 0.4263 [0.4101, 0.4502] | complete; accuracy qualified |
| df | r²SCAN | 12 | GenerativeQC Standard | 3.9726 | 0.3477 [0.3475, 0.3521] | complete; accuracy qualified |
| df | r²SCAN | 48 | GenerativeQC Fast | 16.1420 | 3.1878 [3.1781, 3.2138] | complete; accuracy qualified |
| df | r²SCAN | 48 | GPU4PySCF | 8.0342 | 2.2077 [2.2010, 2.2256] | complete; accuracy qualified |
| df | r²SCAN | 48 | GenerativeQC Standard | 16.2956 | 3.2263 [3.2093, 3.2366] | complete; accuracy qualified |
| df | r²SCAN | 96 | GenerativeQC Fast | 47.3490 | 7.8344 [7.8083, 7.8385] | complete; accuracy qualified |
| df | r²SCAN | 96 | GPU4PySCF | 15.7827 | 6.5260 [6.5252, 6.5323] | complete; accuracy qualified |
| df | r²SCAN | 96 | GenerativeQC Standard | 48.1304 | 7.9182 [7.8964, 7.9194] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GenerativeQC Fast | 3.0420 | 0.1178 [0.1177, 0.1222] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GPU4PySCF | 4.8373 | 0.3333 [0.3070, 0.4483] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GenerativeQC Standard | 2.9373 | 0.1176 [0.1169, 0.1283] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GenerativeQC Fast | 4.4475 | 0.4561 [0.4457, 0.4977] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GPU4PySCF | 5.0854 | 0.5542 [0.5212, 0.5648] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GenerativeQC Standard | 4.2793 | 0.4065 [0.4056, 0.4076] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GenerativeQC Fast | 38.4117 | 7.8145 [7.8120, 7.8184] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GPU4PySCF | 10.6433 | 3.8336 [3.8272, 3.8538] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GenerativeQC Standard | 38.9156 | 7.9391 [7.9269, 8.0144] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GenerativeQC Fast | 1633.0099 | 172.0747 [172.0709, 172.0822] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GPU4PySCF | 26.3398 | 12.8696 [12.8666, 12.8705] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GenerativeQC Standard | 1648.7017 | 173.9676 [173.3600, 174.1645] | complete; accuracy qualified |
| df | M06-2X | 3 | GenerativeQC Fast | 3.1545 | 0.1141 [0.1134, 0.2025] | complete; accuracy failed |
| df | M06-2X | 3 | GPU4PySCF | 5.1787 | 0.3569 [0.3197, 0.4362] | complete; accuracy failed |
| df | M06-2X | 3 | GenerativeQC Standard | 2.9578 | 0.1325 [0.1088, 0.1421] | complete; accuracy failed |
| df | M06-2X | 12 | GenerativeQC Fast | 4.1714 | 0.3776 [0.3775, 0.4171] | complete; accuracy qualified |
| df | M06-2X | 12 | GPU4PySCF | 6.0346 | 0.5392 [0.5363, 0.5416] | complete; accuracy qualified |
| df | M06-2X | 12 | GenerativeQC Standard | 4.0359 | 0.3825 [0.3796, 0.4712] | complete; accuracy qualified |
| df | M06-2X | 48 | GenerativeQC Fast | 19.4665 | 5.3161 [5.3146, 5.3516] | complete; accuracy qualified |
| df | M06-2X | 48 | GPU4PySCF | 8.9965 | 3.4080 [3.3971, 3.4226] | complete; accuracy qualified |
| df | M06-2X | 48 | GenerativeQC Standard | 19.2050 | 5.1874 [5.1805, 5.2114] | complete; accuracy qualified |
| df | M06-2X | 96 | GenerativeQC Fast | 64.6481 | 12.1629 [12.1553, 12.1865] | complete; accuracy qualified |
| df | M06-2X | 96 | GPU4PySCF | 31.0548 | 15.2655 [15.2516, 15.2682] | complete; accuracy qualified |
| df | M06-2X | 96 | GenerativeQC Standard | 65.7727 | 12.3414 [12.3195, 12.3501] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GenerativeQC Fast | 4.6654 | 0.3335 [0.3326, 0.3365] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GPU4PySCF | 4.7373 | 0.8499 [0.6691, 0.9320] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GenerativeQC Standard | 4.8174 | 0.3337 [0.3328, 0.3342] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GenerativeQC Fast | 13.0465 | 0.8885 [0.8846, 0.9546] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GPU4PySCF | 10.9018 | 2.7598 [2.7482, 2.8901] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GenerativeQC Standard | 13.2161 | 0.8828 [0.8812, 0.8829] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GenerativeQC Fast | 183.7459 | 11.8471 [11.8330, 11.8478] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GPU4PySCF | 91.4267 | 32.3733 [32.3609, 32.3847] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GenerativeQC Standard | 187.2039 | 12.0691 [12.0627, 12.0757] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GenerativeQC Fast | 716.5941 | 107.1155 [107.1143, 107.1181] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GPU4PySCF | 385.6702 | 133.1003 [133.0912, 133.1128] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GenerativeQC Standard | 729.8919 | 113.2249 [113.2244, 113.2839] | complete; accuracy qualified |
| direct | HF | 3 | GenerativeQC Fast | 1.7892 | 0.0731 [0.0717, 0.0737] | complete; accuracy qualified |
| direct | HF | 3 | GPU4PySCF | 4.4981 | 0.9485 [0.9079, 1.0626] | complete; accuracy qualified |
| direct | HF | 3 | GenerativeQC Standard | 1.6454 | 0.0721 [0.0715, 0.0727] | complete; accuracy qualified |
| direct | HF | 12 | GenerativeQC Fast | 2.2517 | 0.1180 [0.1158, 0.1208] | complete; accuracy qualified |
| direct | HF | 12 | GPU4PySCF | 6.6559 | 1.2895 [1.0929, 1.5126] | complete; accuracy qualified |
| direct | HF | 12 | GenerativeQC Standard | 2.0593 | 0.1180 [0.1174, 0.1193] | complete; accuracy qualified |
| direct | HF | 48 | GenerativeQC Fast | 8.2728 | 0.7494 [0.7491, 0.7530] | complete; accuracy qualified |
| direct | HF | 48 | GPU4PySCF | 8.4333 | 2.0232 [2.0036, 2.0330] | complete; accuracy qualified |
| direct | HF | 48 | GenerativeQC Standard | 8.3316 | 0.7564 [0.7544, 0.7574] | complete; accuracy qualified |
| direct | HF | 96 | GenerativeQC Fast | 26.3331 | 2.9715 [2.9714, 2.9734] | complete; accuracy qualified |
| direct | HF | 96 | GPU4PySCF | 11.7608 | 3.3876 [3.3657, 3.4396] | complete; accuracy qualified |
| direct | HF | 96 | GenerativeQC Standard | 26.5391 | 2.9992 [2.9968, 3.0011] | complete; accuracy qualified |
| direct | PBE | 3 | GenerativeQC Fast | 1.5826 | 0.0593 [0.0584, 0.0595] | complete; accuracy qualified |
| direct | PBE | 3 | GPU4PySCF | 5.5481 | 0.8437 [0.7919, 0.9085] | complete; accuracy qualified |
| direct | PBE | 3 | GenerativeQC Standard | 1.4981 | 0.0579 [0.0575, 0.0588] | complete; accuracy qualified |
| direct | PBE | 12 | GenerativeQC Fast | 1.9845 | 0.2892 [0.2886, 0.2893] | complete; accuracy qualified |
| direct | PBE | 12 | GPU4PySCF | 6.4089 | 0.8606 [0.7761, 1.0878] | complete; accuracy qualified |
| direct | PBE | 12 | GenerativeQC Standard | 2.3227 | 0.2968 [0.2955, 0.3021] | complete; accuracy qualified |
| direct | PBE | 48 | GenerativeQC Fast | 12.1601 | 2.1301 [2.1179, 2.1326] | complete; accuracy qualified |
| direct | PBE | 48 | GPU4PySCF | 9.4173 | 2.2931 [2.2913, 2.2954] | complete; accuracy qualified |
| direct | PBE | 48 | GenerativeQC Standard | 15.1357 | 2.1447 [2.1439, 2.1808] | complete; accuracy qualified |
| direct | PBE | 96 | GenerativeQC Fast | 35.8124 | 8.2069 [8.1988, 8.2553] | complete; accuracy qualified |
| direct | PBE | 96 | GPU4PySCF | 16.0211 | 5.7156 [5.6548, 5.7695] | complete; accuracy qualified |
| direct | PBE | 96 | GenerativeQC Standard | 43.9863 | 8.7869 [8.7673, 8.8317] | complete; accuracy qualified |
| direct | PBE0 | 3 | GenerativeQC Fast | 1.8529 | 0.0909 [0.0908, 0.0911] | complete; accuracy qualified |
| direct | PBE0 | 3 | GPU4PySCF | 10.4636 | 1.7479 [1.6186, 2.0750] | complete; accuracy qualified |
| direct | PBE0 | 3 | GenerativeQC Standard | 2.0934 | 0.0877 [0.0871, 0.0879] | complete; accuracy qualified |
| direct | PBE0 | 12 | GenerativeQC Fast | 2.4836 | 0.3805 [0.3778, 0.4001] | complete; accuracy qualified |
| direct | PBE0 | 12 | GPU4PySCF | 11.3925 | 1.6750 [1.5622, 1.6938] | complete; accuracy qualified |
| direct | PBE0 | 12 | GenerativeQC Standard | 4.1822 | 0.4136 [0.3998, 0.4270] | complete; accuracy qualified |
| direct | PBE0 | 48 | GenerativeQC Fast | 13.0814 | 2.4504 [2.4466, 2.4849] | complete; accuracy qualified |
| direct | PBE0 | 48 | GPU4PySCF | 12.0795 | 3.0778 [3.0261, 3.6662] | complete; accuracy qualified |
| direct | PBE0 | 48 | GenerativeQC Standard | 18.2839 | 2.6963 [2.6951, 2.7270] | complete; accuracy qualified |
| direct | PBE0 | 96 | GenerativeQC Fast | 40.2178 | 9.4829 [9.4437, 9.4881] | complete; accuracy qualified |
| direct | PBE0 | 96 | GPU4PySCF | 19.8690 | 6.6554 [6.6539, 6.7205] | complete; accuracy qualified |
| direct | PBE0 | 96 | GenerativeQC Standard | 67.9683 | 10.1243 [10.1191, 10.1658] | complete; accuracy qualified |
| direct | B3LYP | 3 | GenerativeQC Fast | 1.8698 | 0.0937 [0.0936, 0.0938] | complete; accuracy qualified |
| direct | B3LYP | 3 | GPU4PySCF | 9.7079 | 1.9865 [1.9630, 2.7852] | complete; accuracy qualified |
| direct | B3LYP | 3 | GenerativeQC Standard | 2.1843 | 0.0931 [0.0929, 0.0936] | complete; accuracy qualified |
| direct | B3LYP | 12 | GenerativeQC Fast | 2.5330 | 0.4255 [0.4148, 0.4637] | complete; accuracy qualified |
| direct | B3LYP | 12 | GPU4PySCF | 10.6334 | 2.0627 [1.9328, 2.0714] | complete; accuracy qualified |
| direct | B3LYP | 12 | GenerativeQC Standard | 3.9837 | 0.4145 [0.4077, 0.4177] | complete; accuracy qualified |
| direct | B3LYP | 48 | GenerativeQC Fast | 13.5493 | 2.6159 [2.5653, 2.6166] | complete; accuracy qualified |
| direct | B3LYP | 48 | GPU4PySCF | 12.5931 | 2.9992 [2.9854, 3.0390] | complete; accuracy qualified |
| direct | B3LYP | 48 | GenerativeQC Standard | 19.7011 | 2.7705 [2.7672, 2.8410] | complete; accuracy qualified |
| direct | B3LYP | 96 | GenerativeQC Fast | 40.2666 | 9.5413 [9.5158, 9.5450] | complete; accuracy qualified |
| direct | B3LYP | 96 | GPU4PySCF | 21.0682 | 6.7178 [6.7163, 6.7499] | complete; accuracy qualified |
| direct | B3LYP | 96 | GenerativeQC Standard | 73.6630 | 10.4625 [10.4531, 10.4703] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GenerativeQC Fast | 1.6526 | 0.0940 [0.0940, 0.1083] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GPU4PySCF | 5.2468 | 0.4745 [0.4447, 0.8850] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GenerativeQC Standard | 1.6891 | 0.0921 [0.0921, 0.0927] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GenerativeQC Fast | 2.2138 | 0.4097 [0.4033, 0.4305] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GPU4PySCF | 5.9678 | 0.8547 [0.7789, 0.9322] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GenerativeQC Standard | 2.7214 | 0.3726 [0.3664, 0.3852] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GenerativeQC Fast | 12.9199 | 2.4043 [2.4019, 2.4100] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GPU4PySCF | 10.4341 | 2.9853 [2.9845, 2.9963] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GenerativeQC Standard | 16.9321 | 2.4787 [2.4769, 2.4818] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GenerativeQC Fast | 37.3035 | 8.7694 [8.7592, 8.7938] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GPU4PySCF | 19.5407 | 7.7027 [7.7009, 7.7159] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GenerativeQC Standard | 46.1071 | 9.4166 [9.3981, 9.4606] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GenerativeQC Fast | 1.6535 | 0.1351 [0.1345, 0.1376] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GPU4PySCF | 6.0268 | 0.9567 [0.7651, 1.5431] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GenerativeQC Standard | 1.8746 | 0.1279 [0.1275, 0.1305] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GenerativeQC Fast | 5.4954 | 0.9142 [0.9131, 0.9182] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GPU4PySCF | 6.6713 | 1.5474 [1.4387, 1.8595] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GenerativeQC Standard | 6.1932 | 0.9357 [0.9354, 0.9422] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GenerativeQC Fast | 44.8207 | 5.1487 [5.1405, 5.2296] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GPU4PySCF | 16.5647 | 5.8436 [5.8411, 5.9989] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GenerativeQC Standard | 23.0917 | 5.3494 [5.3446, 5.3796] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GenerativeQC Fast | 143.8331 | 24.6987 [24.6968, 24.7222] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GPU4PySCF | 32.9466 | 15.2256 [15.2200, 15.2267] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GenerativeQC Standard | 245.2919 | 26.0624 [26.0577, 26.1058] | complete; accuracy qualified |
| direct | M06-2X | 3 | GenerativeQC Fast | 2.0213 | 0.1112 [0.1109, 0.1113] | complete; accuracy failed |
| direct | M06-2X | 3 | GPU4PySCF | 8.7731 | 1.5902 [1.4705, 2.0980] | complete; accuracy failed |
| direct | M06-2X | 3 | GenerativeQC Standard | 2.4190 | 0.1062 [0.1061, 0.1063] | complete; accuracy failed |
| direct | M06-2X | 12 | GenerativeQC Fast | 2.6005 | 0.4365 [0.4362, 0.4388] | complete; accuracy qualified |
| direct | M06-2X | 12 | GPU4PySCF | 9.3633 | 1.5519 [1.5027, 2.6823] | complete; accuracy qualified |
| direct | M06-2X | 12 | GenerativeQC Standard | 4.4485 | 0.4547 [0.4546, 0.4962] | complete; accuracy qualified |
| direct | M06-2X | 48 | GenerativeQC Fast | 14.3313 | 2.6942 [2.6848, 2.7156] | complete; accuracy qualified |
| direct | M06-2X | 48 | GPU4PySCF | 13.7901 | 3.7907 [3.7892, 3.8007] | complete; accuracy qualified |
| direct | M06-2X | 48 | GenerativeQC Standard | 20.2857 | 2.8193 [2.7954, 2.8462] | complete; accuracy qualified |
| direct | M06-2X | 96 | GenerativeQC Fast | 42.4504 | 9.6939 [9.6791, 9.7293] | complete; accuracy qualified |
| direct | M06-2X | 96 | GPU4PySCF | 24.4744 | 8.5296 [8.5289, 8.5398] | complete; accuracy qualified |
| direct | M06-2X | 96 | GenerativeQC Standard | 73.1597 | 10.3130 [10.2860, 10.3507] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GenerativeQC Fast | 4.0821 | 0.3697 [0.3689, 0.3717] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GPU4PySCF | 9.4418 | 2.1691 [1.8701, 2.7911] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GenerativeQC Standard | 5.5426 | 0.3577 [0.3571, 0.3994] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GenerativeQC Fast | 7.3836 | 0.8722 [0.8598, 0.8793] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GPU4PySCF | 15.7818 | 4.0058 [3.9635, 4.6771] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GenerativeQC Standard | 12.3380 | 0.8954 [0.8909, 0.9237] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GenerativeQC Fast | 69.7877 | 6.5677 [6.5664, 6.5751] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GPU4PySCF | 96.0503 | 32.4419 [32.4337, 32.4708] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GenerativeQC Standard | 88.9801 | 7.1302 [7.1072, 7.1380] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GenerativeQC Fast | 291.1514 | 30.8478 [30.7965, 30.9457] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GPU4PySCF | 378.6633 | 122.1778 [122.1729, 122.2302] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GenerativeQC Standard | 326.5164 | 32.7031 [32.6183, 32.7084] | complete; accuracy qualified |
