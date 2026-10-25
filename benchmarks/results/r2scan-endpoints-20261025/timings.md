# Default endpoint timings

Absolute latency; qualification is limited to the sampled static geometries.

| Mode | Method | Atoms | Engine | Cold (s) | Warm median [range] (s) | Status |
|---|---|---:|---|---:|---|---|
| df | HF | 3 | GPU4PySCF | 2.8472 | 0.2243 [0.2220, 0.2626] | complete; accuracy failed |
| df | HF | 3 | GenerativeQC Standard | 0.5043 | 0.0178 [0.0177, 0.0225] | complete; accuracy failed |
| df | HF | 12 | GPU4PySCF | 2.8991 | 0.2744 [0.2701, 0.3666] | complete; accuracy failed |
| df | HF | 12 | GenerativeQC Standard | 0.9166 | 0.1067 [0.1066, 0.1096] | complete; accuracy failed |
| df | HF | 48 | GPU4PySCF | 7.8640 | 1.9194 [1.8052, 1.9290] | complete; accuracy failed |
| df | HF | 48 | GenerativeQC Standard | 3.5710 | 0.9136 [0.8980, 0.9604] | complete; accuracy failed |
| df | HF | 96 | GPU4PySCF | 19.8397 | 9.9191 [9.8957, 10.1553] | complete; accuracy failed |
| df | HF | 96 | GenerativeQC Standard | 20.8211 | 4.0599 [4.0391, 4.0613] | complete; accuracy failed |
| df | PBE | 3 | GPU4PySCF | 4.0730 | 0.2529 [0.2404, 0.3030] | complete; accuracy qualified |
| df | PBE | 3 | GenerativeQC Standard | 1.9436 | 0.0654 [0.0606, 0.1450] | complete; accuracy qualified |
| df | PBE | 12 | GPU4PySCF | 6.8158 | 0.4893 [0.4094, 0.5166] | complete; accuracy qualified |
| df | PBE | 12 | GenerativeQC Standard | 2.6502 | 0.2176 [0.2174, 0.2206] | complete; accuracy qualified |
| df | PBE | 48 | GPU4PySCF | 9.2200 | 1.6576 [1.4882, 1.6790] | complete; accuracy qualified |
| df | PBE | 48 | GenerativeQC Standard | 5.4746 | 1.3508 [1.3300, 1.3788] | complete; accuracy qualified |
| df | PBE | 96 | GPU4PySCF | 13.8045 | 4.7271 [4.6442, 4.7371] | complete; accuracy qualified |
| df | PBE | 96 | GenerativeQC Standard | 12.8943 | 4.4438 [4.4265, 4.4625] | complete; accuracy qualified |
| df | PBE0 | 3 | GPU4PySCF | 3.3762 | 0.2946 [0.2861, 0.2991] | complete; accuracy qualified |
| df | PBE0 | 3 | GenerativeQC Standard | 1.6227 | 0.0700 [0.0674, 0.0788] | complete; accuracy qualified |
| df | PBE0 | 12 | GPU4PySCF | 6.2297 | 0.5722 [0.4751, 0.6474] | complete; accuracy qualified |
| df | PBE0 | 12 | GenerativeQC Standard | 2.6089 | 0.2910 [0.2892, 0.3788] | complete; accuracy qualified |
| df | PBE0 | 48 | GPU4PySCF | 10.5196 | 2.7379 [2.7188, 2.7545] | complete; accuracy qualified |
| df | PBE0 | 48 | GenerativeQC Standard | 6.7583 | 2.0015 [1.9861, 2.0312] | complete; accuracy qualified |
| df | PBE0 | 96 | GPU4PySCF | 25.6555 | 13.2527 [13.2267, 13.3176] | complete; accuracy qualified |
| df | PBE0 | 96 | GenerativeQC Standard | 26.1806 | 8.0773 [8.0667, 8.0933] | complete; accuracy qualified |
| df | B3LYP | 3 | GPU4PySCF | 4.4838 | 0.4108 [0.4089, 0.4550] | complete; accuracy qualified |
| df | B3LYP | 3 | GenerativeQC Standard | 1.7278 | 0.0793 [0.0766, 0.1698] | complete; accuracy qualified |
| df | B3LYP | 12 | GPU4PySCF | 3.8997 | 0.4647 [0.4615, 0.4861] | complete; accuracy qualified |
| df | B3LYP | 12 | GenerativeQC Standard | 2.6165 | 0.3038 [0.3038, 0.3132] | complete; accuracy qualified |
| df | B3LYP | 48 | GPU4PySCF | 8.5285 | 2.8494 [2.7337, 2.9108] | complete; accuracy qualified |
| df | B3LYP | 48 | GenerativeQC Standard | 6.7381 | 2.0255 [2.0171, 2.0517] | complete; accuracy qualified |
| df | B3LYP | 96 | GPU4PySCF | 27.9447 | 13.5530 [13.5402, 13.6499] | complete; accuracy qualified |
| df | B3LYP | 96 | GenerativeQC Standard | 26.9044 | 8.3183 [8.3147, 8.3383] | complete; accuracy qualified |
| df | r²SCAN | 3 | GPU4PySCF | 5.3139 | 0.2435 [0.2407, 0.2880] | complete; accuracy qualified |
| df | r²SCAN | 3 | GenerativeQC Standard | 1.8794 | 0.0864 [0.0846, 0.0873] | complete; accuracy qualified |
| df | r²SCAN | 12 | GPU4PySCF | 6.8415 | 0.4511 [0.4184, 0.4517] | complete; accuracy qualified |
| df | r²SCAN | 12 | GenerativeQC Standard | 2.5613 | 0.3143 [0.3107, 0.3190] | complete; accuracy qualified |
| df | r²SCAN | 48 | GPU4PySCF | 10.6063 | 2.3766 [2.2744, 2.3827] | complete; accuracy qualified |
| df | r²SCAN | 48 | GenerativeQC Standard | 6.4082 | 1.6107 [1.6075, 1.6764] | complete; accuracy qualified |
| df | r²SCAN | 96 | GPU4PySCF | 16.6898 | 6.7371 [6.6615, 6.8494] | complete; accuracy qualified |
| df | r²SCAN | 96 | GenerativeQC Standard | 14.9250 | 5.0023 [4.9927, 5.0071] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GPU4PySCF | 5.4795 | 0.4231 [0.2954, 0.4494] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GenerativeQC Standard | 2.0169 | 0.0873 [0.0865, 0.0971] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GPU4PySCF | 7.2728 | 0.5687 [0.5467, 0.7007] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GenerativeQC Standard | 3.2224 | 0.3697 [0.3682, 0.3748] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GPU4PySCF | 13.5214 | 4.0007 [3.9696, 4.0172] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GenerativeQC Standard | 9.5731 | 3.3679 [3.3412, 3.3843] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GPU4PySCF | 26.3533 | 12.7555 [12.7501, 12.7763] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GenerativeQC Standard | 26.1082 | 8.2695 [8.2611, 8.2929] | complete; accuracy qualified |
| df | M06-2X | 3 | GPU4PySCF | 4.9865 | 0.3685 [0.3335, 0.4319] | complete; accuracy failed |
| df | M06-2X | 3 | GenerativeQC Standard | 1.9096 | 0.0922 [0.0922, 0.1033] | complete; accuracy failed |
| df | M06-2X | 12 | GPU4PySCF | 6.5280 | 0.6024 [0.5473, 0.7646] | complete; accuracy qualified |
| df | M06-2X | 12 | GenerativeQC Standard | 2.8531 | 0.3586 [0.3330, 0.3610] | complete; accuracy qualified |
| df | M06-2X | 48 | GPU4PySCF | 10.0214 | 3.6180 [3.5847, 3.7106] | complete; accuracy qualified |
| df | M06-2X | 48 | GenerativeQC Standard | 7.8645 | 2.2290 [2.2212, 2.2671] | complete; accuracy qualified |
| df | M06-2X | 96 | GPU4PySCF | 30.4039 | 15.1053 [15.0929, 15.1502] | complete; accuracy qualified |
| df | M06-2X | 96 | GenerativeQC Standard | 29.1523 | 8.3842 [8.3758, 8.3849] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GPU4PySCF | 6.5798 | 0.7011 [0.6873, 0.8394] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GenerativeQC Standard | 4.0539 | 0.3160 [0.3145, 0.3304] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GPU4PySCF | 12.6034 | 2.9228 [2.8116, 2.9636] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GenerativeQC Standard | 8.2017 | 0.6907 [0.6901, 0.7611] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GPU4PySCF | 90.8835 | 32.3010 [32.2561, 32.4151] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GenerativeQC Standard | 104.8538 | 5.7000 [5.6969, 5.7053] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GPU4PySCF | 385.0005 | 132.0473 [132.0061, 132.0741] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GenerativeQC Standard | 315.3125 | 38.2931 [38.2832, 38.3342] | complete; accuracy qualified |
| direct | HF | 3 | GPU4PySCF | 4.1390 | 0.9093 [0.8672, 0.9136] | complete; accuracy qualified |
| direct | HF | 3 | GenerativeQC Standard | 1.5164 | 0.0723 [0.0723, 0.0726] | complete; accuracy qualified |
| direct | HF | 12 | GPU4PySCF | 8.9025 | 2.4116 [2.1135, 2.9317] | complete; accuracy qualified |
| direct | HF | 12 | GenerativeQC Standard | 1.7588 | 0.1204 [0.1202, 0.1420] | complete; accuracy qualified |
| direct | HF | 48 | GPU4PySCF | 10.4844 | 2.5395 [2.2325, 2.7624] | complete; accuracy qualified |
| direct | HF | 48 | GenerativeQC Standard | 6.2218 | 0.7560 [0.7541, 0.7606] | complete; accuracy qualified |
| direct | HF | 96 | GPU4PySCF | 14.9873 | 3.5598 [3.5316, 3.7003] | complete; accuracy qualified |
| direct | HF | 96 | GenerativeQC Standard | 18.2407 | 2.9271 [2.9266, 2.9309] | complete; accuracy qualified |
| direct | PBE | 3 | GPU4PySCF | 4.1874 | 0.5883 [0.5874, 0.5942] | complete; accuracy qualified |
| direct | PBE | 3 | GenerativeQC Standard | 1.3977 | 0.0493 [0.0493, 0.0501] | complete; accuracy qualified |
| direct | PBE | 12 | GPU4PySCF | 7.1787 | 1.2773 [0.9396, 1.2847] | complete; accuracy qualified |
| direct | PBE | 12 | GenerativeQC Standard | 2.0840 | 0.2697 [0.2681, 0.2819] | complete; accuracy qualified |
| direct | PBE | 48 | GPU4PySCF | 13.3136 | 2.6317 [2.4392, 2.6884] | complete; accuracy qualified |
| direct | PBE | 48 | GenerativeQC Standard | 6.8458 | 1.7759 [1.7610, 1.7830] | complete; accuracy qualified |
| direct | PBE | 96 | GPU4PySCF | 16.6955 | 5.8000 [5.7683, 5.8254] | complete; accuracy qualified |
| direct | PBE | 96 | GenerativeQC Standard | 15.0340 | 5.1647 [5.1583, 5.1807] | complete; accuracy qualified |
| direct | PBE0 | 3 | GPU4PySCF | 10.1169 | 1.8537 [1.6918, 2.1652] | complete; accuracy qualified |
| direct | PBE0 | 3 | GenerativeQC Standard | 2.0626 | 0.0770 [0.0766, 0.0776] | complete; accuracy qualified |
| direct | PBE0 | 12 | GPU4PySCF | 11.6229 | 2.4059 [2.1535, 2.8665] | complete; accuracy qualified |
| direct | PBE0 | 12 | GenerativeQC Standard | 2.9727 | 0.3719 [0.3656, 0.3737] | complete; accuracy qualified |
| direct | PBE0 | 48 | GPU4PySCF | 18.3067 | 3.6039 [3.3702, 3.7786] | complete; accuracy qualified |
| direct | PBE0 | 48 | GenerativeQC Standard | 9.2837 | 2.3541 [2.3520, 2.3596] | complete; accuracy qualified |
| direct | PBE0 | 96 | GPU4PySCF | 19.6927 | 6.7640 [6.7008, 7.2723] | complete; accuracy qualified |
| direct | PBE0 | 96 | GenerativeQC Standard | 19.3188 | 6.5604 [6.5387, 6.5643] | complete; accuracy qualified |
| direct | B3LYP | 3 | GPU4PySCF | 11.8538 | 2.2451 [2.2090, 2.2816] | complete; accuracy qualified |
| direct | B3LYP | 3 | GenerativeQC Standard | 2.1019 | 0.0839 [0.0831, 0.0842] | complete; accuracy qualified |
| direct | B3LYP | 12 | GPU4PySCF | 14.9455 | 2.4342 [2.2640, 2.8159] | complete; accuracy qualified |
| direct | B3LYP | 12 | GenerativeQC Standard | 3.1252 | 0.3923 [0.3880, 0.4059] | complete; accuracy qualified |
| direct | B3LYP | 48 | GPU4PySCF | 19.1466 | 3.6551 [3.5977, 3.7937] | complete; accuracy qualified |
| direct | B3LYP | 48 | GenerativeQC Standard | 9.5507 | 2.4732 [2.4576, 2.4784] | complete; accuracy qualified |
| direct | B3LYP | 96 | GPU4PySCF | 21.3798 | 6.8545 [6.8510, 6.9679] | complete; accuracy qualified |
| direct | B3LYP | 96 | GenerativeQC Standard | 20.0083 | 6.7598 [6.7244, 6.8246] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GPU4PySCF | 4.3628 | 0.7673 [0.5637, 0.7903] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GenerativeQC Standard | 1.5018 | 0.0836 [0.0834, 0.0841] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GPU4PySCF | 7.7225 | 1.3222 [1.2999, 1.5208] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GenerativeQC Standard | 2.0866 | 0.3394 [0.3385, 0.3451] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GPU4PySCF | 13.2316 | 3.0742 [3.0740, 3.1149] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GenerativeQC Standard | 7.2387 | 2.0797 [2.0757, 2.0823] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GPU4PySCF | 19.4124 | 7.6160 [7.5773, 7.7243] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GenerativeQC Standard | 16.4447 | 5.5508 [5.5493, 5.5584] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GPU4PySCF | 7.2652 | 1.3144 [1.1830, 1.8612] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GenerativeQC Standard | 1.4986 | 0.1026 [0.1017, 0.1029] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GPU4PySCF | 8.9492 | 1.7985 [1.7599, 1.8668] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GenerativeQC Standard | 3.7146 | 0.9030 [0.8772, 0.9103] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GPU4PySCF | 17.5277 | 5.9268 [5.8850, 5.9445] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GenerativeQC Standard | 11.5998 | 4.6434 [4.6413, 4.6837] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GPU4PySCF | 31.9156 | 14.7282 [14.6999, 14.8462] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GenerativeQC Standard | 33.1681 | 13.8040 [13.7894, 13.8094] | complete; accuracy qualified |
| direct | M06-2X | 3 | GPU4PySCF | 11.6638 | 2.3759 [1.7431, 2.4853] | complete; accuracy failed |
| direct | M06-2X | 3 | GenerativeQC Standard | 2.2522 | 0.1022 [0.1012, 0.1023] | complete; accuracy failed |
| direct | M06-2X | 12 | GPU4PySCF | 14.0990 | 2.0925 [1.9467, 2.9380] | complete; accuracy qualified |
| direct | M06-2X | 12 | GenerativeQC Standard | 3.1124 | 0.4213 [0.4195, 0.4217] | complete; accuracy qualified |
| direct | M06-2X | 48 | GPU4PySCF | 18.7775 | 4.4274 [4.2807, 4.8560] | complete; accuracy qualified |
| direct | M06-2X | 48 | GenerativeQC Standard | 11.0052 | 2.6345 [2.6222, 2.6438] | complete; accuracy qualified |
| direct | M06-2X | 96 | GPU4PySCF | 24.4088 | 8.6232 [8.5239, 8.6859] | complete; accuracy qualified |
| direct | M06-2X | 96 | GenerativeQC Standard | 22.9757 | 6.8112 [6.8054, 6.8184] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GPU4PySCF | 13.0744 | 2.8266 [2.8172, 3.8778] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GenerativeQC Standard | 4.4847 | 0.3393 [0.3388, 0.3408] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GPU4PySCF | 21.5134 | 4.9603 [4.3057, 5.6185] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GenerativeQC Standard | 10.6595 | 0.8178 [0.8172, 0.8206] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GPU4PySCF | 98.3496 | 33.2187 [33.1495, 33.2737] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GenerativeQC Standard | 106.9459 | 6.7148 [6.7110, 6.8127] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GPU4PySCF | 374.3407 | 119.6296 [119.6083, 119.8752] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GenerativeQC Standard | 260.2626 | 28.6527 [28.6295, 28.6535] | complete; accuracy qualified |
