# Default endpoint timings

Absolute latency; qualification is limited to the sampled static geometries.

| Mode | Method | Atoms | Engine | Cold (s) | Warm median [range] (s) | Status |
|---|---|---:|---|---:|---|---|
| df | HF | 3 | GPU4PySCF | 2.7244 | 0.2265 [0.2220, 0.2337] | complete; accuracy failed |
| df | HF | 3 | GenerativeQC Standard | 0.5230 | 0.0173 [0.0173, 0.0183] | complete; accuracy failed |
| df | HF | 12 | GPU4PySCF | 3.2977 | 0.2892 [0.2863, 0.3157] | complete; accuracy failed |
| df | HF | 12 | GenerativeQC Standard | 0.9351 | 0.1112 [0.1059, 0.1400] | complete; accuracy failed |
| df | HF | 48 | GPU4PySCF | 5.5638 | 1.7128 [1.7078, 1.7200] | complete; accuracy failed |
| df | HF | 48 | GenerativeQC Standard | 3.3264 | 0.8687 [0.8680, 0.8749] | complete; accuracy failed |
| df | HF | 96 | GPU4PySCF | 20.5280 | 10.0822 [10.0709, 10.1272] | complete; accuracy failed |
| df | HF | 96 | GenerativeQC Standard | 19.1311 | 4.1393 [4.1265, 4.1515] | complete; accuracy failed |
| df | PBE | 3 | GPU4PySCF | 3.4776 | 0.2352 [0.2351, 0.2388] | complete; accuracy qualified |
| df | PBE | 3 | GenerativeQC Standard | 1.6086 | 0.0556 [0.0541, 0.0587] | complete; accuracy qualified |
| df | PBE | 12 | GPU4PySCF | 3.7877 | 0.3939 [0.3712, 0.4185] | complete; accuracy qualified |
| df | PBE | 12 | GenerativeQC Standard | 2.5003 | 0.2235 [0.2189, 0.2247] | complete; accuracy qualified |
| df | PBE | 48 | GPU4PySCF | 7.3167 | 1.4774 [1.4737, 1.4839] | complete; accuracy qualified |
| df | PBE | 48 | GenerativeQC Standard | 5.6429 | 1.3378 [1.3359, 1.3417] | complete; accuracy qualified |
| df | PBE | 96 | GPU4PySCF | 12.5799 | 4.5673 [4.5617, 4.5808] | complete; accuracy qualified |
| df | PBE | 96 | GenerativeQC Standard | 12.5278 | 4.4679 [4.4626, 4.4703] | complete; accuracy qualified |
| df | PBE0 | 3 | GPU4PySCF | 4.0421 | 0.3127 [0.3110, 0.3294] | complete; accuracy qualified |
| df | PBE0 | 3 | GenerativeQC Standard | 1.9390 | 0.0671 [0.0666, 0.0721] | complete; accuracy qualified |
| df | PBE0 | 12 | GPU4PySCF | 4.0555 | 0.4908 [0.4888, 0.4931] | complete; accuracy qualified |
| df | PBE0 | 12 | GenerativeQC Standard | 2.5537 | 0.2934 [0.2854, 0.2950] | complete; accuracy qualified |
| df | PBE0 | 48 | GPU4PySCF | 7.7652 | 2.7681 [2.7152, 2.7761] | complete; accuracy qualified |
| df | PBE0 | 48 | GenerativeQC Standard | 6.7221 | 1.9946 [1.9936, 1.9982] | complete; accuracy qualified |
| df | PBE0 | 96 | GPU4PySCF | 26.2575 | 13.5114 [13.4501, 13.6133] | complete; accuracy qualified |
| df | PBE0 | 96 | GenerativeQC Standard | 26.9349 | 10.5863 [10.5577, 10.5927] | complete; accuracy qualified |
| df | B3LYP | 3 | GPU4PySCF | 3.6487 | 0.3650 [0.3132, 0.5546] | complete; accuracy qualified |
| df | B3LYP | 3 | GenerativeQC Standard | 1.6844 | 0.0824 [0.0760, 0.1278] | complete; accuracy qualified |
| df | B3LYP | 12 | GPU4PySCF | 4.4476 | 0.4633 [0.4630, 0.4763] | complete; accuracy qualified |
| df | B3LYP | 12 | GenerativeQC Standard | 2.6581 | 0.2996 [0.2995, 0.3049] | complete; accuracy qualified |
| df | B3LYP | 48 | GPU4PySCF | 7.5021 | 2.7180 [2.6917, 2.7252] | complete; accuracy qualified |
| df | B3LYP | 48 | GenerativeQC Standard | 6.7474 | 2.0280 [2.0241, 2.0385] | complete; accuracy qualified |
| df | B3LYP | 96 | GPU4PySCF | 28.3283 | 13.4956 [13.4746, 13.6487] | complete; accuracy qualified |
| df | B3LYP | 96 | GenerativeQC Standard | 26.9567 | 10.6306 [10.6172, 10.6904] | complete; accuracy qualified |
| df | r²SCAN | 3 | GPU4PySCF | 3.3679 | 0.2496 [0.2468, 0.2686] | complete; accuracy qualified |
| df | r²SCAN | 3 | GenerativeQC Standard | 1.7527 | 0.0851 [0.0836, 0.0879] | complete; accuracy qualified |
| df | r²SCAN | 12 | GPU4PySCF | 5.4561 | 0.4517 [0.4422, 0.5556] | complete; accuracy qualified |
| df | r²SCAN | 12 | GenerativeQC Standard | 2.7045 | 0.3107 [0.3091, 0.3192] | complete; accuracy qualified |
| df | r²SCAN | 48 | GPU4PySCF | 8.2213 | 2.2051 [2.1916, 2.3091] | complete; accuracy qualified |
| df | r²SCAN | 48 | GenerativeQC Standard | 6.2910 | 1.5858 [1.5800, 1.5878] | complete; accuracy qualified |
| df | r²SCAN | 96 | GPU4PySCF | 15.6298 | 6.4916 [6.4616, 6.4930] | complete; accuracy qualified |
| df | r²SCAN | 96 | GenerativeQC Standard | 14.0730 | 4.8515 [4.8450, 4.8576] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GPU4PySCF | 4.7466 | 0.2930 [0.2840, 0.3247] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GenerativeQC Standard | 1.7059 | 0.0862 [0.0845, 0.0883] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GPU4PySCF | 4.9070 | 0.5451 [0.5353, 0.5536] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GenerativeQC Standard | 2.9508 | 0.3709 [0.3708, 0.3747] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GPU4PySCF | 11.8602 | 3.9106 [3.9031, 3.9111] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GenerativeQC Standard | 8.8094 | 2.6571 [2.6456, 2.6615] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GPU4PySCF | 26.2459 | 12.5876 [12.5604, 12.6089] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GenerativeQC Standard | 25.0954 | 8.1872 [8.1777, 8.2343] | complete; accuracy qualified |
| df | M06-2X | 3 | GPU4PySCF | 3.3040 | 0.3085 [0.3001, 0.3450] | complete; accuracy failed |
| df | M06-2X | 3 | GenerativeQC Standard | 1.7521 | 0.0856 [0.0853, 0.0902] | complete; accuracy failed |
| df | M06-2X | 12 | GPU4PySCF | 4.1542 | 0.5197 [0.5050, 0.5357] | complete; accuracy qualified |
| df | M06-2X | 12 | GenerativeQC Standard | 2.5215 | 0.3273 [0.3252, 0.3416] | complete; accuracy qualified |
| df | M06-2X | 48 | GPU4PySCF | 9.4939 | 3.4312 [3.4220, 3.4347] | complete; accuracy qualified |
| df | M06-2X | 48 | GenerativeQC Standard | 7.8062 | 2.1611 [2.1551, 2.1648] | complete; accuracy qualified |
| df | M06-2X | 96 | GPU4PySCF | 32.3415 | 15.2061 [15.2061, 15.2431] | complete; accuracy qualified |
| df | M06-2X | 96 | GenerativeQC Standard | 27.3602 | 8.4646 [8.4629, 8.4688] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GPU4PySCF | 4.4037 | 0.6790 [0.6622, 0.7163] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GenerativeQC Standard | 3.9042 | 0.3082 [0.3072, 0.3459] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GPU4PySCF | 10.6700 | 2.6575 [2.6524, 2.7462] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GenerativeQC Standard | 8.1015 | 0.6781 [0.6727, 0.6822] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GPU4PySCF | 93.3650 | 32.6621 [32.6567, 32.6820] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GenerativeQC Standard | 68.6855 | 5.7644 [5.7610, 5.7737] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GPU4PySCF | 393.2624 | 135.6602 [135.6496, 135.9416] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GenerativeQC Standard | 250.3376 | 39.2742 [39.2474, 39.2948] | complete; accuracy qualified |
| direct | HF | 3 | GPU4PySCF | 3.7043 | 1.2857 [1.1611, 1.9320] | complete; accuracy qualified |
| direct | HF | 3 | GenerativeQC Standard | 1.4432 | 0.0710 [0.0710, 0.0713] | complete; accuracy qualified |
| direct | HF | 12 | GPU4PySCF | 5.3651 | 1.5250 [1.2068, 1.7850] | complete; accuracy qualified |
| direct | HF | 12 | GenerativeQC Standard | 1.5967 | 0.1174 [0.1169, 0.1356] | complete; accuracy qualified |
| direct | HF | 48 | GPU4PySCF | 8.3509 | 2.0392 [2.0116, 2.2157] | complete; accuracy qualified |
| direct | HF | 48 | GenerativeQC Standard | 6.1230 | 0.7390 [0.7386, 0.7420] | complete; accuracy qualified |
| direct | HF | 96 | GPU4PySCF | 12.2179 | 3.3920 [3.3915, 3.3933] | complete; accuracy qualified |
| direct | HF | 96 | GenerativeQC Standard | 11.5736 | 2.7518 [2.7480, 2.7566] | complete; accuracy qualified |
| direct | PBE | 3 | GPU4PySCF | 6.0763 | 0.5502 [0.5267, 0.6263] | complete; accuracy qualified |
| direct | PBE | 3 | GenerativeQC Standard | 1.4038 | 0.0493 [0.0490, 0.0698] | complete; accuracy qualified |
| direct | PBE | 12 | GPU4PySCF | 5.9733 | 0.9619 [0.8148, 1.1255] | complete; accuracy qualified |
| direct | PBE | 12 | GenerativeQC Standard | 1.8163 | 0.2680 [0.2660, 0.3143] | complete; accuracy qualified |
| direct | PBE | 48 | GPU4PySCF | 9.6280 | 2.3279 [2.3123, 2.3728] | complete; accuracy qualified |
| direct | PBE | 48 | GenerativeQC Standard | 6.6401 | 1.7620 [1.7569, 1.7636] | complete; accuracy qualified |
| direct | PBE | 96 | GPU4PySCF | 16.0857 | 5.6251 [5.6225, 5.6416] | complete; accuracy qualified |
| direct | PBE | 96 | GenerativeQC Standard | 13.9395 | 5.1049 [5.0960, 5.1137] | complete; accuracy qualified |
| direct | PBE0 | 3 | GPU4PySCF | 8.1686 | 1.1686 [1.0425, 1.3014] | complete; accuracy qualified |
| direct | PBE0 | 3 | GenerativeQC Standard | 1.9351 | 0.0794 [0.0792, 0.0800] | complete; accuracy qualified |
| direct | PBE0 | 12 | GPU4PySCF | 7.9973 | 1.6321 [1.3317, 1.8867] | complete; accuracy qualified |
| direct | PBE0 | 12 | GenerativeQC Standard | 3.1095 | 0.3828 [0.3736, 0.3862] | complete; accuracy qualified |
| direct | PBE0 | 48 | GPU4PySCF | 13.1788 | 3.1024 [3.0728, 3.1366] | complete; accuracy qualified |
| direct | PBE0 | 48 | GenerativeQC Standard | 9.4848 | 2.4011 [2.3988, 2.4096] | complete; accuracy qualified |
| direct | PBE0 | 96 | GPU4PySCF | 19.7667 | 6.7549 [6.7511, 6.8775] | complete; accuracy qualified |
| direct | PBE0 | 96 | GenerativeQC Standard | 19.2787 | 6.5913 [6.5582, 6.5920] | complete; accuracy qualified |
| direct | B3LYP | 3 | GPU4PySCF | 5.8180 | 1.0193 [0.9726, 1.3239] | complete; accuracy qualified |
| direct | B3LYP | 3 | GenerativeQC Standard | 1.9201 | 0.0846 [0.0842, 0.0885] | complete; accuracy qualified |
| direct | B3LYP | 12 | GPU4PySCF | 8.1559 | 1.5977 [1.5115, 1.9437] | complete; accuracy qualified |
| direct | B3LYP | 12 | GenerativeQC Standard | 2.9817 | 0.3894 [0.3875, 0.3900] | complete; accuracy qualified |
| direct | B3LYP | 48 | GPU4PySCF | 13.5048 | 3.1660 [3.0841, 3.2123] | complete; accuracy qualified |
| direct | B3LYP | 48 | GenerativeQC Standard | 9.5836 | 2.4313 [2.4261, 2.4442] | complete; accuracy qualified |
| direct | B3LYP | 96 | GPU4PySCF | 22.6607 | 6.7121 [6.7075, 6.7360] | complete; accuracy qualified |
| direct | B3LYP | 96 | GenerativeQC Standard | 19.3139 | 6.6303 [6.6179, 6.6416] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GPU4PySCF | 5.5802 | 0.6653 [0.5672, 0.7297] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GenerativeQC Standard | 1.4287 | 0.0808 [0.0808, 0.0818] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GPU4PySCF | 5.1064 | 0.7503 [0.7379, 0.8037] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GenerativeQC Standard | 2.0260 | 0.3445 [0.3420, 0.3544] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GPU4PySCF | 10.6244 | 3.0351 [3.0271, 3.0422] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GenerativeQC Standard | 6.8902 | 2.0441 [2.0371, 2.0467] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GPU4PySCF | 19.5259 | 7.6447 [7.6330, 7.7210] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GenerativeQC Standard | 15.8900 | 5.5816 [5.5744, 5.5821] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GPU4PySCF | 4.1691 | 0.7650 [0.7530, 0.8053] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GenerativeQC Standard | 1.4791 | 0.1040 [0.1024, 0.1090] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GPU4PySCF | 6.8222 | 1.4314 [1.4224, 1.4460] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GenerativeQC Standard | 3.7198 | 0.8896 [0.8890, 0.8917] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GPU4PySCF | 16.6667 | 5.8446 [5.8438, 5.8457] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GenerativeQC Standard | 11.2425 | 4.6554 [4.6531, 4.6719] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GPU4PySCF | 33.4402 | 14.9682 [14.8666, 15.0399] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GenerativeQC Standard | 31.3019 | 13.9079 [13.9074, 13.9342] | complete; accuracy qualified |
| direct | M06-2X | 3 | GPU4PySCF | 6.0479 | 1.0632 [0.9483, 1.4475] | complete; accuracy failed |
| direct | M06-2X | 3 | GenerativeQC Standard | 2.0158 | 0.0991 [0.0986, 0.0992] | complete; accuracy failed |
| direct | M06-2X | 12 | GPU4PySCF | 7.6334 | 1.5983 [1.5711, 1.7553] | complete; accuracy qualified |
| direct | M06-2X | 12 | GenerativeQC Standard | 3.0845 | 0.4087 [0.4055, 0.4087] | complete; accuracy qualified |
| direct | M06-2X | 48 | GPU4PySCF | 13.7745 | 3.8371 [3.8239, 3.8711] | complete; accuracy qualified |
| direct | M06-2X | 48 | GenerativeQC Standard | 10.9039 | 2.5554 [2.5355, 2.5584] | complete; accuracy qualified |
| direct | M06-2X | 96 | GPU4PySCF | 25.0133 | 8.8024 [8.7744, 8.8101] | complete; accuracy qualified |
| direct | M06-2X | 96 | GenerativeQC Standard | 23.0039 | 6.8734 [6.8560, 6.8834] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GPU4PySCF | 8.9069 | 1.8431 [1.8120, 1.8994] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GenerativeQC Standard | 4.4984 | 0.3447 [0.3446, 0.3449] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GPU4PySCF | 15.5729 | 3.9659 [3.9286, 3.9805] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GenerativeQC Standard | 10.8214 | 0.8274 [0.8245, 0.8297] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GPU4PySCF | 95.8766 | 32.1176 [32.1080, 32.1594] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GenerativeQC Standard | 81.9937 | 6.4989 [6.4913, 6.5131] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GPU4PySCF | 366.7703 | 118.0343 [117.9980, 118.0772] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GenerativeQC Standard | 255.0967 | 28.0384 [28.0225, 28.0675] | complete; accuracy qualified |
