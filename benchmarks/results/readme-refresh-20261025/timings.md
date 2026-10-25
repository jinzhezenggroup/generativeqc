# Default endpoint timings

Absolute latency; qualification is limited to the sampled static geometries.

| Mode | Method | Atoms | Engine | Cold (s) | Warm median [range] (s) | Status |
|---|---|---:|---|---:|---|---|
| df | HF | 3 | GPU4PySCF | 4.0027 | 0.2701 [0.2173, 0.2709] | complete; accuracy failed |
| df | HF | 3 | GenerativeQC Standard | 0.5218 | 0.0179 [0.0177, 0.0183] | complete; accuracy failed |
| df | HF | 12 | GPU4PySCF | 4.9627 | 0.2836 [0.2818, 0.3086] | complete; accuracy failed |
| df | HF | 12 | GenerativeQC Standard | 0.9491 | 0.1082 [0.1081, 0.1127] | complete; accuracy failed |
| df | HF | 48 | GPU4PySCF | 5.8647 | 1.7888 [1.7638, 1.8041] | complete; accuracy failed |
| df | HF | 48 | GenerativeQC Standard | 3.4347 | 0.9012 [0.8963, 0.9019] | complete; accuracy failed |
| df | HF | 96 | GPU4PySCF | 20.5643 | 10.1413 [10.1353, 10.1464] | complete; accuracy failed |
| df | HF | 96 | GenerativeQC Standard | 21.2856 | 4.1604 [4.1604, 4.1848] | complete; accuracy failed |
| df | PBE | 3 | GPU4PySCF | 4.5287 | 0.2482 [0.2400, 0.2502] | complete; accuracy qualified |
| df | PBE | 3 | GenerativeQC Standard | 1.6988 | 0.0570 [0.0548, 0.0825] | complete; accuracy qualified |
| df | PBE | 12 | GPU4PySCF | 6.0798 | 0.3728 [0.3702, 0.4066] | complete; accuracy qualified |
| df | PBE | 12 | GenerativeQC Standard | 2.4088 | 0.2209 [0.2150, 0.2209] | complete; accuracy qualified |
| df | PBE | 48 | GPU4PySCF | 7.4153 | 1.4848 [1.4806, 1.4914] | complete; accuracy qualified |
| df | PBE | 48 | GenerativeQC Standard | 5.4825 | 1.3336 [1.3330, 1.3380] | complete; accuracy qualified |
| df | PBE | 96 | GPU4PySCF | 13.3088 | 4.5574 [4.5564, 4.6213] | complete; accuracy qualified |
| df | PBE | 96 | GenerativeQC Standard | 12.8787 | 4.4360 [4.4346, 4.4378] | complete; accuracy qualified |
| df | PBE0 | 3 | GPU4PySCF | 5.1006 | 0.2911 [0.2861, 0.3220] | complete; accuracy qualified |
| df | PBE0 | 3 | GenerativeQC Standard | 1.8675 | 0.0746 [0.0729, 0.0881] | complete; accuracy qualified |
| df | PBE0 | 12 | GPU4PySCF | 5.1722 | 0.5639 [0.4839, 0.6410] | complete; accuracy qualified |
| df | PBE0 | 12 | GenerativeQC Standard | 2.5130 | 0.2812 [0.2809, 0.2843] | complete; accuracy qualified |
| df | PBE0 | 48 | GPU4PySCF | 8.8705 | 2.8677 [2.8046, 2.9886] | complete; accuracy qualified |
| df | PBE0 | 48 | GenerativeQC Standard | 6.9458 | 2.0305 [2.0248, 2.0314] | complete; accuracy qualified |
| df | PBE0 | 96 | GPU4PySCF | 27.3003 | 13.6117 [13.5841, 13.6166] | complete; accuracy qualified |
| df | PBE0 | 96 | GenerativeQC Standard | 26.8950 | 8.3152 [8.3080, 8.3183] | complete; accuracy qualified |
| df | B3LYP | 3 | GPU4PySCF | 3.8446 | 0.3012 [0.2973, 0.3483] | complete; accuracy qualified |
| df | B3LYP | 3 | GenerativeQC Standard | 1.7706 | 0.0760 [0.0732, 0.0774] | complete; accuracy qualified |
| df | B3LYP | 12 | GPU4PySCF | 5.7285 | 0.6157 [0.5639, 0.6414] | complete; accuracy qualified |
| df | B3LYP | 12 | GenerativeQC Standard | 2.4970 | 0.3063 [0.3016, 0.3070] | complete; accuracy qualified |
| df | B3LYP | 48 | GPU4PySCF | 8.9101 | 2.7762 [2.7730, 2.8748] | complete; accuracy qualified |
| df | B3LYP | 48 | GenerativeQC Standard | 6.7735 | 2.0472 [2.0229, 2.0678] | complete; accuracy qualified |
| df | B3LYP | 96 | GPU4PySCF | 27.2158 | 13.1682 [13.1640, 13.1876] | complete; accuracy qualified |
| df | B3LYP | 96 | GenerativeQC Standard | 26.2562 | 8.1105 [8.0965, 8.1176] | complete; accuracy qualified |
| df | r²SCAN | 3 | GPU4PySCF | 5.6019 | 0.2742 [0.2639, 0.2779] | complete; accuracy qualified |
| df | r²SCAN | 3 | GenerativeQC Standard | 1.8285 | 0.0943 [0.0875, 0.1291] | complete; accuracy qualified |
| df | r²SCAN | 12 | GPU4PySCF | 5.9756 | 0.4868 [0.4585, 0.5106] | complete; accuracy qualified |
| df | r²SCAN | 12 | GenerativeQC Standard | 2.6192 | 0.3115 [0.3068, 0.3124] | complete; accuracy qualified |
| df | r²SCAN | 48 | GPU4PySCF | 9.0733 | 2.2022 [2.2002, 2.2729] | complete; accuracy qualified |
| df | r²SCAN | 48 | GenerativeQC Standard | 6.0182 | 1.5543 [1.5486, 1.5544] | complete; accuracy qualified |
| df | r²SCAN | 96 | GPU4PySCF | 16.3244 | 6.6209 [6.5902, 6.6212] | complete; accuracy qualified |
| df | r²SCAN | 96 | GenerativeQC Standard | 14.7939 | 4.9812 [4.9756, 4.9992] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GPU4PySCF | 5.7087 | 0.2912 [0.2861, 0.2915] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GenerativeQC Standard | 1.7734 | 0.0889 [0.0889, 0.0911] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GPU4PySCF | 5.6580 | 0.5479 [0.5277, 0.5484] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GenerativeQC Standard | 2.9435 | 0.3672 [0.3671, 0.3705] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GPU4PySCF | 11.0456 | 3.8745 [3.8585, 3.8753] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GenerativeQC Standard | 9.5521 | 3.3003 [3.2901, 3.3058] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GPU4PySCF | 26.5151 | 12.8386 [12.8154, 12.9168] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GenerativeQC Standard | 26.1444 | 8.2914 [8.2823, 8.3020] | complete; accuracy qualified |
| df | M06-2X | 3 | GPU4PySCF | 5.3259 | 0.4068 [0.3310, 0.4279] | complete; accuracy failed |
| df | M06-2X | 3 | GenerativeQC Standard | 1.7952 | 0.0941 [0.0890, 0.2054] | complete; accuracy failed |
| df | M06-2X | 12 | GPU4PySCF | 5.5348 | 0.5306 [0.5215, 0.5786] | complete; accuracy qualified |
| df | M06-2X | 12 | GenerativeQC Standard | 2.6112 | 0.3368 [0.3349, 0.3421] | complete; accuracy qualified |
| df | M06-2X | 48 | GPU4PySCF | 9.1753 | 3.6647 [3.5465, 3.6957] | complete; accuracy qualified |
| df | M06-2X | 48 | GenerativeQC Standard | 7.8312 | 2.2116 [2.2103, 2.2238] | complete; accuracy qualified |
| df | M06-2X | 96 | GPU4PySCF | 32.4817 | 15.3035 [15.2562, 15.3658] | complete; accuracy qualified |
| df | M06-2X | 96 | GenerativeQC Standard | 31.8145 | 10.8903 [10.8757, 10.9194] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GPU4PySCF | 5.5578 | 0.7115 [0.6680, 0.8057] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GenerativeQC Standard | 3.9268 | 0.3125 [0.3102, 0.3440] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GPU4PySCF | 10.8033 | 2.6869 [2.6864, 2.7033] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GenerativeQC Standard | 8.0431 | 0.6858 [0.6856, 0.6890] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GPU4PySCF | 89.6827 | 31.7855 [31.7528, 31.8121] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GenerativeQC Standard | 103.4813 | 5.5940 [5.5904, 5.6096] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GPU4PySCF | 384.8600 | 132.8503 [132.8109, 132.8935] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GenerativeQC Standard | 316.9773 | 38.5461 [38.5186, 38.5864] | complete; accuracy qualified |
| direct | HF | 3 | GPU4PySCF | 5.0673 | 1.5548 [0.9053, 2.3527] | complete; accuracy qualified |
| direct | HF | 3 | GenerativeQC Standard | 1.5135 | 0.0705 [0.0703, 0.0711] | complete; accuracy qualified |
| direct | HF | 12 | GPU4PySCF | 6.7163 | 2.0326 [1.7893, 2.2567] | complete; accuracy qualified |
| direct | HF | 12 | GenerativeQC Standard | 1.6096 | 0.1175 [0.1170, 0.1332] | complete; accuracy qualified |
| direct | HF | 48 | GPU4PySCF | 10.4010 | 2.2511 [2.2029, 3.0245] | complete; accuracy qualified |
| direct | HF | 48 | GenerativeQC Standard | 6.2985 | 0.7531 [0.7526, 0.7555] | complete; accuracy qualified |
| direct | HF | 96 | GPU4PySCF | 11.8281 | 3.3288 [3.3279, 3.3369] | complete; accuracy qualified |
| direct | HF | 96 | GenerativeQC Standard | 18.0151 | 2.9083 [2.9053, 2.9085] | complete; accuracy qualified |
| direct | PBE | 3 | GPU4PySCF | 3.8423 | 0.4655 [0.4416, 0.9285] | complete; accuracy qualified |
| direct | PBE | 3 | GenerativeQC Standard | 1.4918 | 0.0489 [0.0489, 0.0495] | complete; accuracy qualified |
| direct | PBE | 12 | GPU4PySCF | 7.0284 | 0.9375 [0.9111, 0.9681] | complete; accuracy qualified |
| direct | PBE | 12 | GenerativeQC Standard | 1.9872 | 0.2715 [0.2665, 0.2825] | complete; accuracy qualified |
| direct | PBE | 48 | GPU4PySCF | 9.6716 | 2.2735 [2.2551, 2.2927] | complete; accuracy qualified |
| direct | PBE | 48 | GenerativeQC Standard | 6.4521 | 1.7301 [1.7223, 1.7331] | complete; accuracy qualified |
| direct | PBE | 96 | GPU4PySCF | 15.8112 | 5.5487 [5.5265, 5.6580] | complete; accuracy qualified |
| direct | PBE | 96 | GenerativeQC Standard | 14.5709 | 5.0388 [5.0258, 5.0510] | complete; accuracy qualified |
| direct | PBE0 | 3 | GPU4PySCF | 8.2152 | 1.6999 [1.6693, 1.9370] | complete; accuracy qualified |
| direct | PBE0 | 3 | GenerativeQC Standard | 1.9065 | 0.0769 [0.0765, 0.0777] | complete; accuracy qualified |
| direct | PBE0 | 12 | GPU4PySCF | 11.3590 | 1.7109 [1.6940, 2.0864] | complete; accuracy qualified |
| direct | PBE0 | 12 | GenerativeQC Standard | 3.0733 | 0.3703 [0.3690, 0.3705] | complete; accuracy qualified |
| direct | PBE0 | 48 | GPU4PySCF | 12.2463 | 3.1711 [3.0908, 3.3542] | complete; accuracy qualified |
| direct | PBE0 | 48 | GenerativeQC Standard | 9.2497 | 2.3849 [2.3795, 2.3887] | complete; accuracy qualified |
| direct | PBE0 | 96 | GPU4PySCF | 19.6866 | 6.6961 [6.6666, 6.8518] | complete; accuracy qualified |
| direct | PBE0 | 96 | GenerativeQC Standard | 19.3339 | 6.5560 [6.5534, 6.5615] | complete; accuracy qualified |
| direct | B3LYP | 3 | GPU4PySCF | 8.2388 | 1.3836 [1.3466, 1.4058] | complete; accuracy qualified |
| direct | B3LYP | 3 | GenerativeQC Standard | 1.8873 | 0.0857 [0.0854, 0.1048] | complete; accuracy qualified |
| direct | B3LYP | 12 | GPU4PySCF | 8.2641 | 1.9600 [1.7503, 2.7903] | complete; accuracy qualified |
| direct | B3LYP | 12 | GenerativeQC Standard | 3.0442 | 0.3907 [0.3898, 0.3918] | complete; accuracy qualified |
| direct | B3LYP | 48 | GPU4PySCF | 12.9420 | 3.1833 [3.0919, 3.3117] | complete; accuracy qualified |
| direct | B3LYP | 48 | GenerativeQC Standard | 9.2530 | 2.4026 [2.3856, 2.4057] | complete; accuracy qualified |
| direct | B3LYP | 96 | GPU4PySCF | 21.1321 | 6.6502 [6.6443, 6.6589] | complete; accuracy qualified |
| direct | B3LYP | 96 | GenerativeQC Standard | 19.5646 | 6.5948 [6.5844, 6.6342] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GPU4PySCF | 5.0912 | 0.6883 [0.5925, 0.6920] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GenerativeQC Standard | 1.4868 | 0.0829 [0.0829, 0.0837] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GPU4PySCF | 6.5076 | 0.9975 [0.9577, 1.0286] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GenerativeQC Standard | 2.0627 | 0.3542 [0.3514, 0.3596] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GPU4PySCF | 11.7016 | 3.0656 [3.0604, 3.0704] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GenerativeQC Standard | 7.0844 | 2.0768 [2.0743, 2.0808] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GPU4PySCF | 18.9104 | 7.6350 [7.6240, 7.6420] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GenerativeQC Standard | 16.5369 | 5.5933 [5.5917, 5.6057] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GPU4PySCF | 6.6524 | 0.8022 [0.7816, 0.8565] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GenerativeQC Standard | 1.4608 | 0.1061 [0.1061, 0.1071] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GPU4PySCF | 8.1144 | 1.7074 [1.7035, 2.0450] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GenerativeQC Standard | 3.6890 | 0.9116 [0.8791, 0.9248] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GPU4PySCF | 17.5872 | 5.8981 [5.8835, 6.0152] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GenerativeQC Standard | 11.7472 | 4.6957 [4.6900, 4.7101] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GPU4PySCF | 31.7729 | 14.7668 [14.7586, 14.7818] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GenerativeQC Standard | 33.4322 | 13.8003 [13.7965, 13.8372] | complete; accuracy qualified |
| direct | M06-2X | 3 | GPU4PySCF | 9.3616 | 1.1207 [1.0488, 1.6196] | complete; accuracy failed |
| direct | M06-2X | 3 | GenerativeQC Standard | 2.0904 | 0.1015 [0.1007, 0.1018] | complete; accuracy failed |
| direct | M06-2X | 12 | GPU4PySCF | 11.9333 | 1.9741 [1.9318, 2.2787] | complete; accuracy qualified |
| direct | M06-2X | 12 | GenerativeQC Standard | 3.0810 | 0.4153 [0.4052, 0.4223] | complete; accuracy qualified |
| direct | M06-2X | 48 | GPU4PySCF | 14.5255 | 3.8824 [3.8751, 4.0089] | complete; accuracy qualified |
| direct | M06-2X | 48 | GenerativeQC Standard | 10.9738 | 2.6103 [2.6071, 2.6205] | complete; accuracy qualified |
| direct | M06-2X | 96 | GPU4PySCF | 26.7546 | 8.7806 [8.7706, 8.7898] | complete; accuracy qualified |
| direct | M06-2X | 96 | GenerativeQC Standard | 23.5686 | 6.9854 [6.9841, 6.9887] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GPU4PySCF | 8.2615 | 2.0219 [1.7166, 2.1790] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GenerativeQC Standard | 4.4867 | 0.3503 [0.3502, 0.3506] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GPU4PySCF | 16.0699 | 4.0503 [4.0352, 4.0704] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GenerativeQC Standard | 10.8856 | 0.8437 [0.8406, 0.8564] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GPU4PySCF | 96.6470 | 32.4913 [32.4546, 32.6887] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GenerativeQC Standard | 104.8765 | 6.5893 [6.5869, 6.6006] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GPU4PySCF | 370.8951 | 118.9753 [118.9571, 119.0609] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GenerativeQC Standard | 258.7786 | 28.3838 [28.3493, 28.4070] | complete; accuracy qualified |
