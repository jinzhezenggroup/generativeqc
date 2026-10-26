# Default endpoint timings

Absolute latency; qualification is limited to the sampled static geometries.

| Mode | Method | Atoms | Engine | Cold (s) | Warm median [range] (s) | Status |
|---|---|---:|---|---:|---|---|
| df | HF | 3 | GPU4PySCF | 3.1949 | 0.2299 [0.2184, 0.2330] | complete; accuracy failed |
| df | HF | 3 | GenerativeQC Standard | 0.5513 | 0.0175 [0.0175, 0.0180] | complete; accuracy failed |
| df | HF | 12 | GPU4PySCF | 3.1121 | 0.2849 [0.2725, 0.2871] | complete; accuracy failed |
| df | HF | 12 | GenerativeQC Standard | 0.9287 | 0.1081 [0.1081, 0.1101] | complete; accuracy failed |
| df | HF | 48 | GPU4PySCF | 6.4997 | 1.7212 [1.6826, 1.7370] | complete; accuracy failed |
| df | HF | 48 | GenerativeQC Standard | 3.2484 | 0.8716 [0.8699, 0.8732] | complete; accuracy failed |
| df | HF | 96 | GPU4PySCF | 20.4402 | 10.1653 [10.1639, 10.1918] | complete; accuracy failed |
| df | HF | 96 | GenerativeQC Standard | 19.2431 | 4.1615 [4.1537, 4.1747] | complete; accuracy failed |
| df | PBE | 3 | GPU4PySCF | 3.3281 | 0.2339 [0.2306, 0.2388] | complete; accuracy qualified |
| df | PBE | 3 | GenerativeQC Standard | 1.6721 | 0.0560 [0.0546, 0.0589] | complete; accuracy qualified |
| df | PBE | 12 | GPU4PySCF | 5.5274 | 0.4201 [0.4148, 0.4278] | complete; accuracy qualified |
| df | PBE | 12 | GenerativeQC Standard | 2.3966 | 0.2235 [0.2187, 0.2236] | complete; accuracy qualified |
| df | PBE | 48 | GPU4PySCF | 6.8048 | 1.4508 [1.4386, 1.4526] | complete; accuracy qualified |
| df | PBE | 48 | GenerativeQC Standard | 5.3734 | 1.3432 [1.3430, 1.3432] | complete; accuracy qualified |
| df | PBE | 96 | GPU4PySCF | 12.6651 | 4.5733 [4.5669, 4.5773] | complete; accuracy qualified |
| df | PBE | 96 | GenerativeQC Standard | 12.5195 | 4.4876 [4.4768, 4.4934] | complete; accuracy qualified |
| df | PBE0 | 3 | GPU4PySCF | 4.0675 | 0.3039 [0.3004, 0.3106] | complete; accuracy qualified |
| df | PBE0 | 3 | GenerativeQC Standard | 1.7263 | 0.0726 [0.0685, 0.0790] | complete; accuracy qualified |
| df | PBE0 | 12 | GPU4PySCF | 3.8343 | 0.4705 [0.4644, 0.4777] | complete; accuracy qualified |
| df | PBE0 | 12 | GenerativeQC Standard | 2.4242 | 0.2816 [0.2766, 0.2851] | complete; accuracy qualified |
| df | PBE0 | 48 | GPU4PySCF | 7.6809 | 2.7278 [2.6675, 2.7606] | complete; accuracy qualified |
| df | PBE0 | 48 | GenerativeQC Standard | 6.5382 | 1.9796 [1.9773, 2.0026] | complete; accuracy qualified |
| df | PBE0 | 96 | GPU4PySCF | 26.1172 | 13.3414 [13.3271, 13.3698] | complete; accuracy qualified |
| df | PBE0 | 96 | GenerativeQC Standard | 24.2767 | 8.1406 [8.1405, 8.1483] | complete; accuracy qualified |
| df | B3LYP | 3 | GPU4PySCF | 3.3070 | 0.2935 [0.2888, 0.2993] | complete; accuracy qualified |
| df | B3LYP | 3 | GenerativeQC Standard | 1.6625 | 0.0745 [0.0711, 0.0782] | complete; accuracy qualified |
| df | B3LYP | 12 | GPU4PySCF | 5.0578 | 0.4971 [0.4900, 0.4975] | complete; accuracy qualified |
| df | B3LYP | 12 | GenerativeQC Standard | 2.5012 | 0.3045 [0.2994, 0.3097] | complete; accuracy qualified |
| df | B3LYP | 48 | GPU4PySCF | 7.6502 | 2.7928 [2.7570, 2.8789] | complete; accuracy qualified |
| df | B3LYP | 48 | GenerativeQC Standard | 6.8327 | 2.0774 [2.0762, 2.0780] | complete; accuracy qualified |
| df | B3LYP | 96 | GPU4PySCF | 27.7059 | 13.3635 [13.3328, 13.4899] | complete; accuracy qualified |
| df | B3LYP | 96 | GenerativeQC Standard | 24.4838 | 8.2021 [8.1787, 8.2094] | complete; accuracy qualified |
| df | r²SCAN | 3 | GPU4PySCF | 3.9199 | 0.2469 [0.2409, 0.2518] | complete; accuracy qualified |
| df | r²SCAN | 3 | GenerativeQC Standard | 1.7250 | 0.0852 [0.0824, 0.0887] | complete; accuracy qualified |
| df | r²SCAN | 12 | GPU4PySCF | 3.9952 | 0.4076 [0.4055, 0.4144] | complete; accuracy qualified |
| df | r²SCAN | 12 | GenerativeQC Standard | 2.5586 | 0.3091 [0.3064, 0.3142] | complete; accuracy qualified |
| df | r²SCAN | 48 | GPU4PySCF | 9.2551 | 2.2487 [2.2443, 2.2537] | complete; accuracy qualified |
| df | r²SCAN | 48 | GenerativeQC Standard | 6.1760 | 1.5919 [1.5905, 1.5929] | complete; accuracy qualified |
| df | r²SCAN | 96 | GPU4PySCF | 15.5000 | 6.4159 [6.4155, 6.4548] | complete; accuracy qualified |
| df | r²SCAN | 96 | GenerativeQC Standard | 14.1376 | 4.8495 [4.8366, 4.8599] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GPU4PySCF | 3.9723 | 0.3049 [0.2970, 0.3181] | complete; accuracy qualified |
| df | r²SCAN-3c | 3 | GenerativeQC Standard | 1.7681 | 0.0869 [0.0856, 0.0887] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GPU4PySCF | 5.6623 | 0.5505 [0.5477, 0.5628] | complete; accuracy qualified |
| df | r²SCAN-3c | 12 | GenerativeQC Standard | 2.9545 | 0.3791 [0.3789, 0.3806] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GPU4PySCF | 11.1417 | 3.8692 [3.8674, 3.8822] | complete; accuracy qualified |
| df | r²SCAN-3c | 48 | GenerativeQC Standard | 8.6533 | 2.6188 [2.6078, 2.6230] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GPU4PySCF | 26.6038 | 12.7425 [12.7244, 12.7481] | complete; accuracy qualified |
| df | r²SCAN-3c | 96 | GenerativeQC Standard | 25.2522 | 8.2708 [8.2549, 8.2981] | complete; accuracy qualified |
| df | M06-2X | 3 | GPU4PySCF | 4.1264 | 0.3297 [0.2959, 0.3461] | complete; accuracy failed |
| df | M06-2X | 3 | GenerativeQC Standard | 1.9075 | 0.0861 [0.0843, 0.0867] | complete; accuracy failed |
| df | M06-2X | 12 | GPU4PySCF | 3.7820 | 0.5075 [0.5059, 0.5430] | complete; accuracy qualified |
| df | M06-2X | 12 | GenerativeQC Standard | 2.6537 | 0.3323 [0.3313, 0.3335] | complete; accuracy qualified |
| df | M06-2X | 48 | GPU4PySCF | 9.4683 | 3.4868 [3.4861, 3.5124] | complete; accuracy qualified |
| df | M06-2X | 48 | GenerativeQC Standard | 7.7561 | 2.2065 [2.2013, 2.2117] | complete; accuracy qualified |
| df | M06-2X | 96 | GPU4PySCF | 32.7762 | 15.5766 [15.5511, 15.6127] | complete; accuracy qualified |
| df | M06-2X | 96 | GenerativeQC Standard | 27.7991 | 8.5865 [8.5791, 8.5997] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GPU4PySCF | 5.1806 | 0.6876 [0.6700, 0.7008] | complete; accuracy qualified |
| df | ωB97M-V | 3 | GenerativeQC Standard | 3.9311 | 0.3221 [0.3209, 0.3234] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GPU4PySCF | 10.7024 | 2.6957 [2.6944, 2.7011] | complete; accuracy qualified |
| df | ωB97M-V | 12 | GenerativeQC Standard | 8.1360 | 0.7064 [0.6958, 0.7146] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GPU4PySCF | 90.9731 | 32.3360 [32.2885, 32.3387] | complete; accuracy qualified |
| df | ωB97M-V | 48 | GenerativeQC Standard | 67.9493 | 5.7257 [5.7170, 5.7298] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GPU4PySCF | 380.7586 | 131.2812 [131.2552, 131.2870] | complete; accuracy qualified |
| df | ωB97M-V | 96 | GenerativeQC Standard | 242.3062 | 38.0822 [38.0690, 38.1332] | complete; accuracy qualified |
| direct | HF | 3 | GPU4PySCF | 4.9809 | 0.9204 [0.8485, 1.0423] | complete; accuracy qualified |
| direct | HF | 3 | GenerativeQC Standard | 1.5870 | 0.0732 [0.0723, 0.0733] | complete; accuracy qualified |
| direct | HF | 12 | GPU4PySCF | 7.1441 | 1.7838 [1.6502, 1.9694] | complete; accuracy qualified |
| direct | HF | 12 | GenerativeQC Standard | 1.7877 | 0.1209 [0.1202, 0.1340] | complete; accuracy qualified |
| direct | HF | 48 | GPU4PySCF | 8.6143 | 2.1835 [2.1015, 2.2366] | complete; accuracy qualified |
| direct | HF | 48 | GenerativeQC Standard | 6.0848 | 0.7287 [0.7263, 0.7317] | complete; accuracy qualified |
| direct | HF | 96 | GPU4PySCF | 11.6808 | 3.2948 [3.2865, 3.3492] | complete; accuracy qualified |
| direct | HF | 96 | GenerativeQC Standard | 11.2194 | 2.6666 [2.6642, 2.6710] | complete; accuracy qualified |
| direct | PBE | 3 | GPU4PySCF | 4.9892 | 0.5540 [0.5240, 0.5953] | complete; accuracy qualified |
| direct | PBE | 3 | GenerativeQC Standard | 1.4459 | 0.0497 [0.0492, 0.0528] | complete; accuracy qualified |
| direct | PBE | 12 | GPU4PySCF | 4.7927 | 0.7463 [0.7410, 0.8243] | complete; accuracy qualified |
| direct | PBE | 12 | GenerativeQC Standard | 1.9388 | 0.2643 [0.2638, 0.2798] | complete; accuracy qualified |
| direct | PBE | 48 | GPU4PySCF | 9.1247 | 2.2740 [2.2698, 2.2741] | complete; accuracy qualified |
| direct | PBE | 48 | GenerativeQC Standard | 6.4191 | 1.7373 [1.7372, 1.7484] | complete; accuracy qualified |
| direct | PBE | 96 | GPU4PySCF | 15.9807 | 5.6870 [5.6724, 5.6923] | complete; accuracy qualified |
| direct | PBE | 96 | GenerativeQC Standard | 14.2679 | 5.1566 [5.1552, 5.1674] | complete; accuracy qualified |
| direct | PBE0 | 3 | GPU4PySCF | 9.1532 | 1.5264 [1.3045, 1.9246] | complete; accuracy qualified |
| direct | PBE0 | 3 | GenerativeQC Standard | 1.8711 | 0.0789 [0.0786, 0.0791] | complete; accuracy qualified |
| direct | PBE0 | 12 | GPU4PySCF | 7.4484 | 1.3225 [1.2871, 1.3333] | complete; accuracy qualified |
| direct | PBE0 | 12 | GenerativeQC Standard | 2.9523 | 0.3716 [0.3708, 0.3740] | complete; accuracy qualified |
| direct | PBE0 | 48 | GPU4PySCF | 13.1874 | 3.3057 [3.1480, 3.3637] | complete; accuracy qualified |
| direct | PBE0 | 48 | GenerativeQC Standard | 9.3780 | 2.4242 [2.4231, 2.4329] | complete; accuracy qualified |
| direct | PBE0 | 96 | GPU4PySCF | 20.3324 | 6.7480 [6.7458, 6.7497] | complete; accuracy qualified |
| direct | PBE0 | 96 | GenerativeQC Standard | 19.1329 | 6.5739 [6.5659, 6.6033] | complete; accuracy qualified |
| direct | B3LYP | 3 | GPU4PySCF | 9.1876 | 1.7651 [1.6181, 1.8548] | complete; accuracy qualified |
| direct | B3LYP | 3 | GenerativeQC Standard | 2.0195 | 0.0832 [0.0831, 0.0840] | complete; accuracy qualified |
| direct | B3LYP | 12 | GPU4PySCF | 11.5024 | 1.8703 [1.6021, 2.1664] | complete; accuracy qualified |
| direct | B3LYP | 12 | GenerativeQC Standard | 3.0146 | 0.3993 [0.3956, 0.4181] | complete; accuracy qualified |
| direct | B3LYP | 48 | GPU4PySCF | 12.5903 | 3.1141 [3.0731, 3.1380] | complete; accuracy qualified |
| direct | B3LYP | 48 | GenerativeQC Standard | 9.2775 | 2.4347 [2.4251, 2.4353] | complete; accuracy qualified |
| direct | B3LYP | 96 | GPU4PySCF | 21.7066 | 6.9330 [6.8967, 7.0286] | complete; accuracy qualified |
| direct | B3LYP | 96 | GenerativeQC Standard | 19.7860 | 6.7968 [6.7700, 6.7979] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GPU4PySCF | 5.0012 | 0.5847 [0.5242, 0.6279] | complete; accuracy qualified |
| direct | r²SCAN | 3 | GenerativeQC Standard | 1.5659 | 0.0812 [0.0805, 0.0822] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GPU4PySCF | 6.3118 | 1.0760 [0.9303, 1.1666] | complete; accuracy qualified |
| direct | r²SCAN | 12 | GenerativeQC Standard | 2.0833 | 0.3476 [0.3427, 0.3612] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GPU4PySCF | 11.5464 | 3.0537 [3.0507, 3.0743] | complete; accuracy qualified |
| direct | r²SCAN | 48 | GenerativeQC Standard | 7.0007 | 2.0524 [2.0481, 2.0535] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GPU4PySCF | 18.9044 | 7.6070 [7.5745, 7.6149] | complete; accuracy qualified |
| direct | r²SCAN | 96 | GenerativeQC Standard | 15.5737 | 5.5448 [5.5332, 5.5508] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GPU4PySCF | 5.2525 | 0.9735 [0.7504, 1.1291] | complete; accuracy qualified |
| direct | r²SCAN-3c | 3 | GenerativeQC Standard | 1.4152 | 0.1038 [0.1025, 0.1045] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GPU4PySCF | 7.8045 | 1.4158 [1.4054, 1.4872] | complete; accuracy qualified |
| direct | r²SCAN-3c | 12 | GenerativeQC Standard | 3.7286 | 0.9219 [0.9155, 0.9226] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GPU4PySCF | 17.5900 | 5.9983 [5.9883, 6.0029] | complete; accuracy qualified |
| direct | r²SCAN-3c | 48 | GenerativeQC Standard | 11.4121 | 4.7789 [4.7673, 4.7872] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GPU4PySCF | 34.0975 | 15.1857 [15.1530, 15.2054] | complete; accuracy qualified |
| direct | r²SCAN-3c | 96 | GenerativeQC Standard | 32.1170 | 14.1938 [14.1910, 14.2079] | complete; accuracy qualified |
| direct | M06-2X | 3 | GPU4PySCF | 5.7164 | 0.9696 [0.9665, 1.0336] | complete; accuracy failed |
| direct | M06-2X | 3 | GenerativeQC Standard | 2.0198 | 0.0998 [0.0994, 0.1001] | complete; accuracy failed |
| direct | M06-2X | 12 | GPU4PySCF | 7.4930 | 1.5810 [1.2953, 1.6131] | complete; accuracy qualified |
| direct | M06-2X | 12 | GenerativeQC Standard | 3.0977 | 0.4130 [0.4119, 0.4217] | complete; accuracy qualified |
| direct | M06-2X | 48 | GPU4PySCF | 15.3317 | 4.0278 [3.8455, 4.0514] | complete; accuracy qualified |
| direct | M06-2X | 48 | GenerativeQC Standard | 10.8625 | 2.5807 [2.5724, 2.5908] | complete; accuracy qualified |
| direct | M06-2X | 96 | GPU4PySCF | 25.4480 | 8.8658 [8.8214, 8.8675] | complete; accuracy qualified |
| direct | M06-2X | 96 | GenerativeQC Standard | 23.1776 | 7.0372 [7.0340, 7.0410] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GPU4PySCF | 7.4787 | 1.6747 [1.6482, 1.7586] | complete; accuracy qualified |
| direct | ωB97M-V | 3 | GenerativeQC Standard | 4.4864 | 0.3445 [0.3442, 0.3449] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GPU4PySCF | 18.7378 | 4.1445 [4.1307, 4.4440] | complete; accuracy qualified |
| direct | ωB97M-V | 12 | GenerativeQC Standard | 10.8202 | 0.8488 [0.8239, 0.8641] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GPU4PySCF | 95.5331 | 31.9582 [31.9334, 31.9949] | complete; accuracy qualified |
| direct | ωB97M-V | 48 | GenerativeQC Standard | 81.5004 | 6.4698 [6.4673, 6.4930] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GPU4PySCF | 373.1626 | 120.1241 [120.0126, 120.2193] | complete; accuracy qualified |
| direct | ωB97M-V | 96 | GenerativeQC Standard | 259.1398 | 28.4964 [28.4519, 28.5622] | complete; accuracy qualified |
