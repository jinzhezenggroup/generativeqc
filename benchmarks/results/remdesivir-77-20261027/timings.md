# Remdesivir-77 endpoint timings

Absolute latency of one cold and three frozen-density warm complete
energy-plus-force calls per process. Failure rows retain the reason that
prevented a measurement; they are not imputed or hidden.

| Basis | Mode | Method | Engine | Cold (s) | Warm median [range] (s) | Status |
| --- | --- | --- | --- | ---: | --- | --- |
| def2-svp | direct | hf | GenerativeQC Standard | 103.1948 | 14.2005 [14.1978, 14.2066] | complete |
| def2-svp | direct | hf | GPU4PySCF 1.8.1 | 60.2558 | 11.8893 [11.8884, 11.8933] | complete |
| def2-svp | df | hf | GenerativeQC Standard | 25.8226 | 5.9296 [5.9275, 5.9323] | complete |
| def2-svp | df | hf | GPU4PySCF 1.8.1 | 46.4441 | 19.2170 [19.2164, 19.2192] | complete |
| def2-svp | direct | pbe | GenerativeQC Standard | 72.8473 | 33.4342 [33.4304, 33.4693] | complete |
| def2-svp | direct | pbe | GPU4PySCF 1.8.1 | 46.5778 | 15.4969 [15.4930, 15.5104] | complete |
| def2-svp | df | pbe | GenerativeQC Standard | 29.9593 | 5.1826 [5.1783, 5.2174] | complete |
| def2-svp | df | pbe | GPU4PySCF 1.8.1 | 33.6686 | 9.0601 [9.0598, 9.0606] | complete |
| def2-svp | direct | pbe0 | GenerativeQC Standard | 127.7354 | 43.3856 [43.3533, 43.4327] | complete |
| def2-svp | direct | pbe0 | GPU4PySCF 1.8.1 | 68.1439 | 23.6090 [23.6043, 23.6164] | complete |
| def2-svp | df | pbe0 | GenerativeQC Standard | 43.2812 | 11.7048 [11.7016, 11.7287] | complete |
| def2-svp | df | pbe0 | GPU4PySCF 1.8.1 | 54.9960 | 28.1040 [28.0993, 28.1255] | complete |
| def2-svp | direct | b3lyp | GenerativeQC Standard | 148.1899 | 43.5582 [43.5132, 43.5673] | complete |
| def2-svp | direct | b3lyp | GPU4PySCF 1.8.1 | 70.9178 | 16.3531 [16.3472, 16.3665] | complete |
| def2-svp | df | b3lyp | GenerativeQC Standard | 48.3373 | 11.7584 [11.7516, 11.7907] | complete |
| def2-svp | df | b3lyp | GPU4PySCF 1.8.1 | 56.5807 | 23.4713 [23.4464, 23.4728] | complete |
| def2-svp | direct | r2scan | GenerativeQC Standard | 81.5175 | 34.4515 [34.4488, 34.4863] | complete |
| def2-svp | direct | r2scan | GPU4PySCF 1.8.1 | 58.1700 | 22.4551 [22.4537, 22.4704] | complete |
| def2-svp | df | r2scan | GenerativeQC Standard | 40.1646 | 6.1449 [6.1403, 6.1842] | complete |
| def2-svp | df | r2scan | GPU4PySCF 1.8.1 | 46.1124 | 15.9779 [15.9766, 15.9989] | complete |
| def2-svp | direct | r2scan-3c | GenerativeQC Standard | 253.7484 | 135.0966 [135.0835, 135.1184] | complete |
| def2-svp | direct | r2scan-3c | GPU4PySCF 1.8.1 | 224.9915 | 118.9168 [118.0023, 118.9276] | complete |
| def2-svp | df | r2scan-3c | GenerativeQC Standard | - | - | failed: RuntimeError: batched item failures: 0: generated force failed (RuntimeError): GENERATIVEQC error 7: density-fitted stationary response exceeded the p |
| def2-svp | df | r2scan-3c | GPU4PySCF 1.8.1 | 148.4298 | 74.9574 [74.9352, 74.9582] | complete |
| def2-svp | direct | m06-2x | GenerativeQC Standard | 130.2515 | 44.1650 [44.1025, 44.1855] | complete |
| def2-svp | direct | m06-2x | GPU4PySCF 1.8.1 | 80.2849 | 21.3130 [21.3056, 21.3175] | complete |
| def2-svp | df | m06-2x | GenerativeQC Standard | 50.2606 | 12.5705 [12.5663, 12.6032] | complete |
| def2-svp | df | m06-2x | GPU4PySCF 1.8.1 | 67.4911 | 28.4281 [28.4221, 28.4390] | complete |
| def2-svp | direct | wb97m-v | GenerativeQC Standard | - | - | failed: RuntimeError: batched item failures: 0: generated force failed (RuntimeError): GENERATIVEQC error 7: out of memory |
| def2-svp | direct | wb97m-v | GPU4PySCF 1.8.1 | 436.1061 | 111.3245 [111.3089, 111.3281] | complete |
| def2-svp | df | wb97m-v | GenerativeQC Standard | 1130.4575 | 193.4567 [193.4541, 193.4991] | complete |
| def2-svp | df | wb97m-v | GPU4PySCF 1.8.1 | 409.3858 | 122.5499 [122.5361, 122.5593] | complete |
| def2-tzvp | direct | hf | GenerativeQC Standard | - | - | failed: RuntimeError: batched item failures: 0: out of memory |
| def2-tzvp | direct | hf | GPU4PySCF 1.8.1 | 906.4129 | 312.2932 [311.3858, 313.1746] | complete |
| def2-tzvp | df | hf | GenerativeQC Standard | - | - | failed: RuntimeError: batched item failures: 0: out of memory |
| def2-tzvp | df | hf | GPU4PySCF 1.8.1 | - | - | failed: CUDARuntimeError: cudaErrorMemoryAllocation: out of memory |
| def2-tzvp | direct | pbe | GenerativeQC Standard | - | - | failed: RuntimeError: GENERATIVEQC error 7: std::bad_alloc |
| def2-tzvp | direct | pbe | GPU4PySCF 1.8.1 | 458.0405 | 291.4620 [289.9626, 291.5255] | complete |
| def2-tzvp | df | pbe | GenerativeQC Standard | - | - | failed: RuntimeError: GENERATIVEQC error 1: DF memory budget cannot hold the metric and one contraction tile |
| def2-tzvp | df | pbe | GPU4PySCF 1.8.1 | 115.1564 | 55.6609 [55.6577, 55.6719] | complete |
| def2-tzvp | direct | pbe0 | GenerativeQC Standard | - | - | failed: RuntimeError: GENERATIVEQC error 7: std::bad_alloc |
| def2-tzvp | direct | pbe0 | GPU4PySCF 1.8.1 | 911.6186 | 202.6838 [202.6567, 339.6230] | complete |
| def2-tzvp | df | pbe0 | GenerativeQC Standard | - | - | failed: interrupted: no endpoint retained (point timeout or job time limit) |
| def2-tzvp | df | pbe0 | GPU4PySCF 1.8.1 | - | - | failed: CUDARuntimeError: cudaErrorMemoryAllocation: out of memory |
| def2-tzvp | direct | b3lyp | GenerativeQC Standard | - | - | failed: interrupted: no endpoint retained (point timeout or job time limit) |
| def2-tzvp | direct | b3lyp | GPU4PySCF 1.8.1 | 447.4045 | 156.7132 [156.7051, 156.7170] | complete |
| def2-tzvp | df | b3lyp | GenerativeQC Standard | - | - | failed: interrupted: no endpoint retained (point timeout or job time limit) |
| def2-tzvp | df | b3lyp | GPU4PySCF 1.8.1 | - | - | failed: CUDARuntimeError: cudaErrorMemoryAllocation: out of memory |
| def2-tzvp | direct | r2scan | GenerativeQC Standard | - | - | failed: interrupted: no endpoint retained (point timeout or job time limit) |
| def2-tzvp | direct | r2scan | GPU4PySCF 1.8.1 | 249.8122 | 152.7985 [152.7953, 152.7991] | complete |
| def2-tzvp | df | r2scan | GenerativeQC Standard | - | - | failed: RuntimeError: batched item failures: 0: SCF did not converge |
| def2-tzvp | df | r2scan | GPU4PySCF 1.8.1 | 92.7653 | 42.0557 [42.0512, 42.0606] | complete |
| def2-tzvp | direct | m06-2x | GenerativeQC Standard | - | - | failed: interrupted: no endpoint retained (point timeout or job time limit) |
| def2-tzvp | direct | m06-2x | GPU4PySCF 1.8.1 | 473.9915 | 170.6824 [170.6772, 170.6877] | complete |
| def2-tzvp | df | m06-2x | GenerativeQC Standard | - | - | failed: interrupted: no endpoint retained (point timeout or job time limit) |
| def2-tzvp | df | m06-2x | GPU4PySCF 1.8.1 | - | - | failed: CUDARuntimeError: cudaErrorMemoryAllocation: out of memory |
| def2-tzvp | direct | wb97m-v | GenerativeQC Standard | - | - | failed: interrupted: no endpoint retained (point timeout or job time limit) |
| def2-tzvp | direct | wb97m-v | GPU4PySCF 1.8.1 | 1107.1385 | 501.8673 [501.8660, 501.8819] | complete |
| def2-tzvp | df | wb97m-v | GenerativeQC Standard | - | - | failed: interrupted: no endpoint retained (point timeout or job time limit) |
| def2-tzvp | df | wb97m-v | GPU4PySCF 1.8.1 | - | - | failed: CUDARuntimeError: cudaErrorMemoryAllocation: out of memory |
