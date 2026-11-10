# Original KS final scientific gates

Clean measured source `926c778cb47c99b250378727832924dcea72d943`, native input `baf0bc3589f99d2caa332ba48e36a6a90425cf8f209ecf2422dfa3896876a563`. Non-fast-compile Release/sm120 library `9a50430e953fc7322edde32eaf63ec414a22a7b072b17eb2c9e0ba99325d56e8`. The diagnostic build and its measurements remain separately identified.

All original UKS (8 controls, 40 physical states), range-only PBE DF-primary/Exact-LR RSH (4 controls, 16 E/F/states), and water96 default/tighter-only controls pass on their measured devices. Water96 retains the original Treutler3 grid, def2-SVP orbital/auxiliary bases, FP64, DF512MiB, tile1024 and 1e-8/1e-7 E/F gates. Independent canonical-density and force gates remain 1e-8 and 3e-5. Exact metrics, source/library and Slurm UUIDs are in the compressed records.

Same-device PRO6000 parent/default reproduces 45/288 force failures at 4.794283556819323e-7; parent/tighter-only gives 1.3814727140015748e-11 on the same original grid. Reviewed repair/default gives 3.398952230782015e-10. This supports stopping sensitivity in the identified current parent, not historical introduction by #2910. The raw-AO UKS cold-start failures and rejected oracle experiments retain their own identities and failed status.

The user explicitly excludes A100 from completion requirements; it remains unmeasured and unqualified. These numerical controls do not prove actual Graph launches from mask requests, promote performance/universal production support, or waive required CI. Keep the PR draft/off queue/without auto-merge until original scope and required checks are resolved.

## Complete lossless recovery

All selected receipt values, complete SCF histories and full force/reference vectors remain in `be5c9a904c2fe31070d473689299e1e5aa2ae334`.

```bash
git show be5c9a904c2fe31070d473689299e1e5aa2ae334:benchmarks/results/ks-original-final-complete-20261110/samples.json.xz > /tmp/ks-final-samples.json.xz
```

SHA256 `e2429224864582911f285fadf4f4686bbe9260fc81140e4af5a4d0e9a106d857`, 558640 bytes. The selected record authenticates these exact bytes. No rounding, status rewriting, evidence-cap increase, unrelated-evidence deletion, external archive or release is used.
