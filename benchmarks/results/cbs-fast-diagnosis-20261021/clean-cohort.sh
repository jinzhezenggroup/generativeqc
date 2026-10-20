#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/qc-cbs-fast-mixed-20261020/fast-diagnosis-20261021
bash "$root/run-candidate.sh" baseline 4 0 -clean-standard standard
bash "$root/run-candidate.sh" lambda-mixed-q32 4 0 -clean
bash "$root/run-candidate.sh" baseline 4 0 -clean
bash "$root/run-candidate.sh" baseline 2 0 -clean
bash "$root/run-candidate.sh" lambda-mixed-q32 2 0 -clean
