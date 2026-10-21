#!/usr/bin/env bash
set -euo pipefail
cd /data/jzzeng/qc-cbs-fast-memory-20261021
bash run-one.sh baseline 7 fast c7-ablation
bash run-one.sh candidate 7 fast c7-fast
bash run-one.sh candidate 7 standard c7-standard
