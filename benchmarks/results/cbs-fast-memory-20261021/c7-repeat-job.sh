#!/usr/bin/env bash
set -euo pipefail
cd /data/jzzeng/qc-cbs-fast-memory-20261021
for variant in candidate baseline; do bash run-one.sh "$variant" 7 fast c7-repeat; done
