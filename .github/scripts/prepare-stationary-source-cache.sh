#!/usr/bin/env bash
set -euo pipefail
# Atomic cache entries retain TemporaryDirectory's private mode inside manylinux.
# Transfer only this job's source cache back to the host runner for archive/save;
# preserve modes and never follow links into unrelated cache or workspace data.
cache="${RUNNER_TEMP:?}/stationary-aot-source-cache"
if [[ -d "$cache" && ! -L "$cache" ]]; then
  sudo chown -R -P --no-dereference "$(id -u):$(id -g)" -- "$cache"
fi
