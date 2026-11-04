#!/usr/bin/env bash
set -euo pipefail

if [[ "${GITHUB_ACTIONS:-}" != "true" || "${RUNNER_ENVIRONMENT:-}" != "github-hosted" ]]; then
  echo "CUDA disk reclamation is limited to GitHub-hosted runners."
  exit 0
fi

# CUDA objects, a restored compiler cache and the monolithic archive coexist at
# link time. Remove only unrelated hosted toolchains, never sources or caches.
workspace=${GITHUB_WORKSPACE:-.}
df -h "$workspace"
available_bytes=$(df -B1 --output=avail -- "$workspace" | tail -n 1 | tr -d '[:space:]') || available_bytes=""
# This is a cleanup heuristic, not a disk reservation or permission to skip builds.
if [[ "$available_bytes" =~ ^[0-9]+$ ]] && ((available_bytes >= 64 * 1024 * 1024 * 1024)); then
  echo "Skipping optional toolchain cleanup: at least 64 GiB is already available."
  exit 0
fi

cleanup_status=0
# The privileged timeout must be able to signal the privileged deletion group.
sudo timeout --kill-after=5s 120s rm -rf -- \
  /usr/local/lib/android /usr/share/dotnet /opt/ghc || cleanup_status=$?
case "$cleanup_status" in
  0) ;;
  124|137) echo "::warning::Optional toolchain cleanup reached its time bound; continuing with actual available space." ;;
  *) exit "$cleanup_status" ;;
esac
df -h "$workspace"
