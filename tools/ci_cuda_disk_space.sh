#!/usr/bin/env bash
set -euo pipefail

if [[ "${GITHUB_ACTIONS:-}" != "true" || "${RUNNER_ENVIRONMENT:-}" != "github-hosted" ]]; then
  echo "CUDA disk reclamation is limited to GitHub-hosted runners."
  exit 0
fi

# CUDA objects, a restored compiler cache and the monolithic archive coexist at
# link time. Remove only unrelated hosted toolchains, never sources or caches.
df -h "${GITHUB_WORKSPACE:-.}"
sudo rm -rf -- /usr/local/lib/android /usr/share/dotnet /opt/ghc
df -h "${GITHUB_WORKSPACE:-.}"
