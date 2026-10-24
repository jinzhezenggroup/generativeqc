#!/usr/bin/env bash
# Authenticate the tested PR merge before applying the unchanged evidence policy.
set -eu

# The event base can predate the target actually merged for this run.
read -r merge base head extra <<< "$(git rev-list --parents -n 1 HEAD)"
if [[ -z "$base" || -z "$head" || -n "$extra" || "$head" != "$EVIDENCE_HEAD" ]]; then
  echo "Expected a two-parent PR test merge with the requested head" >&2
  exit 1
fi
if [[ -z "$EVIDENCE_BASE" ]] || ! git cat-file -e "$EVIDENCE_BASE^{commit}" ||
   ! git merge-base --is-ancestor "$EVIDENCE_BASE" "$base"; then
  echo "Event evidence base must be an available ancestor of the tested base" >&2
  exit 1
fi
python3 tools/evidence.py check-change --base "$base"
