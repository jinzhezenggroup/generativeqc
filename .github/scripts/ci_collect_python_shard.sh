#!/usr/bin/env bash
# Source after ci_select_python_shard.sh. Keep shard ownership and pytest's
# deselection/ordering plugins unchanged; only avoid repeated directory discovery.
collection_targets=("${test_targets[@]}")
collection_args=()

ci_prepare_python_collection() {
  case "${GENERATIVEQC_CI_SHARD:?}" in
    core-a|core-b|core-c) ;;
    *) return ;;
  esac
  # Directory discovery has lexical order. Retain explicit targets whenever a
  # selector supplies a different priority order, duplicates, or node IDs.
  local path relative previous="" LC_ALL=C
  local -A selected=()
  for path in "${test_targets[@]}"; do
    relative="${path#tests/python/}"
    if [[ "$path" != tests/python/test_*.py || "$path" == *::* || "$relative" == */* ]]; then
      return
    fi
    if [[ ! -f "$path" ]]; then
      echo "::error::selected Python test is missing: $path" >&2
      return 2
    fi
    if [[ -n "$previous" && ! "$previous" < "$path" ]]; then
      return
    fi
    previous="$path"
    selected["$path"]=1
  done
  if (( ${#selected[@]} == 0 )); then
    echo "::error::core shard selected no tests" >&2
    return 2
  fi
  # Standard --ignore acts before import/parameter generation. Include untracked
  # entries in the complement, so directory discovery cannot admit tests or
  # child conftests absent from the selector's exact flat-file inventory. Ignore
  # whole child directories, including symlinks, without traversing them.
  local inventory
  inventory="$(mktemp)" || return
  if ! find tests/python -mindepth 1 -maxdepth 1 -print0 > "$inventory"; then
    rm -f "$inventory"
    echo "::error::could not enumerate Python collection exclusions" >&2
    return 2
  fi
  while IFS= read -r -d '' path; do
    if [[ -z "${selected[$path]:-}" ]]; then
      collection_args+=("--ignore=$path")
    fi
  done < "$inventory"
  rm -f "$inventory"
  collection_targets=(tests/python)
}
ci_prepare_python_collection || return "$?"
unset -f ci_prepare_python_collection
