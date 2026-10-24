#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 ABSOLUTE_ARCHIVE ABSOLUTE_FRESH_DESTINATION" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage
archive=$1
destination=$2
[[ $archive = /* && $destination = /* && -f $archive && ! -e $destination ]] || usage
for tool in awk file find install sha256sum stat tar uname; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "missing offline-bootstrap prerequisite: $tool" >&2
    exit 2
  }
done

expected_name=sccache-v0.16.0-x86_64-unknown-linux-musl.tar.gz
expected_size=9490528
expected_sha256=aec995a83ad3dff3d14b6314e08858b7b73d35ca85a5bcf3d3a9ec07dee35588

[[ $(basename -- "$archive") == "$expected_name" ]] || {
  echo "unexpected sccache archive name" >&2
  exit 3
}
[[ $(stat -c %s "$archive") == "$expected_size" ]] || {
  echo "unexpected sccache archive size" >&2
  exit 3
}
observed_sha256=$(sha256sum "$archive" | awk '{print $1}')
[[ $observed_sha256 == "$expected_sha256" ]] || {
  echo "unexpected sccache archive SHA-256" >&2
  exit 3
}
[[ $(uname -m) == x86_64 ]] || {
  echo "sccache asset requires x86_64 Linux" >&2
  exit 4
}

mapfile -t members < <(tar -tzf "$archive")
[[ ${#members[@]} -gt 0 ]] || {
  echo "sccache archive is empty" >&2
  exit 5
}
for member in "${members[@]}"; do
  case "$member" in
    /* | ../* | */../* | */..)
      echo "unsafe sccache archive member" >&2
      exit 5
      ;;
  esac
done

mkdir -p -- "$destination/extracted" "$destination/bin"
tar -xzf "$archive" --no-same-owner -C "$destination/extracted"
mapfile -t candidates < <(find "$destination/extracted" -type f -name sccache -print)
[[ ${#candidates[@]} -eq 1 ]] || {
  echo "sccache archive must contain exactly one executable named sccache" >&2
  exit 5
}
install -m 0755 -- "${candidates[0]}" "$destination/bin/sccache"
version_output=$("$destination/bin/sccache" --version)
version=${version_output%%$'\n'*}
[[ $version == "sccache 0.16.0" ]] || {
  echo "extracted sccache version mismatch" >&2
  exit 6
}

{
  printf 'ASSET_NAME=%s\n' "$expected_name"
  printf 'ASSET_SIZE=%s\n' "$expected_size"
  printf 'ASSET_SHA256=%s\n' "$expected_sha256"
  printf 'ARCH=%s\n' "$(uname -m)"
  printf 'VERSION=%s\n' "$version"
  printf 'BINARY_SHA256=%s\n' "$(sha256sum "$destination/bin/sccache" | awk '{print $1}')"
} >"$destination/bootstrap-receipt.txt"
file "$destination/bin/sccache" >"$destination/file.txt"
if command -v ldd >/dev/null 2>&1; then
  ldd "$destination/bin/sccache" >"$destination/ldd.txt" 2>&1 || true
fi
printf 'SCCACHE_OFFLINE_BOOTSTRAP=READY\n'
printf 'SCCACHE_BIN=%s\n' "$destination/bin/sccache"
