# Offline compiler-cache bootstrap

The failed 2026-10-15 CPU preflight proves only that neither `sccache` nor
`ccache` was visible through that shell's `PATH`. It does not prove that every
location in the image or the persistent issue directory lacks a usable launcher.
Older #1765 build records name these persistent candidates, but retain no current
file identity or ABI evidence:

```text
/inspire/qb-ilm/project/chemicalreaction/czxs25220150/issue1765-attribution-20261014/runtime/bin/sccache
/inspire/qb-ilm/project/chemicalreaction/czxs25220150/issue1765-attribution-20261014/runtime/local/bin/sccache
```

Also inspect the conventional image location `/usr/local/bin/sccache`; current
evidence does not establish that the older build used it. In a separately
authorized future window, inspect the candidates before any bootstrap:

```bash
for candidate in \
  /inspire/qb-ilm/project/chemicalreaction/czxs25220150/issue1765-attribution-20261014/runtime/bin/sccache \
  /inspire/qb-ilm/project/chemicalreaction/czxs25220150/issue1765-attribution-20261014/runtime/local/bin/sccache \
  /usr/local/bin/sccache
do
  if test -x "$candidate"; then
    "$candidate" --version
    stat -c 'size=%s mode=%a inode=%i' "$candidate"
    sha256sum "$candidate"
    file "$candidate"
    ldd "$candidate" || true
  fi
done
```

Select one only if it reports `sccache >=0.16.0` and matches the current Linux
architecture/ABI. Launcher presence alone is insufficient; retain pre/post
`--show-stats` and the actual wrapped compile/link commands.

If no retained candidate qualifies, prepare this official Mozilla release asset
outside the resource window and carry it in with the observer archive:

```text
name: sccache-v0.16.0-x86_64-unknown-linux-musl.tar.gz
source: https://github.com/mozilla/sccache/releases/download/v0.16.0/sccache-v0.16.0-x86_64-unknown-linux-musl.tar.gz
size: 9490528 bytes
sha256: aec995a83ad3dff3d14b6314e08858b7b73d35ca85a5bcf3d3a9ec07dee35588
platform: Linux x86_64, portable musl build
```

Do not download this asset from inside the bounded build instance. Before a
future upload, fetch it outside the resource window from the exact URL above,
then verify filename, byte size and SHA-256 locally:

```bash
test "$(stat -c %s sccache-v0.16.0-x86_64-unknown-linux-musl.tar.gz)" = 9490528
printf '%s  %s\n' \
  aec995a83ad3dff3d14b6314e08858b7b73d35ca85a5bcf3d3a9ec07dee35588 \
  sccache-v0.16.0-x86_64-unknown-linux-musl.tar.gz | sha256sum -c -
```

Record the transfer archive's independent size/hash. On the remote side, run
only inside a fresh directory under the authorized issue root:

```bash
bash benchmarks/rhf_provider_retention_fake/prepare_sccache_offline.sh \
  /absolute/input/sccache-v0.16.0-x86_64-unknown-linux-musl.tar.gz \
  /absolute/fresh/output/sccache-0.16.0

GENERATIVEQC_RHF_PROVIDER_CACHE=/absolute/fresh/output/sccache-0.16.0/bin/sccache \
  benchmarks/rhf_provider_retention_fake/build_and_run.sh \
  /absolute/fresh/result-directory \
  <full-observer-commit>
```

`prepare_sccache_offline.sh` performs no network access. It rejects the wrong
name, size, hash, architecture, unsafe tar members, duplicate executables and a
version other than `sccache 0.16.0`. It records the extracted binary hash plus
`file`/`ldd` output. The build script accepts the absolute override only after
verifying that it is executable and reports a supported cache/version.

This plan does not authorize another resource window, an installation into the
image or shared system paths, a cache clear, a shared-server stop, or an uncached
compile. A future window must still re-query quota/image/policy/timer and prove
the compiler, SDK headers and cache are available before compilation.
