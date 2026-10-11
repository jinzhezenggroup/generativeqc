#!/usr/bin/env bash
set -euo pipefail
root=/data/jzzeng/qc-ccsdt-profile-master-20261010-37227b53
deps=/data/jzzeng/issue1972-20261005/deps
export PATH="$deps/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/home/jzzeng/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$deps/lib:/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
campaign="$root/p2-lambda-frontier-master"
for side in baseline candidate; do
  out="$campaign/$side"
  source="$out/source"
  mkdir -p "$out"
  # Archives retain commit mtimes; never extract a different tree over old objects.
  if [[ -e "$source" ]]; then
    printf 'Refusing to reuse an existing staged source: %s\n' "$source" >&2
    exit 1
  fi
  mkdir "$source"
  tar -xf "$campaign/master-7685.tar" -C "$source"
  if [[ "$side" == candidate ]]; then
    git -C "$source" apply "$campaign/master-prototype.patch"
  fi
  cd "$source"
  export CCACHE_BASEDIR="$source" PYTHONPATH="$source/python:$source:$deps"
  ccache --version > "$out/ccache-version.txt"
  ccache --show-stats > "$out/ccache-before.txt"
  cmake -S . -B build-cuda -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DPython3_EXECUTABLE=/home/jzzeng/miniconda3/bin/python \
    -DGENERATIVEQC_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DGENERATIVEQC_CUDA_FAST_COMPILE=OFF \
    -DGENERATIVEQC_BUILD_TESTS=OFF -DGENERATIVEQC_BUILD_CLI=OFF \
    -DGENERATIVEQC_ENABLE_AOT_SHELLS=ON -DGENERATIVEQC_AOT_PROFILE=portable_cuda \
    -DGENERATIVEQC_BUNDLE_XTB_OPENBLAS=OFF \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    > "$out/configure.log" 2>&1
  grep -m4 'LAUNCHER = ccache' build-cuda/build.ninja > "$out/compiler-launchers.txt"
  /usr/bin/time -v -o "$out/build.time.txt" \
    cmake --build build-cuda --target generativeqc -j12 > "$out/build.log" 2>&1
  dispatch=build-cuda/CMakeFiles/generativeqc_direct_native.dir/src/scf/cuda/direct_bounded_fallback.cu.o
  stat -c '%y %n' src/scf/cuda/direct_bounded_fallback.cu "$dispatch" > "$out/dispatch-timestamps.txt"
  sha256sum src/scf/cuda/direct_bounded_fallback.cu "$dispatch" > "$out/dispatch.sha256"
  nm -C "$dispatch" > "$out/dispatch-symbols.txt"
  grep ' U .*launch_direct_force_class_domains' "$out/dispatch-symbols.txt" > "$out/class-domain-call.txt"
  cp -L build-cuda/libgenerativeqc.so "$out/libgenerativeqc.so"
  ln -sfn libgenerativeqc.so "$out/libgenerativeqc.so.0"
  ccache g++ -std=c++20 -O2 -DGENERATIVEQC_HAS_CUDA=1 -Iinclude -Isrc \
    -I/group/software/cuda-12.9.1/include \
    -c benchmarks/df_ccsdt_force_endpoint.cpp -o "$out/endpoint.o"
  g++ "$out/endpoint.o" "$out/libgenerativeqc.so" -Wl,-rpath,"$out" -o "$out/endpoint"
  if [[ "$side" == candidate ]]; then
    ccache g++ -std=c++20 -O2 -DGENERATIVEQC_HAS_CUDA=1 -Iinclude -Isrc \
      -I/group/software/cuda-12.9.1/include \
      -c tests/native/test_cuda_fock_provider.cpp -o "$out/provider.o"
    g++ "$out/provider.o" "$out/libgenerativeqc.so" -L/group/software/cuda-12.9.1/lib64 \
      -lcudart -Wl,-rpath,"$out" -o "$out/provider"
    sha256sum "$out/provider" > "$out/provider.sha256"
  fi
  sha256sum "$out/libgenerativeqc.so" "$out/endpoint" > "$out/binaries.sha256"
  sha256sum "$campaign/master-7685.tar" "$campaign/master-prototype.patch" > "$out/source-archive.sha256"
  ccache --show-stats > "$out/ccache-after.txt"
  { date --iso-8601=seconds; hostname; g++ --version | head -1; nvcc --version;
    cmake --version | head -1; ccache --version; } > "$out/toolchain.txt"
  grep -E 'CMAKE_BUILD_TYPE:|CMAKE_CUDA_ARCHITECTURES:|GENERATIVEQC_AOT_PROFILE:|GENERATIVEQC_CUDA_FAST_COMPILE:' \
    build-cuda/CMakeCache.txt > "$out/configuration.txt"
  printf '%s\n' complete > "$out/status.txt"
done
