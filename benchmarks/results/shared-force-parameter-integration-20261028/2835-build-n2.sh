set -euo pipefail
cd /home/jzzeng/tmp/qc-pr-unblock-2835-20261028
export PATH=/home/jzzeng/tmp/qc-1123-shared-aot-20261027/.venv/bin:/group/software/cuda-12.9.1/bin:/usr/bin:/bin:/home/jzzeng/miniconda3/bin
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=python:.
sccache --version
sccache --show-stats > .artifacts/cache-before.txt
trap 'sccache --show-stats > .artifacts/cache-after.txt' EXIT
cmake -S . -B build/integration -G Ninja \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
 -DCMAKE_PREFIX_PATH=/home/jzzeng/miniconda3 -DOPENSSL_ROOT_DIR=/home/jzzeng/miniconda3 \
 -Dnlohmann_json_DIR=/home/jzzeng/miniconda3/share/cmake/nlohmann_json \
 -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
 -DCMAKE_CXX_COMPILER_LAUNCHER=/home/jzzeng/tmp/qc-1123-shared-aot-20261027/.venv/bin/sccache \
 -DCMAKE_CUDA_COMPILER_LAUNCHER=/home/jzzeng/tmp/qc-1123-shared-aot-20261027/.venv/bin/sccache \
 -DPython3_EXECUTABLE=/home/jzzeng/tmp/qc-1123-shared-aot-20261027/.venv/bin/python \
 -DGENERATIVEQC_ENABLE_CUDA=ON -DGENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=120-real \
 -DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=OFF -DGENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA=ON \
 -DGENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT=OFF -DGENERATIVEQC_ENABLE_AOT_SHELLS=OFF \
 -DGENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT=ON \
 -DGENERATIVEQC_CUDA_COMPILE_JOBS=6 -DGENERATIVEQC_AOT_COMPILE_JOBS=6 \
 -DGENERATIVEQC_AOT_SPLIT_COMPILE_THREADS=2 -DGENERATIVEQC_CUDA_SPLIT_COMPILE_THREADS=2 \
 -DGENERATIVEQC_BUILD_TESTS=ON -DGENERATIVEQC_BUILD_CLI=ON \
 '-DGENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS=LDA_C_VWN_1;LDA_C_VWN_4;GGA_X_APBE;MGGA_X_LTA;MGGA_X_KTBM_0;MGGA_X_KTBM_1' \
 '-DGENERATIVEQC_STATIONARY_AOT_PROFILES=pbe_rks;pbe_uks;wb97mv_rks;wb97mv_uks;libxc_196636_rks;libxc_196636_uks;libxc_196639_rks;libxc_196639_uks;libxc_196792_rks;libxc_196792_uks;libxc_196809_rks;libxc_196809_uks;libxc_197343_rks;libxc_197343_uks;libxc_197344_rks;libxc_197344_uks' \
 > .artifacts/configure.log 2>&1
ninja -C build/integration -t commands generativeqc_native_stationary generativeqc_libxc_semilocal_cuda > .artifacts/compiler-commands.txt
cmake --build build/integration --parallel 20 --target generativeqc generativeqc_cli generativeqc_mp2_contract_tests generativeqc_ump2_contract_tests generativeqc_scf_diagnostic_tests generativeqc_cosx_cuda_tests > .artifacts/build.log 2>&1
sccache --show-stats > .artifacts/cache-after.txt
