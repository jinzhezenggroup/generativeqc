# Conventional iodine-ECP regression inputs

These five JSON files are byte-identical to the public reproduction bundle for
issue #3034, Gist `aa953baf1ab547472294493a9cc136d5`, revision
`0947ca38277e62d79efc68621701fbd114a140fa`, published by `Vibsteamer`:

https://gist.github.com/Vibsteamer/aa953baf1ab547472294493a9cc136d5/0947ca38277e62d79efc68621701fbd114a140fa

`SHA256SUMS` retains the upstream checksums for the vendored JSON subset. The
original geometries, native DZ/TZ bases, independent PySCF basis/ECP recipe and
archived RHF results are all retained. No shells, ECP channels or electrons have
been removed. Archived RHF agreement is not CCSD(T) or force qualification.

`tests/python/test_conventional_ecp_cbs_regression.py` independently runs PySCF
for the component/CBS energy and two-step directional-force checks. It also
isolates the reported physical HF and CCSD(T) force steps, using the effective
CBS controls rather than substituting a looser standalone calculation. Select
`-k physical_ccsdt_force_step` to qualify the original orbital-response failure
without waiting for the other CBS components. Long CUDA
tests require `GENERATIVEQC_ECP_CBS_3034_CUDA_TEST=1` and a finite Slurm allocation;
the input-integrity test is host-only.

The physical HF probes require a host C++ compiler and a verified compiler-cache
launcher. Set `CCACHE` to the selected sccache or ccache executable for the shared
native-probe fixture. Once CUDA qualification is explicitly enabled, a missing
probe compiler/cache is an error rather than a silently skipped HF regression.
