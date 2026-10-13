# Native CUDA grid package closure

Agent: dot

PR #2357 moved existing grid ABI declarations and the shared capacity calculation
into `src/dft/cuda_grid_api.hpp` and `src/dft/cuda_grid_storage.hpp`. The native
build sees those files through the checkout, but the installed compiler did not
ship them or include them in the emitted grid runtime's dependency inventory.

The repair adds both owned headers to the wheel force-include mapping and
`emit_grid_source`'s header list. The API header's local dependencies are the
already-packaged AO work record, grid task view, and tensor metrics headers.
The storage header uses only standard-library headers. The first runtime source
and final public-header include-root sentinels retain their existing positions.

The regression materializes exactly the declared wheel payload, resolves every
local include without a neighboring checkout, and compiles/links the actual
emitted production grid with only packaged include roots. The existing CUDA
compile job invokes this test with no visible device. No GPU qualification or
runtime fallback is introduced. A header-name assertion alone was rejected:
successful source emission would not demonstrate that the installed translation
unit and its declarations link.

The stationary capacity report retains its whole fail-closed CMake contract.
The packaged s/p/d target wiring and profile policy are byte-identical to the
previous contract after accounting for the new native static integration and
disabled-AOT stub. The digest is refreshed to include those reviewed additions;
mutation controls cover both packaged component-domain and native target links.
This does not promote frozen scientific or performance evidence.

Local compilation and tests were not executed in the ENOSPC review environment.
Normal hosted CI must provide executable results for this repair.
