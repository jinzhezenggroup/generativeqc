# Decision: require explicit offline CUDA build provenance

Status: implemented
Date: 2026-11-08

## Problem

The native CLI packager labeled every archive CUDA 12.9 / SM120 even though
the release preset accepts caller-selected architectures. Packaging a different
build could therefore misstate both its deployment requirements and image set.
The integral production-profile parser separately retained schema-v1 branches
that invented SM120 defaults and could treat missing acceptance metadata as
architecture qualification.

## Decision

- CMake installs a small `native-cli-build.json` beside the CLI's notices.
  It records the resolved compiler version and ordered architecture image list.
  The packager requires that provenance rather than accepting free-form label
  overrides or probing the packaging host's GPU. Package version and platform
  must match. Archive names retain the historical CUDA major/minor convention;
  packaged metadata retains the full compiler version and all real/virtual
  suffixes. Multi-image packages list every requested image.
- Production shell manifests use only schema v2. Before the first release,
  rejecting the obsolete format is preferable to retaining an implicit target
  or inventing a conversion that certifies unverified tuning evidence. The
  repository's production shell manifest and tuning publisher already use v2;
  the custom test consumer is migrated in the same change.
- Resolve a manifest default only when the caller omits a target. An explicitly
  selected target still receives the existing exact/compatible/portable checks.
  An omitted target still requires an explicit or unambiguous manifest choice.

## Rejected alternatives

- Replacing SM120 with another default merely moves the hidden-target problem.
- New `--cuda-version` / `--architecture` packaging overrides permit arbitrary
  relabeling and lose the installed build's real/virtual image intent.
- Relaxing the separate NVCC closure collector's SM80 whitelist would enlarge
  its explicitly qualified dryrun/closure domain without new toolchain evidence.
  That restriction remains unchanged, as do static architecture resource limits
  and their conservative unknown-target fallback.

## Invariants

This changes offline selection/packaging, not generated mathematics, runtime
kernel defaults, shell coverage, force accuracy gates, or numerical/performance
qualification. Recording an architecture does not certify it. Real/virtual
suffixes are compilation provenance and must not be collapsed into profile IDs.
No GPU is needed to package an installed build or resolve an offline manifest.

## Evidence

- `tests/python/test_native_cli_package.py` runs the actual CMake metadata
  producer and packages existing Linux ELF fixtures for single/multiple images,
  distinct real/virtual images, and different CUDA versions. It checks archive
  identity, README and metadata, reproducibility, and fail-closed errors.
- `tests/python/test_production_profile.py` covers schema-v1 rejection with and
  without acceptance evidence, explicit targets without defaults, portable
  fallback, and manifest-only omitted-target resolution.
- `tests/python/test_codegen_profiles.py` retains current-manifest coverage,
  exact/compatible/portable selection, ambiguity and generator-ABI rejection.
- A one-off comparison against parser source at
  `4f2441dbca5748027f15d2e0405cc5561c218928` found all 14 resolved dataclass
  payloads identical for the current production shell manifest: omitted target
  and SM80/86/89/90/100/120, each with `auto` and `portable_cuda`. This is migration
  evidence, not a permanent generated-source fingerprint test.

## Revisit when

A new supported distribution platform or compiler backend needs its own
validated deployment contract, or actual toolchain closure qualification is
extended beyond the current offline collector's supported domain.
