# Proposal: share low-order long-range force roots across components and centers

Status: proposed; independent device and complete endpoint qualification pending
Date: 2026-10-03

## Measured problem

The complete 96-atom WB97M-V diagnostic trace in n1 Slurm 5493 contains two
bounded force traversals: 28.802885 s and 188.013665 s. The frozen source's
full-then-LR launch order identifies these as full-range J/K and LR exchange;
Nsight records their order, not radial arguments. VV10 separately takes
114.341840 s in SCF and 158.525478 s in forces. All geometry kernels together
take 36.638392 s, so fusing geometry cannot remove the entire nonlocal drain.

The profiled source is `8bfde172cf9595e7cd82a9d2bcf856a9a07dcc37`, with library
`7345024fb005320f605027e5b2a1aaac555e51ba670c73dd17c27707cb8505f1`.
Its complete warm diagnostic endpoint is 630.426497 s. Cold and warm pass an
independent clean reference96 comparison (maximum energy error 1.13e-10 Eh and
force error 3.27e-10 Eh/Bohr). This is a profile, not a clean timing sample for
the new candidate. Raw trace, SQLite, hashes and the verifier are retained in
the storage experiment's `.artifacts/wb97m-force-storage/profile-review96/` and
`profile96-summary.json`.

## Candidate

Start independently at master `d35ae539f645cb5e8b9c0c2fb7a426c5e2ad08e8`,
which already shares low-order full-range J/K geometry. Reuse its compiler-owned
weighted force roots for LR exchange of total angular order zero through three.
The bounded shell scheduler builds one exchange weight vector and evaluates one
radial moment ladder per primitive quartet, then accumulates all independent
center derivatives. It no longer repeats the Cartesian Dual3 recurrence for
each component and unique atom in this closed low-order domain.

This does not introduce a new integral formula. At fixed primitive exponents
and omega, LR moments obey `dM_n/dT = -M_(n+1)`, exactly the derivative relation
consumed by the existing weighted roots. The scalar moment owner remains
`integrals/range_moments.hpp`; component/center algebra remains compiler-owned.
The dependency checker admits only that exact shared numerical header to the
native contraction adapters, without admitting the CPU tensor/oracle layer.

## Boundaries and gates

Full-range consumers retain their two independent source channels. Short-range,
fused RSH and higher angular orders retain their existing bounded/AOT paths.
The molecular omega=0.3 path still reconstructs SR as Full minus LR, with each
functional coefficient applied by its existing owner. Precision, screening,
basis normalization, source meanings and allocation bounds are unchanged.
Invalid moment inputs propagate nonfinite output to the existing failure gate.

AO contraction and atom accumulation order change for the selected LR tasks,
so bitwise LR equality is not claimed. Require independent CPU displaced
integrals at 3e-8 derivative error, complete RKS/UKS and changed-geometry gates
at 1e-8 Eh / 1e-7 Eh/Bohr, all repeats, and controlled full-grid 24/96 endpoints.
The native tests call the actual bounded molecular boundary as well as the
canonical provider, with a four-center d/p/s/s fixture and repeated atom binding.

The existing host control checks 13,026,816 full-range coordinates bitwise equal
to the retained scalar workers. Focused host checks total 110 passing tests.
The untouched latest-master library is
`c52dd96d9c35ea6f035e5d690a45fad70c7c2d2b6329f94c072c6be182f9275c`;
all 442 build compiler commands invoke ccache. Before/after host-wide cache
receipts, source archive, candidate patches and binaries remain under
`.artifacts/range-weighted-force/`. No endpoint improvement is established yet.

## Rejected diagnostic direction

n2 Slurm 2149 reconstructed VV10 predicates on an independently converged
reference96 density: 1,923,992 density-active rows, 1,920,439 nonzero-weight
partners, and only 3,553 exact-zero weight rows. Removing those rows in a
strictly weighted-only consumer could save only 0.1847% of reconstructed pair
work. This is not a native executed counter and does not justify changing the
native density policy or discarding observable weight derivatives. The known
GPU4PySCF additional absolute-weight filter is not adopted as an optimization.

## Revisit when

Promote only after controlled complete endpoint improvement, larger-system
qualification and independent numerical gates. Reconsider the scalar shell
schedule if its register pressure or serialization regresses the endpoint;
the generic higher-order recurrence remains the explicit bounded fallback.
