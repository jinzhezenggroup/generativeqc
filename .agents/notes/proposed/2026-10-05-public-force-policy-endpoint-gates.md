# Evidence: public force-map policy and the remaining Direct force gap

Status: proposed policy; endpoint evidence retained without default promotion
Date: 2026-10-05

The [complete publication](../../../benchmarks/results/pbe0-public-force-policy-20261005/README.md)
compares the shared #1881 force-map profile against an empty public registry.
It replaces neither the public policy owner with #1833's older atom-only
low-level selector nor the exact-direct scientific method with DF/COSX.

The current composition has generic SCF local-AO admission from #1847, formal
precision guards from #1872, phased Becke from #1830, indexed force scheduling,
and actual public Fock counts from #1920. Every native/reference and
candidate/control numerical pairing passes. Current warm E+F observations are
13.1681 to 10.7415 s at 48 atoms and 46.8146 to 32.4317 s at 96 atoms. The
observed force AO-square work reductions are retained, not inferred from speed.

There are two reasons not to transfer an isolated warm result directly into a
universal default claim. First, an older composition's 96-atom cold result
regressed 8.393% while SCF builds changed 25 to 29. The current composition's
cold result is faster, but builds still differ 31 to 30. These process-ordered
observations do not identify a causal force-map effect on SCF or permit
iteration-normalized timing. Second, the finite structural workload profile's
interpolation beyond the measured examples remains an explicit policy
assumption. Subsequent cold repetitions/default decisions need their own
source, cache, numerical and endpoint qualification.

Separate intrusive profiles distinguish actual Becke kernels from other grid
geometry response. At 96 atoms, the bounded Direct derivative consumes 13.4196 s
(43.96% of summed GPU kernel duration), Becke phases 4.0954 s and AO/grid
geometry 2.7062 s. The generated J/K value kernels account for 2.4724/1.9853 s;
the corresponding 48-atom values are 1.1772/1.0631 s. J/K are attributed by the
pinned source's submission order, with ordered launch CSVs and a single public
Fock build. They exclude preparation, other kernels and synchronization.

The derivative profile motivates work reduction in its existing compiler
owner; it does not establish which angular classes dominate, measured spills,
or a speedup for the separate primitive-root reuse experiment. Repeating
rejected angular passes, compact pages or Becke gather fusion requires a new
mechanism and full endpoint evidence. #1895's 1.25x exact-direct warm target
remains unmet, and these records do not close that tracker.
