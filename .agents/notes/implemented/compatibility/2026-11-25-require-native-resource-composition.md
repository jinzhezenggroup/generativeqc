# Require resource composition after runtime selection

The portable resource planner remains useful for offline metadata and as an
independent comparison. It cannot make decisions for a selected runtime. In
particular, an explicit feasible `ResourcePlan` is a data-only descriptor; it
does not prove the selected runtime admitted the fixed choice or its later
force-workspace upgrade. A missing composition symbol previously allowed both
paths to compare Python-computed peaks and proceed without native admission.

Leave library-free planning independent of the loader. Once a runtime has
been selected, require its existing native composition query before choosing
alternatives, validating an explicit plan, or changing a prepared plan. This
does not promote a composition result to a provider allocation or lease.
