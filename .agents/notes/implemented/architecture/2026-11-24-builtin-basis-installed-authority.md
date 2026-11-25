# Decision: require the native catalog for installed bundled basis execution

Status: implemented
Date: 2026-11-24

An exact bundled basis record represents the native catalog, not a request to
try several Python and C++ expansion routes. The Python frontend now requires
the same name and construction protocols as the installed native path; absence
of either is an ABI error, not a selector or resource refusal to work around.
The compiled pack digest must still match the immutable local metadata before
execution. A legacy constructor remains available to native callers, but
Python no longer dispatches to it when a selected library lacks the newer
catalog capabilities.

Offline lookup and explicit caller data do not request that catalog contract.
Keep library-free metadata, explicit basis templates/shells, and the separate
correlation auxiliary constructor. Test the same name with a different
fingerprint, native checksum rejection, missing symbols, and equal complete
ragged outcomes for the same floating-point basis data in cold, warm, and
moved-coordinate executions. This is an R5/R6 ownership change under #2227,
not new basis, provider, method, or numerical qualification.
