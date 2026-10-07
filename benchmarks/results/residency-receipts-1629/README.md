# Historical DF trace adapter receipt

This is a source-matched consumer check against an existing production diagnostic,
not a current-device run or an acceptance claim for #1629. The retained source is
`benchmarks/results/issue206-practical-auxiliary/diagnosis/control-diagnosis-v1/oh-def2-svp-spherical-uhf-auto.jsonl`.
Its historical `vibeqc.df_trace` format is identified by the retention manifest in
that directory. The adapter contract selects this exact retained record and its
SHA-256 of LF-normalized bytes:
`4a57178922c890bf16830087c26a3971f4970ea6d267d1ab62508dd84e35c6c0`.
The Windows checkout may contain CRLF bytes; no record contents were altered.

The measured Python commit is `67b35f3a0a236de39ebbdb2876908a3c2da78c6c`;
native source identity is `06fa1f9d7108f86edfaa4ba9285df58c20a1d191b3c20785a2500cad1798730d`;
native library SHA-256 is `ea39fae62486240a98021726d970ec80c965e8f4a687056e59636a7179faf5be`.
The untimed diagnostic was Slurm 9905 on historical RTX 5090 resources. The OH
case label retains "def2-svp", while the actual orbital/auxiliary overrides were
cc-pVDZ/cc-pVDZ-JKFIT, 19/93 AOs. See the original issue206 README and manifest.

An independent sum over the eight raw JSONL records gives 300360 H2D bytes,
6048 D2H bytes, six explicitly counted production synchronizations, zero counted
event waits, and 1562 diagnostic profiler events. The trace adds at least one
final diagnostic event synchronization per operation. These are different
categories. The counters cover selected owners only; D2D, complete transfers,
event waits, payload dependencies, and all hot-region sites are unobserved. The
adapter result is therefore **INCOMPLETE**, even though the retained trace itself
is valid and source-matched to its historical manifest. No zero-transfer PASS or
device-residency claim follows.
