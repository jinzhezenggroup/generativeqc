# Follow-up: observe the original RHF guard through its ELF call boundary

Status: proposed
Date: 2026-10-18

The [earlier observer](2026-10-15-rhf-provider-retention-observer.md) retains its
ambiguous a308 evidence. This separate experiment removes the need for a core
instrumentation edit when a particular binary exposes the real guard through a
preemptible symbol. It does not transfer SCF ownership or authorize a repair.

The already built 129d library exports weak `reference_detail::retained_bytes`
and `free_bytes` symbols with real `R_X86_64_JUMP_SLOT` relocations. An isolated
external DSO forwards the original guard's stream, before-free and allowance to
`RTLD_NEXT`, records only the guard's existing memory-query output, and rethrows
the original exception. It adds host/loader work but no CUDA query, allocation,
synchronization, warmup or retry. Inlined or locally bound calls may be invisible;
symbol presence alone cannot prove actual interception. This is experimental
preparation, not a supported repository interface or clean timing method.

Source and qualification are retained in the project shared directory
`/inspire/qb-ilm/project/chemicalreaction/czxs25220150/issue1765-native-guard-20261018-v3`.
Observer source SHA-256:
`cf13dd40247ce353cacbdef2d05ccd178fbfd98135c65b619f158815098ddad0`.
Observer DSO SHA-256:
`7119336f80985eb67b563c47fa6734bbb5a4f43bbdf8cc8679dd3efc30c95302`.
Real-library/fake-CUDA controls passed for normal return, allowance refusal,
query/synchronization failure, OOM, four concurrent threads and nested calls.
Baseline/observer stdout, original query/synchronization counts and argument
checks matched. Existing-file, absent-parent and `/dev/full` logging controls
preserved native behavior. Successful captures used fresh `0600` files and
retained zero loss/error counters.

One plain/observed failure-path pair ran in Job `i1765-guard-129d-1018` on H200,
driver 595.58.03, reusing image
`generativeqc-science-py311-cuda129:20261015` and the unchanged ethane230 input
with energy-only Q8/DIIS8 controls. Native source remains
`129d1874a12876e9d37cefd4108cfcf31003d6be`; library SHA-256 is
`3e9d0a91f7820b48b7c3112f3bd14626a8df34b8b36ba2355af1aceeacb77662`,
endpoint SHA-256 is
`021874bba5f77ca0b61645c81fc5b2b264a22785f67f56e064ad3d8c89a4f660`.
The observed process mapped those exact library/observer hashes. Library build
ID `9004acd49e7dd19b5593b7fdbed4806ccb6475fa`, real guard offset `0x11a3020`
and caller return offset `0x118e493` bind the observation to the actual PLT call
in `execute_hf_cuda_bucket`. One begin/throw pair plus terminal was recovered:

| Direct observation | Bytes |
| --- | ---: |
| Original before-free argument | 131,314,352,128 |
| Guard query after-free | 130,943,156,224 |
| Guard query after-total | 150,111,977,472 |
| Retained delta | 371,195,904 (354 MiB) |
| Original allowance argument | 201,326,592 (192 MiB) |
| Excess over allowance | 169,869,312 (162 MiB) |

Before-query total and actual provider flags remain null. Neither is inferred
from the later total or allowance. cuBLAS/cuBLASLt 12.9.2.10, cuSOLVER 11.7.5.82
and CUDA runtime 12.9.79 were mapped and hashed separately in both processes.
Both native invocations exited 1 with identical 111-byte logs, SHA-256
`72944503040cc4bc621bb7444672470a5969f376953473369bab50b236ba5468`.
The 489-byte journal SHA-256 is
`1ab303ce607d805ed8612365fea3c9f9ec8ec7ed21a64d7e381d4143dca42c2a`;
the complete raw receipt SHA-256 is
`b7cb48a671df33c6020c7eec112af13b1a395a1d50c5c6600f820a7815a16e78`.
Independent raw retrieval, rehash and caller disassembly are retained locally in
`.artifacts/pr2581-review-20261018/`; the raw device outputs remain under
`/inspire/qb-ilm/project/chemicalreaction/czxs25220150/issue1765-guard-device-20261018/outputs`.

Job success means the diagnostic observed the original refusal. CC work,
complete endpoint/scientific acceptance, performance, approval and merge remain
unqualified. The measured excess does not authorize increasing the allowance.
Further core repair still needs the SCF owner's coordinated handoff, requested
in PR #2287 comment 6185131043. Environment/image reuse worked; repeating the
unchanged campaign or rebuilding the environment would not fix this refusal.
