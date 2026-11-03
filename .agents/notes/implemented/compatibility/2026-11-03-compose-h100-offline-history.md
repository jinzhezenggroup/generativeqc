# Decision: compose H100 history recovery through the public offline reader

The merge-queue predecessor and final H100 delivery independently migrated the
same two Nsight parts to authenticated existing Git objects. Preserve the final
delivery's explicit `historical-snapshot.json` and public `read_evidence_bytes`
owner rather than reintroducing dynamic imports of private recovery helpers.
The older manifest remains historical metadata, not another active loader.

Public receivers also stage copied packets outside the repository. If one of the
two declared archive parts is absent there, resolve its canonical repository path
through the supplied authenticated snapshot; a temporary staging path cannot be
used as the Git object's repository-relative identity. Do not recover arbitrary
missing artifacts, replace a present corrupt file, or accept a linked/nonregular
staging object. Preserve original archive/member checksums and every failure.

This is queue/history composition, not new GPU qualification. The frozen final
denominator, complete endpoint arrays, SQLite reductions and source limits remain
unchanged, and the CPU receiver regression covers staging and damaged packets.
