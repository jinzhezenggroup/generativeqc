# Incremental Direct-J/K historical negative closeout

Keep #990/#1579 opt-in; inventory: negative-evidence. Only #1598 DoD item 1.
One retained 384-AO RHF pair has warm medians 6.835/13.862 s (+103%);
all three warm E/F samples pass GPU4PySCF and paired gates (1e-8 Eh,
1e-7 Eh/Bohr). Cold 71.076/100.190 s measures prepared-plan execution,
excludes preparation, and has no retained cold numerical arrays.
Seven final failed-retry outcomes do not prove failure causes or 4/4
SCF nonconvergence. Other reported populations are not certified here.

The exact reviewed capsule (13 gzip receipts, README, verifier, manifest)
is retained in existing Git history, pinned by capsule.json. No new archive,
Release, or external publication is needed. Shallow clones may first run:

    git fetch --no-tags https://github.com/jinzhezenggroup/generativeqc.git 0d712298bd59f1842911ae481e3790b9a417df21
    python benchmarks/results/issue1598-incremental-direct-jk-20261011/verify.py

The helper restores temporary Git-original bytes, checks tree/manifest/member
integrity, then runs every original semantic gate. No CUDA build is performed.
Missing Git objects fail closed; --fetch explicitly permits fetching the pin.
See the decision note for build provenance and omitted evidence. capsule.json
is historical metadata, not an accepted scientific publication.json envelope.
