# Decision: Query native named KS registry carriers directly

Status: implemented
Date: 2026-11-24

The backend-neutral Python registry query previously reconstructed a compiler
graph even when the installed library already owned the exact named program.
For correction-free represented programs, the generated catalog's carrier is
the existing public registry projection. Query it without preparing a context
or invoking the compiler resolver. Keep selection and capability lookup on the
same loaded library; a native error or malformed carrier cannot trigger an
alternate lookup.

Do not reuse a correction-bearing program's preparation carrier for registry
presentation: some public corrections use a different registry row, and being
present in the catalog does not imply that a composite spelling is public.
Zero selection leaves those graphs, catalogs without a represented row, and
other compiler-owned selectors on their explicit discovery routes. Automatic
Libxc and contextual `Calculator` admission are unchanged. No registry row
proves endpoint support.

The parity test enumerates the generated named catalog and compares its
represented correction-free selections to the previous compiler projection;
failure injection proves a native hit or error cannot fall through to that
projection. Refs #2227.
