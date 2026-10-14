"""Admit only ordinary README edits to the proposed prose-only CI tier.

This intentionally excludes docs/, evidence, assets and all unknown paths.
Master cache producers, nightly/manual qualification and uncertain ranges retain
full coverage. The CLI queue selector remains the authority for the complete
unmerged queue range; its existing release/fast policy is not changed here.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

from native_cli_ci_scope import _git, _merge_group_base

_SHA = re.compile(r"[0-9a-f]{40}\Z")
_README_LIMIT = 1024 * 1024


def _pr_base(base: str, head: str, pr_head: str) -> str:
    """Use the actual tested base of a verified two-parent PR test merge."""
    if not _SHA.fullmatch(pr_head):
        raise ValueError("missing PR source head")
    parents = _git("rev-list", "--parents", "-n", "1", head).decode().split()
    if len(parents) != 3 or parents[0] != head or parents[2] != pr_head:
        raise ValueError("checkout is not the requested two-parent PR test merge")
    tested_base = parents[1]
    if not _SHA.fullmatch(tested_base):
        raise ValueError("invalid tested base")
    # A newer tested target may include unrelated master work. It must still
    # descend from the event's declared base; unrelated or rewound refs fail full.
    _git("merge-base", "--is-ancestor", base, tested_base)
    return tested_base


def _readme_change(base: str, head: str) -> bool:
    """Accept one modified regular UTF-8 README, including both rename sides."""
    diff = _git(
        "diff",
        "--raw",
        "--no-abbrev",
        "--no-renames",
        "--no-ext-diff",
        "--no-textconv",
        "-z",
        base,
        head,
        "--",
    )
    parts = diff.split(b"\0")
    if len(parts) != 3 or parts[1:] != [b"README.md", b""]:
        return False
    fields = parts[0].decode("ascii").split()
    if len(fields) != 5 or fields[:2] != [":100644", "100644"] or fields[4] != "M":
        return False
    for oid in fields[2:4]:
        if not _SHA.fullmatch(oid):
            return False
        size = int(_git("cat-file", "-s", oid))
        if not 0 < size <= _README_LIMIT:
            return False
        text = _git("cat-file", "blob", oid).decode("utf-8")
        if not text.strip() or any(ord(c) < 32 and c not in "\t\r\n" for c in text):
            return False
    return True


def select_scope(
    event: str, base: str, head: str, target_ref: str = "", pr_head: str = ""
) -> str:
    """Return prose only on positive proof; all uncertainty returns full."""
    if event not in {"pull_request", "merge_group"}:
        return "full"
    if not _SHA.fullmatch(base) or not _SHA.fullmatch(head):
        return "full"
    try:
        if _git("rev-parse", "--verify", "HEAD").decode().strip() != head:
            raise ValueError("head does not match checkout")
        if event == "merge_group":
            # A non-README immediate tail can only require full CI. Resolve the
            # remote target only for potential prose tails, never use this cheap
            # rejection to approve a skip across the earlier queued changes.
            if not _readme_change(base, head):
                return "full"
            base = _merge_group_base(base, head, target_ref)
        else:
            base = _pr_base(base, head, pr_head)
        return "prose" if _readme_change(base, head) else "full"
    except (OSError, subprocess.SubprocessError, ValueError, UnicodeError):
        print("Prose-only comparison unavailable; requiring full CI", file=sys.stderr)
        return "full"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--target-ref", default="")
    parser.add_argument("--pr-head-sha", default="")
    args = parser.parse_args()
    print(
        select_scope(
            args.event, args.base_sha, args.head_sha, args.target_ref, args.pr_head_sha
        )
    )


if __name__ == "__main__":
    main()
