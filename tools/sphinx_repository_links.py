"""Resolve existing out-of-doc source links without weakening Sphinx warnings."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

if TYPE_CHECKING:
    from collections.abc import Iterable

REPOSITORY_URL = "https://github.com/jinzhezenggroup/generativeqc"


def repository_revision(root: Path) -> str:
    """Pin source links to the checkout; archives without Git use the main branch."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "master"
    return result.stdout.strip()


def repository_source_link(
    target: str,
    *,
    root: Path,
    source_dir: Path,
    suffixes: Iterable[str],
    revision: str,
    fragment: str | None = None,
) -> tuple[Path, str] | None:
    """Only map real repository documents outside Sphinx's source directory.

    MyST turns existing Markdown/RST links into absolute, suffix-free doc
    references. Internal docs and unresolved links must retain normal Sphinx
    validation rather than becoming unchecked external URLs.
    """
    if not Path(target).is_absolute():
        return None
    root = root.resolve()
    source_dir = source_dir.resolve()
    for suffix in suffixes:
        candidate = Path(target + suffix).resolve()
        if (
            not candidate.is_relative_to(root)
            or candidate.is_relative_to(source_dir)
            or not candidate.is_file()
        ):
            continue
        path = quote(candidate.relative_to(root).as_posix(), safe="/")
        url = f"{REPOSITORY_URL}/blob/{quote(revision, safe='')}/{path}"
        if fragment:
            url += "#" + quote(fragment, safe="-._~")
        return candidate, url
    return None


def resolve_repository_links(app: Any, doctree: Any) -> None:
    from docutils import nodes
    from sphinx.addnodes import pending_xref

    root = Path(app.confdir).resolve().parent
    for node in doctree.findall(pending_xref):
        if node.get("reftype") != "myst" or node.get("refdomain") != "doc":
            continue
        resolved = repository_source_link(
            node["reftarget"],
            root=root,
            source_dir=Path(app.srcdir),
            suffixes=app.config.source_suffix,
            revision=app.config.repository_source_revision,
            fragment=node.get("reftargetid"),
        )
        if resolved is None:
            continue
        path, url = resolved
        app.env.note_dependency(str(path))
        reference = nodes.reference("", "", refuri=url)
        reference.extend(child.deepcopy() for child in node.children)
        node.replace_self(reference)


def setup_repository_links(app: Any) -> None:
    # Changing commits invalidates saved doctrees, including their source URLs.
    app.add_config_value(
        "repository_source_revision",
        repository_revision(Path(app.confdir).resolve().parent),
        "env",
    )
    app.connect("doctree-read", resolve_repository_links)
