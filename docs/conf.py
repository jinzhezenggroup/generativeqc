import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from __future__ import annotations

from tools.render_public_methods_doc import render_public_methods_markdown

project = "VibeQC"
author = "VibeQC contributors"
language = "en"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx_book_theme",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "AGENTS.md",
    "superpowers/**",
]

html_theme = "sphinx_book_theme"
html_title = "VibeQC documentation"

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]
myst_heading_anchors = 3


def _render_public_methods(app, docname, source):
    if docname == "public_methods":
        source[0] = render_public_methods_markdown()


def setup(app):
    app.connect("source-read", _render_public_methods)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
