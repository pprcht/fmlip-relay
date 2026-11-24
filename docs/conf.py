# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "fmlip-relay"
copyright = "2026, Philipp Pracht"
author = "Philipp Pracht"

# -- General configuration ---------------------------------------------------

# autodoc and sphinx-argparse import the installed package, so the docs are
# built against `pip install ".[docs]"`. Only modules that import without the
# optional ML dependencies are documented.

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinxarg.ext",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
# anchors for "#section" links between the Markdown pages
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

napoleon_numpy_docstring = True

exclude_patterns = ["_build"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

html_theme = "shibuya"
html_title = "fmlip-relay"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/pprcht/fmlip-relay",
}
