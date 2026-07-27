# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "DBOF in Native Grid"
copyright = "2024, J. Xavier Prochaska, P. Cornillon, J. Tallman"
author = "J. Xavier Prochaska, P. Cornillon, J. Tallman"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.mermaid",
]

# MyST parser configuration for markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Render fenced ```mermaid code blocks via the sphinxcontrib-mermaid
# directive (used in docs/mermaid_diagrams.md).
myst_fence_as_directive = ["mermaid"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Create _static directory if it doesn't exist (prevents warning)
import os
if not os.path.exists(os.path.join(os.path.dirname(__file__), "_static")):
    os.makedirs(os.path.join(os.path.dirname(__file__), "_static"))
