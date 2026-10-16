"""Sphinx configuration for the pybaycor documentation."""

from __future__ import annotations

import importlib.metadata

project = "pybaycor"
author = "Peter Scicluna"
copyright = "Peter Scicluna"

try:
    release = importlib.metadata.version("pybaycor")
except importlib.metadata.PackageNotFoundError:  # docs built without the package installed
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST --------------------------------------------------------------------
myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 3

# -- AutoAPI -----------------------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../pybaycor"]
autoapi_root = "autoapi"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "class"
autoapi_keep_files = False

# -- Napoleon (NumPy-style docstrings) --------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True  # avoid duplicate attribute descriptions with AutoAPI

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable/", None),
    "arviz": ("https://python.arviz.org/en/stable/", None),
}

# -- HTML --------------------------------------------------------------------
html_theme = "furo"
html_title = f"{project} {release}"
html_static_path: list[str] = []
html_theme_options = {
    "source_repository": "https://github.com/pscicluna/pybaycor",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- sphinx-copybutton -------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
