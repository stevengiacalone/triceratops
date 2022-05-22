#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("triceratops").version
except DistributionNotFound:
    __version__ = "unknown version"


# General stuff
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_nb",
]

myst_enable_extensions = ["dollarmath", "colon_fence"]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

# General information about the project.
project = "TRICERATOPS"
author = "Steven Giacalone"

version = __version__
release = __version__

exclude_patterns = ["_build"]

# HTML theme
html_theme = "sphinx_book_theme"
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_title = "triceratops"
html_logo = "_static/triceratops.svg"
html_static_path = ["_static"]
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/stevengiacalone/triceratops",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "classic",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
# jupyter_execute_notebooks = "off"
jupyter_execute_notebooks = "cache"
execution_timeout = -1