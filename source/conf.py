from docutils import nodes
from docutils.parsers.rst import roles
import os, sys
from unittest.mock import MagicMock

import seaborn as sns

# Mock sns.__version__ if it's not a string
if not isinstance(sns.__version__, str):
    sns.__version__ = "0.12.2"  # or whatever version you’re using

sys.path.insert(0, os.path.abspath("../src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "EquiBoots"
copyright = "UCLA CTSI ML Team: Leonid Shpaner, Arthur Funnell, Al Rahrooh, Colin Beam, and Panayiotis Petousis"
author = (
    "Leonid Shpaner, Arthur Funnell, Al Rahrooh, Colin Beam, and Panayiotis Petousis"
)
release = "0.0.0a9"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_copybutton",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_multiversion",
    # "sphinxcontrib.bibtex",
]

# Mock out heavy/optional dependencies so autodoc won’t fail
autodoc_mock_imports = [
    "statsmodels",
    "statsmodels.stats.multitest",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
    "packaging",  # packaging.version.parse errors
    # …and any others your code pulls in…
]

templates_path = ["_templates"]
exclude_patterns = []


smv_tag_whitelist = (
    r"^v\d+\.\d+.*$"  # Whitelist tags with versions like v1.0, v2.0, etc.
)
smv_branch_whitelist = r"^main$"  # Whitelist the main branch

# Add this line to specify the bibliography file
# bibtex_bibfiles = ["references.bib"]

extensions.append("sphinx.ext.intersphinx")
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

copybutton_prompt_text = r">|\$ "
copybutton_prompt_is_regexp = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]

# If your documentation is served from a subdirectory, set this to the subdirectory path
html_show_sourcelink = False

# html_static_path = ["_static"]

html_context = {
    "display_github": False,
    "github_user": "your_username",
    "github_repo": "your_repository",
    "github_version": "main/docs/",
    "current_version": "v1.0",
    "versions": [
        ("v1.0", "/en/v1.0/"),
        ("v0.9", "/en/v0.9/"),
        ("latest", "/en/latest/"),
    ],
}

html_css_files = [
    "custom.css",
    "custom.js",
]


def setup(app):
    app.add_css_file("custom.css")
    app.add_js_file("custom.js")


def bold_literal_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    # Creating a strong (bold) node that contains a literal node
    node = nodes.strong()
    literal_node = nodes.literal(text, text, classes=["bold-literal"])
    node += literal_node
    return [node], []


roles.register_canonical_role("bold-literal", bold_literal_role)


################################################################################
