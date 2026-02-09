# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'IACTrace'
copyright = '2025, Gerrit Roellinghoff'
author = 'Gerrit Roellinghoff'

# Version handling
try:
    from iactrace import __version__
    release = __version__
except ImportError:
    release = "dev"

version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_automodapi.automodapi',
    'sphinx_design',
    'myst_nb',
]

# -- Options for myst-nb -----------------------------------------------------

# Don't execute notebooks during build (they should be pre-executed)
nb_execution_mode = 'off'

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.md': 'myst-nb',
}

# myst-parser configuration
myst_enable_extensions = [
    'dollarmath',  # Enable $ and $$ for math
    'colon_fence',  # Enable ::: fences
]

# Add a note to each notebook page
nb_render_markdown_format = 'myst'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for autodoc -----------------------------------------------------

# Clean up JAX/dataclass signatures
autodoc_typehints = 'none'
autodoc_class_signature = 'separated'
autodoc_member_order = 'bysource'

# Don't show inherited members by default
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# -- Options for automodapi --------------------------------------------------

# Don't generate inheritance diagrams (cleaner output)
numpydoc_show_class_members = False

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

# Furo theme options
html_theme_options = {
    "source_repository": "https://github.com/GerritRo/iactrace",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- Options for Napoleon extension ------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_attr_annotations = True
