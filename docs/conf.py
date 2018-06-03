# Configuration file for the Sphinx documentation builder.
#
# For a full list of configuration options, see the documentation:
# http://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys


# Path setup
# --------------------------------------------------

# All extensions and modules to document with autodoc must be in sys.path.
# This function adds paths relative to the documentation root.
add_path = lambda p: sys.path.insert(0, os.path.abspath(p))

add_path('..')


# Project information
# --------------------------------------------------

project = 'Toys'
version = ''  # short X.Y version
release = ''  # full version, including alpha/beta/rc tags

copyright = '2018, Chris Barrick'
author = 'Chris Barrick'


# Configuration
# --------------------------------------------------

needs_sphinx = '1.7'  # v1.7.0 was released 2018-02-12
master_doc = 'index'
language = 'en'
pygments_style = 'sphinx'

templates_path = ['_templates']
source_suffix = ['.rst']
exclude_patterns = ['_build', '_static', 'Thumbs.db', '.DS_Store']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.githubpages',
    'sphinx.ext.ifconfig',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]


# Theme
# --------------------------------------------------

html_theme = 'sphinx_rtd_theme'
# html_logo = '_static/logo.svg'
html_static_path = ['_static']

# Theme specific,
# see https://sphinx-rtd-theme.readthedocs.io/en/latest/configuring.html
html_theme_options = {
    'logo_only': True,
    'display_version': True,

    # Sidebar
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}


# sphinx.ext.intersphinx
# --------------------------------------------------

# A mapping:  id -> (target, invintory)
# where  target  is the base URL of the target documentation,
# and  invintory  is the name of the inventory file, or  None  for the default.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
}


# sphinx.ext.autodoc
# --------------------------------------------------

autodoc_default_flags = ['members']


# sphinx.ext.autosummary
# --------------------------------------------------

autosummary_generate = True


# sphinx.ext.napoleon
# --------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True


# sphinx.ext.todo
# --------------------------------------------------

# Toggle output for  ..todo::  and  ..todolist::
todo_include_todos = True


# Custom setup
# --------------------------------------------------
# See http://www.sphinx-doc.org/en/master/extdev/appapi.html

def setup(app):
    app.add_stylesheet('custom.css')
