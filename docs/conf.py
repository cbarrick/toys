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
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]


# Theme
# --------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme specific,
# see https://sphinx-rtd-theme.readthedocs.io/en/latest/configuring.html
html_theme_options = {
    'logo_only': False,
    'display_version': True,

    # Sidebar
    'collapse_navigation': True,
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
}


# sphinx.ext.todo
# --------------------------------------------------

# Toggle output for  ..todo::  and  ..todolist::
todo_include_todos = True


# Custom setup
# --------------------------------------------------
# See http://www.sphinx-doc.org/en/master/extdev/appapi.html

def setup(app):
    app.add_stylesheet('custom.css')
