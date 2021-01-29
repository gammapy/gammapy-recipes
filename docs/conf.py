# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Gammapy recipes build configuration file.

import os
import datetime

# -- nbsphinx settings
extensions = ["nbsphinx", "sphinx_gallery.load_style", "IPython.sphinxext.ipython_console_highlighting"]
exclude_patterns = ["_build"]
nbsphinx_execute = "never"

# -- project settings
project = "Gammapy recipes"
author = "Gammapy recipes contributors"
copyright = "{}, {}".format(datetime.datetime.now().year, author)

# -- theme settings
# remove view source link
html_show_sourcelink = False

# static files to copy after template files
html_static_path = ["_static"]

# html_theme_options = {
#    'logotext1': 'gamma',  # white,  semi-bold
#    'logotext2': 'py',  # orange, light
#    'logotext3': ':docs'  # white,  light
# }

html_theme_options = {
    "canonical_url": "https://gammapy.org",
    "analytics_id": "",
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
}

# Add any paths that contain custom themes here, relative to this directory.
# To use a different custom theme, add the directory containing the theme.
# html_theme_path = []

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes. To override the custom theme, set this to the
# name of a builtin theme or the name of a custom theme in html_theme_path.
html_theme = "sphinx_rtd_theme"

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = ''
html_favicon = os.path.join(html_static_path[0], "gammapy_logo.ico")

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = ''

# The name for this set of Sphinx documents.
html_title = project


# html_style = ''
def setup(app):
    app.add_css_file("gammapy.css")
