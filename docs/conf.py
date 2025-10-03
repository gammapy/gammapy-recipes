# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Gammapy recipes build configuration file.

import os
import datetime

# -- General configuration ----------------------------------------------------
author = "Gammapy recipes contributors"
copyright = "{}, {}".format(datetime.datetime.now().year, author)


highlight_language = "python3"
html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False
html_static_path = ["_static"]
html_logo = os.path.join(html_static_path[0], "gammapy_logo.png")
html_favicon = os.path.join(html_static_path[0], "gammapy_logo.ico")

html_theme_options = {
    "header_links_before_dropdown": 6,
    "show_toc_level": 2,
    "show_prev_next": False,
    "icon_links": [
        {
            "name": "Github",
            "url": "https://github.com/gammapy/gammapy-recipes",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Slack",
            "url": "https://gammapy.slack.com/",
            "icon": "fab fa-slack",
        },
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navigation_with_keys": True,
    # footers
    "footer_start": ["copyright"],#,"custom-footer.html"],
    "footer_center": ["last-updated"],
    "footer_end": ["sphinx-version", "theme-version"]
}


html_context = {
    "default_mode": "light",
}


sphinx_gallery_conf = {
    "examples_dirs": ["../recipes"],    # path to your recipe scripts
    "gallery_dirs": ["recipes_gallery"],       # where SG will put rst/html
    "plot_gallery": False,            # never re-execute
    #"run_stale_examples": "skip",   False    # don’t try to rebuild
    "filename_pattern": r"^.*\.py$",        # only .py files
    "ignore_pattern": r"__init__\.py",  # skip __init__ etc
    # Appearance
    "remove_config_comments": True,
    "download_all_examples": True,
    "nested_sections": True,
    "show_memory": False,
    "line_numbers": False,
}


extensions = [
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
]