# Configuration file for Sphinx documentation builder
import os
import sys

project = "Cell Painting Feature Extraction Pipeline"
copyright = "2026, Way Lab"
author = "Michael J. Lippincott"
release = "1.0.0"

# Add project root and src to path
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../utils/src"))
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST Parser configuration
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}

# MyST Parser settings
myst_enable_checkboxes = True
myst_enable_html_img = True
myst_heading_anchors = 3
myst_parse_data = True

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
}

html_css_files = ["custom.css"]

# MathJax config for proper rendering
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Allow autodoc to skip missing imports
# docs/source/conf.py - Add mpl_toolkits to mock imports
autodoc_mock_imports = [
    "numpy",
    "scipy",
    "scipy.ndimage",
    "scipy.optimize",
    "scipy.stats",
    "scipy.spatial",
    "scipy.spatial.distance",
    "skimage",
    "skimage.segmentation",
    "skimage.morphology",
    "skimage.filters",
    "skimage.measure",
    "cellprofiler",
    "cellprofiler_core",
    "cellpose",
    "cellpose.models",
    "tifffile",
    "mahotas",
    "medim",
    "moviepy",
    "moviepy.editor",
    "napari_animation",
    "napari_animation.easing",
    "networkx",
    "pandas",
    "matplotlib",
    "matplotlib.pyplot",
    "mpl_toolkits",
    "mpl_toolkits.mplot3d",
    "psutil",
    "tomli",
    "tqdm",
    "cucim",
    "cucim.skimage",
    "cucim.skimage.measure",
    "cucim.skimage.morphology",
    "cupy",
    "cupyx",
    "cupyx.scipy",
    "cupyx.scipy.ndimage",
    "tensorflow",
    "torch",
    "torch.nn",
    "nviz",
    "nviz.image",
    "nviz.image_meta",
    "nviz.view",
]

# Don't treat import warnings as errors
suppress_warnings = [
    "autodoc.import_object",
    "config.deprecated_alias",
    "myst.topmatter",
]

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
