# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = '2R Assignment 2'
copyright = '2025, Constantina Maltezou'
author = 'Constantina Maltezou'
release = '2025'

# -- Path setup --------------------------------------------------------------
import os
import sys
from pathlib import Path

ROOT = Path.cwd().parent.parent      # .../Assignment_2
sys.path.insert(0, str(ROOT))                   # so 'custom_packages' can be imported
sys.path.insert(0, str(ROOT / "src"))           # if you import from src/main.py or similar


# -- Autodoc setup -----------------------------------------------------------
# Generate stubs for modules that aren't available during docs build.
autodoc_mock_imports = [
    "openai",
    "google",
    "googleapiclient",
    "google.genai",
    "googleapiclient.discovery",
    "numpy",
    "pandas",
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
    "wordcloud",
    "scipy",
    "subprocess",
    "tempfile",
    "sklearn",
    "sklearn.metrics",
    "sklearn.decomposition",
    "sklearn.feature_extraction.text",
    "nltk",
    "contractions",
    "pydub",
    "transformers",
    "fastopic",
    "topmost",
    "librosa",
    "numba",
    "soundfile",
    "torch",
    "cv2",
]


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#extensions = [
#    'numpydoc',
#    'sphinx.ext.autodoc',  # For auto-generation of docstrings
#    'sphinx.ext.napoleon',  # For Google-style docstrings
#    'rst2pdf.pdfbuilder',  # For PDF output
#]

extensions = [
    "sphinx.ext.autodoc",    # For auto-generation of docstrings
    "sphinx.ext.napoleon",   # For numpy or google style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]
autosummary_generate = True     # auto-generate stub pages

# --- Autodoc & Python signature formatting ---
# Put type hints in the description block instead of the function signature
autodoc_typehints = "description"
add_module_names = False                    # drop "custom_packages.misc_modules..." prefixes

# Keep type names short (no long module prefixes like pandas.core.frame.DataFrame)
autodoc_typehints_format = "short"   # Sphinx ≥ 7
add_module_names = False             # Drop full module prefixes in object names
python_use_unqualified_type_names = True  # Sphinx ≥ 5


napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True      # render Args/Parameters as :param: fields (nice layout)
napoleon_use_rtype = True


# --- LaTeX/PDF tuning ---
latex_engine = "xelatex"  # optional, but better fonts + hyphenation (needs TeX packages installed)

latex_documents = [
    ('index', '2rassignment2.tex',
     '2R Assignment 2 Documentation',
     'Constantina Maltezou',
     'howto'),   # "howto" offers an article style, instead "manual" offers a book style/format
]

latex_elements = {
    "preamble": r"""
\usepackage[htt]{hyphenat}  % allow wrapping in typewriter font (identifiers)
\usepackage{enumitem}       % tighter parameter lists
\setlist{nosep}
\emergencystretch=3em       % reduce overfull boxes without ugly breaks
""",
"tableofcontents": "",
#    "tableofcontents": r"""
#\renewcommand{\contentsname}{Modules}
#\setcounter{tocdepth}{2}
#\tableofcontents
#\clearpage
#""",
}



templates_path = ['_templates']
exclude_patterns = []


#pdf_documents = [
#    ('index', '2r_assignment_2', '2R, second assignment documentation', 'Constantina Maltezou'),
#]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme' # bizstyle
html_static_path = ['_static']
