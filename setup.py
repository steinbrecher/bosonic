from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np
import codecs
import os
import re

NAME = "bosonic"
PACKAGES = find_packages(where="src")
META_PATH = os.path.join("src", "bosonic", "__init__.py")
KEYWORDS = []
CLASSIFIERS = []
INSTALL_REQUIRES = ["numpy", "Cython"]

## Import metadata from src/bosonic/__init__.py
## This is taken from:
# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/
HERE = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()

META_FILE = read(META_PATH)

def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
        META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))

## Declare aa_phi as an external module so it gets compiled
ext_modules = [
    Extension(
        "bosonic.aa_phi",
        ["src/bosonic/aa_phi.pyx"],
        extra_compile_args=['-fopenmp', '-Ofast'],
        extra_link_args=['-fopenmp'],
    )
]

VERSION = find_meta("version")
URL = find_meta("url")
LONG = (
    read("README.rst")
    )


if __name__ == '__main__':
    setup(
        name=NAME,
        description=find_meta("description"),
        license=find_meta("license"),
        url=URL,
        version=VERSION,
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        keywords=KEYWORDS,
        long_description=LONG,
        ext_modules=cythonize(ext_modules),
        packages=PACKAGES,
        package_dir={"": "src"},
        zip_safe=False,
        install_requires=INSTALL_REQUIRES,
        )
