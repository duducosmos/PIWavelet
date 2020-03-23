import os,glob
from setuptools import  setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

datadir = os.path.join('mfiles','wtc')
datafiles = [(datadir, [f for f in glob.glob(os.path.join(datadir, '*'))])]


setup(
    name = "piwavelet",
    version = "1.0.1",
    author = "Eduardo dos Santos Pereira",
    author_email = "pereira.somoza@gmail.com",
    description = ("Tools For Wavelet Analises in Python"),
    license="Apache License 2.0",
    keywords = "wavelet signal analises ",
    url = "https://github.com/duducosmos/piwavelet",
    packages= find_packages(),
    package_data={'': ['*.m','*.txt', '*.html', '*.png']},
    data_files= datafiles,
    install_requires=['oct2py>=2.0.0', 'numpy>=1.6.2', 'scipy>=0.10.1', 'matplotlib>=1.1.1'],
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
)
