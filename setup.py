import os
from glob import glob
from setuptools import setup, find_packages

exec(open("md-agent/version.py").read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="md-agent",
    version=__version__,
    description="Collection of MD tools for use with language models",
    author="Andrew White",
    author_email="andrew.white@rochester.edu",
    url="https://github.com/ur-whitelab/md-agent",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "pypdf",
       "langchain==0.0.173",
        "paper-qa==1.1.1",
        "google-search-results",
        "rmrkl @ git+https://github.com/doncamilom/robust-mrkl.git",
    ],
    test_suite="tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)