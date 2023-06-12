from setuptools import setup

from mdagent.version import __version__

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
    install_requires=[
        "pypdf",
        "langchain",
        "pqapi",
        "paper-qa",
        "google-search-results",
        "rmrkl",
        "git+https://github.com/blackadad/paper-scraper.git",
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
