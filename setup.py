"""Python setup.py for bno package"""
import io
import os

from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("bno", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="bno",
    version=read("bno", "VERSION"),
    description="Born Neural Operator",
    url="https://github.com/ucl-bug/bno",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Antonio Stanziola, UCL BUG",
    author_email="a.stanziola@ucl.ac.uk",
    packages=find_packages(exclude=["tests", ".github", "docs"]),
    python_requires=">=3.7",
    install_requires=read_requirements("requirements.txt"),
    license="GNU Lesser General Public License (LGPL)",
)
