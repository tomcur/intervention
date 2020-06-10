#!/usr/bin/env python

from intervention import __version__

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name="intervention",
    version=__version__,
    description="A simulator implementation of learning from interventions during autonomous driving",
    author="Thomas Churchman",
    author_email="thomas@kepow.org",
    packages=["intervention",],
    install_requires=[],
    scripts=["scripts/intervention-learning"],
)
