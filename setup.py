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
    install_requires=[
        # Generic:
        "typing-extensions~=3.7",
        "loguru~=0.4",
        "click~=7.0",
        "dataclass-csv~=1.1",
        # Image processing / models:
        "tensorboard~=2.4",
        "torch~=1.5",
        "torchvision~=0.7",
        "opencv-python~=4.0",
        "pygame >=1.9, <3.0a0",
        "pillow~=8.1",
        "matplotlib~=3.0",
        # Computation:
        "numpy~=1.19",
        "numpy-stubs",
        "networkx~=2.5",
    ],
    scripts=["scripts/intervention-learning"],
)
