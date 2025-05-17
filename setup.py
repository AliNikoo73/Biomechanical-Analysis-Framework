#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="baf",
    version="0.1.0",
    author="BAF Team",
    author_email="info@baf-project.org",
    description="Biomechanical Analysis Framework - A comprehensive library for biomechanical simulation and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AliNikoo73/Biomechanical-Analysis-Framework",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.5b2",
            "mypy>=0.910",
        ],
        "gui": [
            "PyQt5>=5.15.0",
            "pyqtgraph>=0.12.0",
        ],
        "ml": [
            "torch>=1.9.0",
            "scikit-learn>=0.24.0",
        ],
    },
) 