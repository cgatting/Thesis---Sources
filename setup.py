"""
Setup script for RefScore academic application.

This script handles package installation and distribution.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="refscore-academic",
    version="1.0.0",
    author="Academic Project Team",
    author_email="academic@university.edu",
    description="A comprehensive tool for analyzing academic references against documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/refscore-academic",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-qt>=4.1.0",
            "pytest-cov>=4.0.0",
            "flake8>=5.0.0",
            "black>=22.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "refscore=refscore.cli:main",
            "refscore-gui=refscore.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "refscore": [
            "resources/icons/*.png",
            "resources/styles/*.qss",
        ],
    },
    zip_safe=False,
)