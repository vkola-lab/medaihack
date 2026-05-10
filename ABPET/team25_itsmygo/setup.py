"""Installable package metadata for `mygo_centiloid`.

Install in editable mode during development:

    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name             = "mygo_centiloid",
    version          = "0.1.0",
    description      = "Amyloid β-PET Centiloid prediction — MedAI Spring 2026 Hackathon.",
    url              = "https://github.com/vkola-lab/medaihack",
    packages         = find_packages(include=["mygo_centiloid", "mygo_centiloid.*"]),
    python_requires  = ">=3.10",
    install_requires = [
        "numpy>=1.26",
        "pandas>=2.0",
        "torch>=2.4",
    ],
    extras_require = {
        "eda": ["matplotlib>=3.7", "seaborn>=0.13", "scipy>=1.10", "tqdm>=4.65"],
    },
)
