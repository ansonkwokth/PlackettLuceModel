from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os
import sys

setup(
    name="plackett_luce",
    version="0.1.0",
    description="Ranking model",
    author="Anson",
    author_email="ansonwos@gmail.com",
    packages=find_packages(where="plackett_luce"),
    package_dir={"": "plackett_luce"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.26",
        "torch>=2.5",
    ],
)
