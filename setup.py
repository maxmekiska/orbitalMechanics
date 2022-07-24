from setuptools import setup, find_packages
import os

with open('README.md', encoding="utf8") as file:
    long_description = file.read()

setup(
    author="Maximilian Mekiska",
    author_email="maxmekiska@gmail.com",
    url="https://github.com/maxmekiska/orbitalMechanics",
    description="Package that implements orbital mechanical concepts.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    name="Orbmec",
    version="0.1.0",
    packages = find_packages(include=["orbmec", "orbmec.*"]),
    install_requires=[
        "numpy>=1.21.6"
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords = ["space", "orbital", "mechanics"],
    python_rquieres=">=3.6"
)
