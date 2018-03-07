#!/usr/bin/env python
from setuptools import setup, find_packages

requirements = [
    'torch'
]

setup(
    name="guided_filter",
    version="0.1",
    description="Learnable Guilded Filter Module",
    url="",
    author="wuhuikai",
    author_email="huikaiwu@icloud.com",
    # Exclude the build files.
    packages=find_packages(exclude=["test",]),
    zip_safe=True,
    install_requires=requirements
)