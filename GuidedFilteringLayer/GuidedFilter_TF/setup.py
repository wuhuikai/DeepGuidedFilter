#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="guided_filter_tf",
    version="1.0",
    description="Deep Guided Filtering Layer",
    url="https://github.com/wuhuikai/DeepGuidedFilter",
    author="wuhuikai",
    author_email="huikaiwu@icloud.com",
    # Exclude the build files.
    packages=find_packages(exclude=["test"]),
    zip_safe=True
)