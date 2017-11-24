#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Rockson Chang
"""

# setup.py
from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
                     'pyYaml>=3.12',
                     'keras>=2.0',
                     'matplotlib',
		             'h5py'
                     ]


setup(
    name='dcgan_trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Trainer package for DC-GAN',
    author='rocksonchang',
    zip_safe=False
)
