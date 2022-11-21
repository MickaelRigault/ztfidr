#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""
from distutils.core import setup
from setuptools import setup, find_packages


packages = find_packages()

VERSION = '0.7.14'
        
setup(name='ztfidr',
      version=VERSION,
      description='Tools for ZTF Ia *Internal*DataReleases',
      author='Mickael Rigault',
      author_email='m.rigault@ipnl.in2p3.fr',
      url='https://github.com/MickaelRigault/ztfdr',
      packages=packages,
#      package_data={'ztfdr': ['data/*']},
#      scripts=["bin/__.py",]
     )
# End of setupy.py ========================================================


