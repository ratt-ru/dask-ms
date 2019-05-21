import os
import sys

try:
    from setuptools import setup, find_packages
    from setuptools.extension import Extension
    from setuptools.dist import Distribution
except ImportError as e:
    raise ImportError("%s\nPlease install setuptools." % e)


PY2 = sys.version_info[0] == 2

extras_require = {
    'testing': ['pytest', 'pytest-runner', 'mock']
}


install_requires = [
    "dask[array] >= 1.1.0",
    "six >= 1.10.0",
    "xarray >= 0.10.0",
]

if PY2:
    install_requires.append("futures >= 3.2.0")

# ==================
# Detect readthedocs
# ==================

on_rtd = os.environ.get('READTHEDOCS') == 'True'

# Add binary blob packages if we're not on RTD
if not on_rtd:
    install_requires += [
        "numpy >= 1.14.0",
        "python-casacore >= 2.2.1",
    ]


def readme():
    with open("README.rst") as f:
        return f.read()


setup(name='xarray-ms',
      version='0.1.4',
      description='xarray Datasets from CASA Tables.',
      long_description=readme(),
      url='http://github.com/ska-sa/xarray-ms',
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          "License :: Other/Proprietary License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "Topic :: Scientific/Engineering :: Astronomy",
      ],
      extras_require=extras_require,
      install_requires=install_requires,
      author='Simon Perkins',
      author_email='sperkins@ska.ac.za',
      packages=find_packages(),
      zip_safe=True)
