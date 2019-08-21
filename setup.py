import os

try:
    from setuptools import setup, find_packages
except ImportError as e:
    raise ImportError("%s\nPlease install setuptools." % e)

extras_require = {
    'xarray': ["xarray > 0.10.0; python_version < '3.0'",
               "xarray > 0.12.0; python_version >= '3.5'"],
    'testing': ['mock', 'pytest', 'pytest-flake8']
}

install_requires = [
    "dask[array] == 1.2.2; python_version < '3.0'",
    "dask[array] >= 2.2.0; python_version >= '3.5'",
    "futures >= 3.2.0; python_version < '3.0'",
    "six >= 1.10.0",
]

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


setup(name='dask-ms',
      version='0.2.0-alpha2',
      description='xarray Datasets from CASA Tables.',
      long_description=readme(),
      url='http://github.com/ska-sa/dask-ms',
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
