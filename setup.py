import os

try:
    from setuptools import setup, find_packages
    from setuptools.extension import Extension
    from setuptools.dist import Distribution
except ImportError as e:
    raise ImportError("%s\nPlease install setuptools." % e)

install_requires = [
    "attrs >= 17.2.0",
    "dask >= 0.18.0",
    "six >= 1.10.0",
    "toolz >= 0.8.2",
    "xarray >= 0.10.0",
]

#===================
# Detect readthedocs
#====================

on_rtd = os.environ.get('READTHEDOCS') == 'True'

# Add binary blob packages if we're not on RTD
if not on_rtd:
    install_requires += [
        "numpy >= 1.14.0",
        "python-casacore >= 2.2.1",
    ]

setup_requirements = ['pytest-runner', ]

test_requirements = [
    'pytest',
    'mock']


def readme():
    with open("README.rst") as f:
        return f.read()


setup(name='xarray-ms',
      version='0.1.3',
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
      install_requires=install_requires,
      setup_requires=setup_requirements,
      tests_require=test_requirements,
      author='Simon Perkins',
      author_email='sperkins@ska.ac.za',
      packages=find_packages(),
      zip_safe=True)
