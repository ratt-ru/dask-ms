import os

try:
    from setuptools import setup, find_packages
except ImportError as e:
    raise ImportError("%s\nPlease install setuptools." % e)

extras_require = {
    'xarray': ["xarray > 0.12.0"],
    'testing': ['pytest', 'pytest-flake8 >= 1.0.6']
}

install_requires = [
    "dask[array] >= 2.2.0",
]

# ==================
# Detect readthedocs
# ==================

on_rtd = os.environ.get('READTHEDOCS') == 'True'

# Add binary blob packages if we're not on RTD
if not on_rtd:
    install_requires += [
        "numpy >= 1.14.0",
        "python-casacore >= 3.3.1",
    ]


def readme():
    with open("README.rst") as f:
        return f.read()


setup(name='dask-ms',
      version='0.2.6',
      description='xarray Datasets from CASA Tables.',
      long_description=readme(),
      url='http://github.com/ska-sa/dask-ms',
      classifiers=[
          "Programming Language :: Python :: 3",
          "Development Status :: 5 - Production/Stable",
          "Intended Audience :: Developers",
          "License :: OSI Approved :: BSD License",
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
      python_requires=">=3.6",
      zip_safe=True)
