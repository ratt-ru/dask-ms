try:
    from setuptools import setup, find_packages
    from setuptools.extension import Extension
    from setuptools.dist import Distribution
except ImportError as e:
    raise ImportError("%s\nPlease install setuptools." % e)

def readme():
    with open("README.rst") as f:
        return f.read()

setup(name='xarray-ms',
    version="0.0.1",
    description='xarray Datasets from Tables.',
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
    author='Simon Perkins',
    author_email='sperkins@ska.ac.za',
    install_requires=[
        "attrs >= 17.2.0",
        "dask >= 0.17.1",
        "numpy >= 1.14.0",
        "six >= 1.10.0",
        "python-casacore >= 2.2.1",
        "toolz >= 0.8.2",
        "xarray >= 0.10.0",
    ],
    packages=find_packages(),
    zip_safe=True)
