from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='spimcube',
      version='1.5',
      description="This package enables the exploration of datacube of spectra within a GUI interface built on matplotlib and brings a handful of function tools.",
      keywords=['spim', 'datacube', 'hyperspectral, 3D map'],
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/Tom-P7/spimcube",
      
      author='Thomas Pelini',
      author_email='thomas.pelini@orange.fr',
      
      packages=['spimcube'],
      python_requires='>=3.6',
      install_requires=['matplotlib>=3.1', 'colorcet', 'numpy', 'despike'],
      license='MIT',
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
      zip_safe=False,
)
