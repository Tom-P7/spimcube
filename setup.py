from setuptools import setup

setup(
      name='spimcube',
      version='1.0',
      packages=['spimcube'],
      license='MIT',
      
      install_requires=['matplotlib>=3.1', 'colorcet', 'numpy']
      
      author="Tom-P7",
      description="This package enables the exploration of datacube of spectra within a rich GUI interface."
      keywords=['spim', 'datacube', 'hyperspectral']
      
      zip_safe=False
)
