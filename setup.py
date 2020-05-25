from setuptools import setup

setup(
      name='spimcube',
      version='1.0',
      packages=['spimcube'],
      license='MIT',
      download_url='https://github.com/Tom-P7/spimcube/archive/v_1.0.tar.gz',
      
      install_requires=['matplotlib>=3.1', 'colorcet', 'numpy'],
      
      author="Tom-P7",
      description="This package enables the exploration of datacube of spectra within a rich GUI interface.",
      keywords=['spim', 'datacube', 'hyperspectral'],
      url='https://github.com/Tom-P7/spimcube',
      
      zip_safe=False,
)
