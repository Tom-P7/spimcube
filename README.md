# Spimcube

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

``spimcube`` provides a graphical user interface (GUI) coded with ``matplotlib`` to explore datacube of spectra, in other words, tridimensional set of data containing spectra.

> Currently, the GUI is designed for datacubes obtained from a spatial mapping where a spectrum is recorded at each pixel of the map. But **in futur releases**, there will be **tools to explore general datacube** where each dimension can be: time, magnetic field, temperature, etc.

The package has three modules:

- ``spimclass`` containing the main objects implemented in the package, i.e ``Spim`` and ``SpimInterface``.
- ``functions`` mostly functions used internally in the classes defined in ``spimclass``.
- ``procedures`` included for personal purposes.

##### ``Spim`` object

Object used to initialize the set of data. It has multiple initialization methods depending on the user raw data.
It possesses methods allowing for diverse kind of map plot: the default one being "spectrally filtered spatial image" with the method ``intensity_map``.

##### ``SpimInterface`` object

Object that implement the GUI.
It is conceived in such a way that if the user identifies interesting data points or ROI within the GUI, he can easily switch to command line interface (CLI) to manipulate those. Actually, this GUI is thought as an exploration tool for datacube before going to more thorough analysis through classical CLI.

### Installation and upgrade

To install ``spimcube`` simply type in your terminal: 

```sh
$ pip install spimcube
```
To update to the last version of the package use:

```sh
$ pip install --upgrade spimcube
```

### Example of use

Start by importing the tools:
```sh
$ from spimcube.spimclass import (Spim, SpimInterface)
```

Depending on the default backend on your system, you may need to use the magic command ``%matplotlib`` available through ipython (either on terminal or jupyter notebook). Just type:
```sh
$ %matplotlib
```
which will allows for interactive windows.

Then, you may proceed as follows (presented below is the simplest case of use of the tools, with no options):

```sh

# If you are on UNIX OS (MacOS, Linux, etc.):
path = "/my/path/to/my/folder/"
filename = "name_of_the_file"

# If you are on Windows OS:
path = "C:\\my\\path\\to\\my\\folder\\"   # Don't forget the double backslash, otherwise python interpret '\' as an escape character.
filename = "name_of_the_file"


spim = Spim(path, filename)
spim.initialization_textfile()
```
Here, note that there exist different "initialization" methods. You can check those by typing:

```sh
dir(Spim)
```

Alternatively, a coordinate file can be used, which will additionaly plot a grid of the scanned positions:
```sh
spim.define_space_range(coordinate_file="/my/path/filename.txt")

# Also, ``define_space_range`` can be used to plot only a restricted ROI:
spim.define_space_range(area=(12, 24, 5, 17))  # In micrometers.

```

Finally, the spim can be explored with the GUI:

```sh
si = SpimInterface(spim)
```

Esthetics of plots can be controlled by normal attribute access:
```sh
si.image.set_interpolation('spline16')
si.image.set_cmap('viridis')
si.ax_spectrum.set_facecolor('blue')
```

> Check the docstring of each object and methods to see the numerous options available.

### Todos

 - Implement a method to remove spikes from the raw data. Currently only ``despike.clean`` is used for the display but is slow. ``remove_spikes`` is on his way!
 - Implement regex formulae to read the correct parameters for initialization of ``Spim`` object directly from the filename or from file containing metadata.


### Meta

Thomas Pelini - thomas.pelini@orange.fr

This project is licensed under the terms of the [MIT license][MITLicense].

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

https://github.com/Tom-P7/spimcube

[MITLicense]: <https://github.com/Tom-P7/spimcube/blob/master/LICENSE>
