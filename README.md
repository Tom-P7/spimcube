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


### Todos

 - Implement a method to remove spikes from the raw data. Actually only ``despike.clean`` is used for the display but it is slow. ``remove_spikes`` is on his way!
 - Implement regex formulae to read the correct parameters for initialization of ``Spim`` object directly from the file name or from file containing metadata.
 - Make the code more flexible so it can handle unfinished map.


### License

This project is licensed under the terms of the [MIT license][MITLicense].

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

[MITLicense]: <https://github.com/Tom-P7/spimcube/blob/master/LICENSE>
