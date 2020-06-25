# Spimcube

``spimcube`` provides a graphical user interface (GUI) coded with ``matplotlib`` to explore datacube of spectra, in other words, tridimensional set of data containing spectra.

> Currently, the GUI is designed for datacubes obtained from a spatial mapping where a spectrum is recorded at each pixel of the map. But **in futur releases**, there will be **tools to explore general datacube** where each dimension can be: time, magnetic field, temperature, etc.

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

 - Implement a method to remove spikes from the raw data. 
 - Implement regex formulae to read the correct parameters for initialization of ``Spim`` object directly from the file name or from file containing metadata.
 -


### License

This project is licensed under the terms of the [MIT license][MITLicense].

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

[MITLicense]: <https://github.com/Tom-P7/spimcube/blob/master/LICENSE>
