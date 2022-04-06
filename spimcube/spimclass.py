import re
import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import (AxesWidget, Slider, Button, RadioButtons, CheckButtons, Cursor, MultiCursor,
                                RectangleSelector, Lasso)
from matplotlib import path
from matplotlib.collections import RegularPolyCollection
import numpy as np
import statistics as stat

from pywinspec.winspec import SpeFile
import despike
import indev.functions as fct


class Spim:
    """This class creates and formats the data cube (SPIM) to be explored and used with the ``SpimInterface`` object.
    
    The data cube is extracted from rawdata files by using one of the 'initialization' methods.
    Main attributes are the shaped data cube ``matrix`` and the list of wavelengths ``xaxis_values``.
    The created object contains also all the metadata declared as attributes.
    A method ``intensity_map`` is defined to compute intensity integrated image from the data cube.

    The choice here is to format the data so that the ``spim.matrix`` always corresponds to a scan
    being made by successive horizontal line from left to right starting at the bottom left corner.
    This is to be consistent with the approach of addressing the data in the ``SpimInterface`` class.

    Example:
    -------
    >> from spimcube.spimclass import (Spim, SpimInterface)
    
    >> path = "/my/path/to/my/data/"
    >> filename = "SPIM_hBN_4K_e266nm_100uW_5x1s_30x50steps_2x2size_slit20_1800gr_c310nm_200621a"

    >> spim = Spim(path, filename)
    >> spim.initialization_textfile()
    
    # Alternatively, a coordinate file can be used, which will additionaly plot a grid of the scanned positions.
    >> spim.define_space_range(coordinate_file="/my/path/filename.txt")
    # Or ``define_space_range`` can be used to plot only a restricted ROI.
    >> spim.define_space_range(area=(12, 24, 5, 17))

    # Then, the spim can be explored using ``SpimInterface`` explorator.

    >> si = SpimInterface(spim)

    -------

    """

    def __init__(self, path, filename, clean_spike=False, scan_direction='h', zigzag=False, scan_unit='step'):

        """
        Parameters
        ----------
        clean_spike : bool (default : False)
            If True, the method ``intensity_map`` returns an image after cleaning the spikes with
            ``despike.clean``. BEWARE!!! This slows down the interaction with the image by a factor 450.
                      
        scan_direction : one of ('hlb'='hl'='h', 'hrb'='hr', 'hlt', 'hrt', 'vtl'='vt'='v', 'vbl'='vb', 'vtr', vbr').
            Default is 'hlb'.
            1st letter : indicates the type of scan: h(orizontal), v(ertical).
            2nd letter : indicates the direction of scanning: l (left to right), r (right to left),
            t (top to bottom), b (bottom to top).
            3rd letter : indicates the starting corner: b(ottom), t(op), l(eft), r(ight),
            which complete the description of the scan orientation depending on the preceding letters.
            
            Example: 'hlb' means scanning by successive horizontal lines from left to right starting at the bottom left corner.
                     'vtr' means scanning by successive vertical lines from top to bottom starting at the top right corner.
                     'h' is equivalent to 'hl' and to 'hlb' because it is thought to as the logic way when scanning horizontally.
                     Same for 'v', 'vt' and 'vtl', when scanning vertically.
                         
        zigzag : bool. Indicates whether the scan is performed in zigzag mode or not. Default: False.

        scan_unit : str. Define the unit of the step in the scan.
            Example: 'µm' is the default, it can also be piezo stepper step, which are not equal to micrometer.
        
        """
        self.path = path
        self.filename = filename[:-4] if filename.endswith('.txt') else filename

        # Scan attributes
        self.scan_direction = scan_direction
        self.zigzag = zigzag
        self.scan_unit = scan_unit

        # Data attributes
        self.matrix = None
        self.xaxis_values = None

        # Attributes of the initialization
        self.start_x, self.start_y = 0, 0
        self.xstep_value, self.ystep_value = [None] * 2
        self.xstep_number, self.ystep_number = [None] * 2
        self.CCD_nb_pixel = None
        self.coordinate = None
        self.z_coordinate = None
        self.x_range, self.y_range = [None] * 2
        self.xpmin, self.xpmax, self.ypmin, self.ypmax = [None] * 4

        # Attributes of the intensity map: integration between min & max wavelength.
        self.int_lambda_min = None
        self.int_lambda_max = None
        self.clean_spike = clean_spike

    # Initialization methods: each method is intended to extract and process a certain format of raw data.
    # Make a new one for each new format of data.

    def initialization_qudi(self):
        """Use this function to initialize the SPIM with compressed '.npz' file from Qudi - L2C.

        Note: currently nothing is done with the z coordinates. It is only add to the attribute of the ``spim`` with
        the same shaped numpy array as the x & y arrays.

        """
        complete_path = self.path + self.filename
        data_file = complete_path + '.npz'
        metadata_file = complete_path + '.dat'

        self.scan_unit = 'step'

        # Extract metadata.
        with open(metadata_file, 'r') as file:
            for line in file:
                if line.startswith('#Data:'):
                    break
                if line.startswith('#start_x: '):
                    start_x = line[10:line.find('\n')]
                    self.start_x = round(float(start_x) * 1e6, 3)
                if line.startswith('#start_y: '):
                    start_y = line[10:line.find('\n')]
                    self.start_y = round(float(start_y) * 1e6, 3)
                if line.startswith('#step_x: '):
                    step_x = line[9:line.find('\n')]
                    self.xstep_value = round(float(step_x) * 1e6, 3)
                if line.startswith('#step_y: '):
                    step_y = line[9:line.find('\n')]
                    self.ystep_value = round(float(step_y) * 1e6, 3)
                if line.startswith('#shape: '):
                    shape = line[8:line.find('\n')]
                    shape = tuple(map(int, shape[1:-1].split(',')))
                    self.xstep_number, self.ystep_number = shape[0], shape[1]
                    self.CCD_nb_pixel = shape[2]

        # Extract x, y and z coordinates.
        file = np.load(data_file)
        x = file.f.x * 1e6
        y = file.f.y * 1e6
        z = file.f.z * 1e6
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
        coordinate = np.concatenate((x, y), axis=1)
        # I reshape the z positions arbitrary as if the scan was horizontal (TO CORRECT)
        self.z_coordinate = np.reshape(z, (self.ystep_number, self.xstep_number))

        # Extract the list of wavelengths on the spectro CCD.
        lambda_key = file.files[3:]
        self.xaxis_values = np.array(list(map(lambda s: round(float(s) * 1e9, ndigits=3), lambda_key)))
        if len(self.xaxis_values) != self.CCD_nb_pixel:
            raise ValueError(
                "The number of CCD pixels given in metadata does not match the number of wavelengths in data.")

        # Extract the intensities.
        data = []
        for i, item in enumerate(file):
            if i >= 3:
                data.append(file[item])
        file.close()
        data = np.array(data)
        data = data.T

        # Check and fix incomplete scan.
        data = self._fix_incomplete_scan(data)
        self._shape3Dmatrix(data)

        # Define default space range of the map.
        self.define_space_range(coordinate=coordinate)

    def initialization(self):
        """This void function is used to issue a deprecation warning."""
        raise DeprecationWarning("This function name has changed, use ``initialization_labviewL2C`` instead.")
    def initialization_labviewL2C(self):
        """Use this function to initialize the SPIM with binary datafile from L2C - Labview."""
        # Create the complete filenames with extension.
        complete_path = self.path + self.filename
        data_file = complete_path + ".dat"
        lambda_file = complete_path + ".lambda"
        init_file = complete_path + ".ini"

        self.scan_unit = 'µm'

        # Load 'data' file, extract the first three values that are metadata and remove them.
        data = np.fromfile(data_file, dtype='>i4')

        self.xstep_number = data[0]
        self.ystep_number = data[1]
        self.CCD_nb_pixel = data[2]
        data = data[3:]

        # Check and fix incomplete scan.
        data = self._fix_incomplete_scan(data)

        # Reshape the matrix.
        self._shape3Dmatrix(data)

        # Load Lambda file in float.
        self.xaxis_values = np.loadtxt(
            lambda_file, encoding='latin1', converters={0: lambda s: s.replace(",", ".")}
        )

        # Fetch metadata with regex.
        self.xstep_value = fct.get_value("PasX", init_file)
        self.ystep_value = fct.get_value("PasY", init_file)

        # Define default space range of the map.
        self.define_space_range()

    def initialization_textfile(self, xstep_number=None, ystep_number=None, xstep_value=None, ystep_value=None,
                                CCD_nb_pixel='auto', data_reversed=False):
        """Use this function to initialize the ``Spim`` with text formatted datafile.
        
        The data text file is expected to be formatted as follows:
            --> First column : wavelengths
            --> Second column : intensities
        If the order is inverse, use the parameter ``data_reversed == True``.
        Both columns must have length equal to ``CCD_nb_pixel`` x ``xstep_number`` x ``ystep_number``.

        If some values among {x,y}step_{number,value} are 'None', a regex procedure proceeds to detect the values
        directly from the name of the file. An error is raised if not found.

        :param xstep_number: (scalar) specify the number of steps in x.
        :param ystep_number: (scalar) specify the number of steps in y.
        :param xstep_value: (scalar) specify the step size in x.
        :param ystep_value: (scalar) specify the step size in y.
        :param data_reversed: (bool) Specify the order of the column in the data text file. Default to False.
            - False --> (1-wavelength, 2-intensity).
            - True  --> (1-intensity, 2-wavelength).
        :param CCD_nb_pixel: (integer or string) Specify the number of pixels on the CCD camera.
            - 'auto' : the number of pixels is found based on the analysis of the wavelength column.
            - integer : user defined value.

        """
        complete_path = self.path + self.filename + ".txt"

        self.scan_unit = 'step'

        # Load data. ``column_1`` is wavelength, ``column_2`` is intensity.
        usecols = (0, 1) if not data_reversed else (1, 0)
        column_1, column_2 = np.loadtxt(complete_path, unpack=True, usecols=usecols, comments='Frame')

        # Find ``CCD_nb_pixel``.
        if CCD_nb_pixel == 'auto':
            n = 1
            while column_1[n] != column_1[0]:
                n += 1
            self.CCD_nb_pixel = n
        elif isinstance(CCD_nb_pixel, int):
            self.CCD_nb_pixel = CCD_nb_pixel

        # Fetch metadata with regex.
        if not all([xstep_number, ystep_number]):
            pattern = r"(\w*)x(\w*)" + "steps"  # Specific to how we write the filename.
            values = re.findall(pattern, self.filename)
            if values:
                self.xstep_number = int(values[0][0])
                self.ystep_number = int(values[0][1])
            else:
                raise AttributeError("``xstep_number`` and ``ystep_number`` not found in filename, must be provided.")
        else:
            self.xstep_number = xstep_number
            self.ystep_number = ystep_number

        if not all([xstep_value, ystep_value]):
            pattern = r"(\w*)x(\w*)" + "pixel"  # Specific to how we write the filename.
            values = re.findall(pattern, self.filename)
            if values:
                self.xstep_value = int(values[0][0])
                self.ystep_value = int(values[0][1])
            else:
                raise AttributeError("``xstep_value`` and ``ystep_value`` not found in filename, must be provided.")
        else:
            self.xstep_value = xstep_value
            self.ystep_value = ystep_value

        # Check and fix incomplete scan.
        column_2 = self._fix_incomplete_scan(column_2)

        # Reshape the matrix.
        self._shape3Dmatrix(column_2)

        # Create the list of x axis values.
        self.xaxis_values = column_1[0:self.CCD_nb_pixel]

        # Define default space range of the map.
        self.define_space_range()

    def initialization_spefile(self, xstep_number=None, ystep_number=None, xstep_value=None, ystep_value=None):
        self.xstep_number = xstep_number
        self.ystep_number = ystep_number
        self.xstep_value = xstep_value
        self.ystep_value = ystep_value

        spefile = SpeFile(self.path + self.filename + '.SPE')
        self.xaxis_values = spefile.xaxis
        data = spefile.data

        self.CCD_nb_pixel = data.shape[1]

        # Check and fix incomplete scan.
        flatten_data = self._fix_incomplete_scan(data)

        # Reshape the matrix.
        self._shape3Dmatrix(data)

        # Define default space range of the map.
        self.define_space_range()

    def define_space_range(self, area=None, coordinate_file=None, coordinate=None):
        """Define the ROI in micrometer unit that is displayed.

        This function has nothing to do with the metadata of the scan, it only allows for ROI choice, for example
        if only a small section of the scan is of interest. If no parameter is given, default to the entire scan area.
        It generates the subsequent x & y scanned positions as well as the x & y min/max pixels positions.
        
        Parameters
        ----------
        area : tuple of 4 floats: ``(xmin, xmax, ymin, ymax)`` representing the limits of the intensity image in
        micrometers. If ``None``, then the full area defined by ``x_range`` and ``y_range`` is supposed.
               
        coordinate_file : the full filename (with the path) containing the scanned positions.
                          If provided, take  precedence on ``area``.

        coordinate : the numpy array containing the x, y positions in the shape: (xstep_number*ystep_number, 2).
        
        """
        if coordinate_file is None and coordinate is None:
            self.coordinate = None
            # Generate X & Y 1D arrays of scanned positions.
            self.x_range = np.linspace(self.start_x + self.xstep_value, self.xstep_value * self.xstep_number,
                                       self.xstep_number, dtype='>f4')  # Builds 1D array of X positions
            self.y_range = np.linspace(self.start_y + self.ystep_value, self.ystep_value * self.ystep_number,
                                       self.ystep_number, dtype='>f4')  # and Y positions in ``scan_unit``.
            if area is None:
                area = (0, self.x_range.max(), 0, self.y_range.max())
            space_area = area  # Define plot area in ``scan_unit`` unit: (xmin, xmax, ymin, ymax).
            pixel_area = []  # Define plot area in pixel unit.

            for i, parameter in enumerate(space_area):
                pixel_number = fct.find_nearest(self.x_range if i < 2 else self.y_range, parameter)
                pixel_area.append(pixel_number)

            self.xpmin, self.xpmax, self.ypmin, self.ypmax = pixel_area

        else:
            if coordinate is None:
                # Get the matrix of coordinates.
                coordinate = np.loadtxt(coordinate_file)
            coordinate = self._fix_incomplete_coordinate(coordinate)
            self.coordinate = coordinate
            self._shape_coordinate(coordinate)
            # Generate X & Y 1D arrays of scanned pixels.
            self.x_range = np.arange(0, self.xstep_number, 1)
            self.y_range = np.arange(0, self.ystep_number, 1)
            # Define plot area in pixel.
            self.xpmin, self.xpmax = [0, 0], [len(self.x_range) - 1, self.x_range[-1]]
            self.ypmin, self.ypmax = [0, 0], [len(self.y_range) - 1, self.y_range[-1]]

    def intensity_map(self, center=None, width=None):
        """Builds the intensity image data in the desired spectral range.
        
        Generate a 2D array contening intensity of the PL integrated over the chosen range in wavelength.
        Takes ``center`` and ``width`` values for determining the wavelength range.
        Note: in the present form, the intensity is normalized to the maximum into the range considered.
        
        """
        center = self.xaxis_values[int(len(self.xaxis_values) / 2)] if center is None else center
        width = 1 / 10 * (self.xaxis_values.max() - self.xaxis_values.min()) if width is None else width
        int_lambda_min = center - width / 2
        int_lambda_max = center + width / 2
        # The function ``find_nearest`` return the pixel number and the corresponding value in nm.
        self.int_lambda_min = fct.find_nearest(self.xaxis_values, int_lambda_min)
        self.int_lambda_max = fct.find_nearest(self.xaxis_values, int_lambda_max)
        # Integrate intensity over the chosen range of wavelength.
        image_data = np.sum(
            self.matrix[self.ypmin[0]:self.ypmax[0] + 1, self.xpmin[0]:self.xpmax[0] + 1,
                        self.int_lambda_min[0]:self.int_lambda_max[0] + 1],
            axis=2,
        )
        if self.clean_spike:
            # Clean the image from the spikes.
            image_data = despike.clean(image_data)
        #return image_data / np.max(image_data)
        # The reason I changed the return array for the one below is because when there is a constant background
        # then substracting the minimum value to all values allows for a more contrasted intensity image without the
        # need to adjust the color limit sliders.
        return (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

    def _fix_incomplete_scan(self, array_to_fix):
        """Check whether or not the input data is as large as expected and fix it if necessary.

        - If the input data set is smaller than the expected size accordingly to ``xstep_number`` and ``ystep_number``,
        then the data set is completed with zeros.
        - If it is larger, a ValueErrror is raised.

        array_to_fix : numpy array containing the data. The array is ravel so no need to do any reshaping before
            passing it to this method.

        """
        expected_length = self.xstep_number * self.ystep_number * self.CCD_nb_pixel
        array_to_fix = np.array(array_to_fix)

        if len(array_to_fix.ravel()) > expected_length:
            raise ValueError(
                "The input data set is larger than the specified number of scan positions.\
                Double-check ``xstep_number`` and ``ystep_number``.")
        elif len(array_to_fix.ravel()) < expected_length:
            delta_length = expected_length - len(array_to_fix.ravel())
            patch_array = np.zeros(shape=delta_length)
            array_fixed = np.concatenate((array_to_fix.ravel(), patch_array))
            # warnings.warn("The data set was smaller than the specified number of scan positions. "
            #              "It has been completed with zeros.")
            print("INCOMPLETE SCAN, data set has been completed with zeros accordingly to the specified number "
                  "of scan positions")
            return array_fixed
        else:
            return array_to_fix

    def _fix_incomplete_coordinate(self, array_to_fix):
        expected_length = self.xstep_number * self.ystep_number * 2
        array_to_fix = np.array(array_to_fix)

        if len(array_to_fix.ravel()) > expected_length:
            raise ValueError(
                "The input data set is larger than the specified number of scan positions.\
                Double-check ``xstep_number`` and ``ystep_number``.")
        elif len(array_to_fix.ravel()) < expected_length:
            delta_length = expected_length - len(array_to_fix.ravel())
            patch_array = np.zeros(shape=delta_length)
            array_fixed = np.concatenate((array_to_fix.ravel(), patch_array)).reshape((int(expected_length/2), 2))
            # warnings.warn("The coordinate data set was smaller than the specified number of scan positions. "
            #              "It has been completed with zeros.")
            print("INCOMPLETE SCAN, data set has been completed with zeros accordingly to the specified number "
                  "of scan positions")
            return array_fixed
        else:
            return array_to_fix

    def _shape3Dmatrix(self, column_of_intensity):
        """Format the 3D matrix in desired dimensions.

        The reshaping of the matrix depends on the type of the scan (horizontal, vertical),
        the direction of the scanning (left to right, top to bottom, etc.) and finally
        on the starting point (top left corner, bottom right corner, etc.). There are 8 possibilities.
        At the end, whatever the type of scan and direction, the matrix is formated as if it was a ``'hlb'`` scan.

        """
        if self.scan_direction in ['hlb', 'hl', 'h']:
            self.matrix = column_of_intensity.reshape(self.ystep_number, self.xstep_number, self.CCD_nb_pixel)
            # No further transformation, this is the reference type of scan.
        elif self.scan_direction in ['hrb', 'hr']:
            self.matrix = column_of_intensity.reshape(self.ystep_number, self.xstep_number, self.CCD_nb_pixel)
            self.matrix = np.flip(self.matrix, axis=1)
        elif self.scan_direction == 'hlt':
            self.matrix = column_of_intensity.reshape(self.ystep_number, self.xstep_number, self.CCD_nb_pixel)
            self.matrix = np.flip(self.matrix, axis=0)
        elif self.scan_direction == 'hrt':
            self.matrix = column_of_intensity.reshape(self.ystep_number, self.xstep_number, self.CCD_nb_pixel)
            self.matrix = np.flip(np.flip(self.matrix, axis=0), axis=1)
        elif self.scan_direction in ['vtl', 'vt', 'v']:
            self.matrix = column_of_intensity.reshape(self.xstep_number, self.ystep_number, self.CCD_nb_pixel)
            self.matrix = np.flip(self.matrix, axis=1).transpose((1, 0, 2))
        elif self.scan_direction in ['vbl', 'vb']:
            self.matrix = column_of_intensity.reshape(self.xstep_number, self.ystep_number, self.CCD_nb_pixel)
            self.matrix = self.matrix.transpose((1, 0, 2))
        elif self.scan_direction == 'vtr':
            self.matrix = column_of_intensity.reshape(self.xstep_number, self.ystep_number, self.CCD_nb_pixel)
            self.matrix = np.flip(np.flip(self.matrix, axis=0), axis=1).transpose((1, 0, 2))
        elif self.scan_direction == 'vbr':
            self.matrix = column_of_intensity.reshape(self.xstep_number, self.ystep_number, self.CCD_nb_pixel)
            self.matrix = np.flip(self.matrix, axis=0).transpose((1, 0, 2))
        else:
            raise ValueError(
                "Innapropriate value for ``scan_direction``. Type 'help(Spim)' and refers to the __init__ section."
            )

        # Inverse one in two line if ``zigzag`` is 'True', starting with the second line.
        if self.zigzag:
            for i in range(1, self.matrix.shape[0], 2):
                self.matrix[i, :, :] = np.flip(self.matrix[i, :, :], axis=0)

    def _shape_coordinate(self, coordinate):
        """Format the coordinate file in desired dimensions."""
        if self.coordinate is None:
            return

        if self.scan_direction in ['hlb', 'hl', 'h']:
            self.coordinate = coordinate.reshape(self.ystep_number, self.xstep_number, 2)
            # No further transformation, this is the reference type of scan.
        elif self.scan_direction in ['hrb', 'hr']:
            self.coordinate = coordinate.reshape(self.ystep_number, self.xstep_number, 2)
            self.coordinate = np.flip(self.coordinate, axis=1)
        elif self.scan_direction == 'hlt':
            self.coordinate = coordinate.reshape(self.ystep_number, self.xstep_number, 2)
            self.coordinate = np.flip(self.coordinate, axis=0)
        elif self.scan_direction == 'hrt':
            self.coordinate = coordinate.reshape(self.ystep_number, self.xstep_number, 2)
            self.coordinate = np.flip(np.flip(self.coordinate, axis=0), axis=1)
        elif self.scan_direction in ['vtl', 'vt', 'v']:
            self.coordinate = coordinate.reshape(self.xstep_number, self.ystep_number, 2)
            self.coordinate = np.flip(self.coordinate, axis=1).transpose((1, 0, 2))
        elif self.scan_direction in ['vbl', 'vb']:
            self.coordinate = coordinate.reshape(self.xstep_number, self.ystep_number, 2)
            self.coordinate = self.coordinate.transpose((1, 0, 2))
        elif self.scan_direction == 'vtr':
            self.coordinate = coordinate.reshape(self.xstep_number, self.ystep_number, 2)
            self.coordinate = np.flip(np.flip(self.coordinate, axis=0), axis=1).transpose((1, 0, 2))
        elif self.scan_direction == 'vbr':
            self.coordinate = coordinate.reshape(self.xstep_number, self.ystep_number, 2)
            self.coordinate = np.flip(self.coordinate, axis=0).transpose((1, 0, 2))
        else:
            raise ValueError(
                "Innapropriate value for ``scan_direction``. Type 'help(Spim)' and refers to the __init__ section."
            )

        # Inverse one in two line if ``zigzag`` is 'True', starting with the second line.
        if self.zigzag:
            for i in range(1, self.coordinate.shape[0], 2):
                self.coordinate[i, :, :] = np.flip(self.coordinate[i, :, :], axis=0)


class SpimInterface:
    """This class allows the exploration of a SPIM datacube.

    A rich interface with multiple widgets make the exploration easier, with the possibility to
    extract some spectra and filtered image in separate windows.

    Example:
    -------
    # One first has to create an object ```Spim`` to be explored. See docstring of ``Spim`` object for help.

    # Then, the spim can be explored using ``SpimInterface`` explorator.

    >> si = SpimInterface(spim)

    # Esthetics of plots can be controlled by normal attribute access:
    >> si.image.set_interpolation('spline16')
    >> si.image.set_cmap('viridis')
    >> si.ax_spectrum.set_facecolor('blue')

    -------
    
    """

    def __init__(self, spim, lambda_init=None, fig_fc='dimgrey', spec_fc='whitesmoke'):
        """
        Parameters
        ----------
        spim : the ``Spim`` datacube to be explored.
        
        lambda_init : Initial central wavelength to compute the intensity image.
                      If out of bound: set to the closest value.
                      If ``None``: most common intense pixel of the SPIM. If more than one are most common,
                      default to most common intense pixel of the first spectrum.

        fig_fc : facecolor of the GUI.

        spec_fc : facecolor of the spectrum window.
        
        """
        if not isinstance(spim, Spim):
            raise ValueError('Spim is not an instance from the ``Spim`` class.')
        if lambda_init is None:
            try:
                lambda_init = spim.xaxis_values[stat.mode(np.argmax(spim.matrix, axis=2).ravel())]
            except stat.StatisticsError:
                lambda_init = spim.xaxis_values[np.argmax(spim.matrix[0, 0, :])]
        # ``spim`` is given as attribute in order to use it in the class methods that define widgets action.
        self.spim = spim
        self.scan_unit = spim.scan_unit

        # Variables for layout.
        l1, b0, w0, h0 = 0.05, 0.22, 0.4, 0.72
        l2, l3, l4 = 0.51, 0.297, 0.825
        b1, b2, b3, b4 = 0.06625, 0.0925, 0.02, 0.0425
        w1, w2, w3 = 0.05, 0.07, 0.27
        h1, h2, h3 = 0.05, 0.0625, 0.04
        hsp1, hsp2 = 0.01, 0.027
        vsp1, vsp2 = 0.03, 0.015

        # Figure and main axes - attributes
        self.fig = plt.figure(figsize=(14, 6.54), facecolor=fig_fc)
        self.ax_spectrum = self.fig.add_axes([l1, b0, w0, h0], facecolor=spec_fc)
        for spine in self.ax_spectrum.spines.values():
            spine.set_lw(1.7)
            spine.set_color('k')  # 'dimgrey'
        self.ax_image = self.fig.add_axes([l2, b0, w0, h0])
        for spine in self.ax_image.spines.values():
            spine.set_lw(1.7)
            spine.set_color('k')
        self.ax_colorbar = self.fig.add_axes([0.915, b0, 0.02, h0])

        # Widgets axes positions - not attributes
        widget_color = 'lightgoldenrodyellow'
        ## - Slider -
        ax__slider_center = self.fig.add_axes([l2, b2, w3, h3], facecolor=widget_color)
        ax__slider_width = self.fig.add_axes([l2, b4, w3, h3], facecolor=widget_color)
        ax__slider_clim_min = self.fig.add_axes([0.97, b0 + vsp1, 0.015, h0 / 2 - vsp1 - vsp2 / 2], fc=widget_color)
        ax__slider_clim_max = self.fig.add_axes([0.97, b0 + h0 / 2 + vsp2 / 2, 0.015, h0 / 2 - vsp1 - vsp2 / 2],
                                                fc=widget_color)
        ## - Button -
        ax__button_reset_sliders = self.fig.add_axes([l4 + w1 + hsp1, b4, w1, h3])
        ax__button_full_range = self.fig.add_axes([l4, b4, w1, h3])
        ax__button_delta_minus = self.fig.add_axes([l4, b2, (w1 - 0.003) / 2, h3])
        ax__button_delta_plus = self.fig.add_axes([l4 + (w1 + 0.003) / 2, b2, (w1 - 0.003) / 2, h3])
        ax__button_plot_image = self.fig.add_axes([l4 + w1 + hsp1, b2, w1, h3])
        ax__button_save_spect = self.fig.add_axes([l1, b2, w1, h3])
        ax__button_plot_spect = self.fig.add_axes([l1, b4, w1, h3])
        ax__button_reset_spect = self.fig.add_axes([l1 + w1 + hsp1, b4, w1, h3])
        ax__button_add_slider = self.fig.add_axes([l1 + 2*w1 + 2*hsp1, b2, w1, h3])
        ax__button_delete_slider = self.fig.add_axes([l1 + 3*w1 + 3*hsp1, b2, w1, h3])
        ax__button_switch_slider = self.fig.add_axes([l1 + 2.5*w1 + 2.5*hsp1, b4, w1, h3])
        # Indication that the above three buttons pertain to sliders.
        self.fig.text((l1+2.5*w1+2.5*hsp1) + w1/2, b4-0.5*h3, '-------Sliders-------', va='center', ha='center',
                      fontsize=12)
        ## - RadioButtons -
        ax__radiobutton_yscale = self.fig.add_axes([l3 + w2 + hsp1, b3, w2, h2], facecolor=widget_color)
        ax__radiobutton_ylim = self.fig.add_axes([l3 + w2 + hsp1, b2, w2, h2], facecolor=widget_color)
        ax__radiobutton_xunit = self.fig.add_axes([l3, b3, w2, h2], facecolor=widget_color)
        ## - Checkbutton -
        ax__checkbutton_measure = self.fig.add_axes([l3, b2, w2, h2], facecolor=widget_color)

        # Widgets definitions - attributes
        ## - Slider -
        self.slider_center = FancySlider(
            ax__slider_center, 'Center\n[nm]', spim.xaxis_values.min() + 0.01, spim.xaxis_values.max(), valinit=lambda_init)
        self.slider_width = FancySlider(
            ax__slider_width, 'Width\n[nm]', 0.02, spim.xaxis_values.max() - spim.xaxis_values.min(), valinit=3)
        self.slider_clim_min = FancySlider(
            ax__slider_clim_min, 'Min', 0, 1, valinit=0, orientation='vertical')
        self.slider_clim_max = FancySlider(
            ax__slider_clim_max, 'Max', 0, 1, valinit=1, orientation='vertical', slidermin=self.slider_clim_min)
        self.slider_clim_max.label.set_position((0.5, 1.06))
        self.slider_clim_max.valtext.set_position((0.5, 1.06))
        self.slider_clim_min.label.set_position((0.5, -0.15))
        ### Connection to callback functions.
        self.slider_center.on_changed(self._update_image)
        self.slider_width.on_changed(self._update_image)
        self.slider_clim_min.on_changed(self._apply_clim)
        self.slider_clim_max.on_changed(self._apply_clim)
        ## - Button -
        # prop = dict(color=widget_color, hovercolor='0.975')
        prop = {}
        self.button_reset_sliders = FancyButton(ax__button_reset_sliders, 'Reset', **prop)
        self.button_full_range = FancyButton(ax__button_full_range, 'Full', **prop)
        self.button_delta_plus = FancyButton(ax__button_delta_plus, '+', **prop)
        self.button_delta_minus = FancyButton(ax__button_delta_minus, '-', **prop)
        self.button_plot_image = FancyButton(ax__button_plot_image, 'Plot', **prop)
        self.button_save_spect = FancyButton(ax__button_save_spect, 'Save', **prop)
        self.button_plot_spect = FancyButton(ax__button_plot_spect, 'Plot', **prop)
        self.button_reset_spect = FancyButton(ax__button_reset_spect, 'Reset', **prop)
        self.button_add_slider = FancyButton(ax__button_add_slider, 'Add', **prop)
        self.button_delete_slider = FancyButton(ax__button_delete_slider, 'Delete', **prop)
        self.button_switch_slider = FancyButton(ax__button_switch_slider, 'Switch', **prop)
        ### Connection to callback functions.
        self.button_reset_sliders.on_clicked(self._reset_sliders)
        self.button_full_range.on_clicked(self._set_full_range)
        self.button_delta_minus.on_clicked(self._delta_minus)
        self.button_delta_plus.on_clicked(self._delta_plus)
        self.button_plot_image.on_clicked(self._plot_image)
        self.button_save_spect.on_clicked(self._save_spect)
        self.button_plot_spect.on_clicked(self._plot_spect)
        self.button_reset_spect.on_clicked(self._reset_spect)
        ## - RadioButtons -
        self.radiobutton_yscale = FancyRadioButtons(ax__radiobutton_yscale, ('Linear', 'Log'))
        self.radiobutton_ylim = FancyRadioButtons(ax__radiobutton_ylim, ('Autoscale', 'Lock'))
        self.radiobutton_xunit = FancyRadioButtons(ax__radiobutton_xunit, ('nm', 'eV'))
        ### Connection to callback functions.
        self.radiobutton_yscale.on_clicked(self._apply_yscale)
        self.radiobutton_ylim.on_clicked(self._apply_ylim)
        self.radiobutton_xunit.on_clicked(self._set_xunit)
        ## - Checkbuttons -
        self.checkbutton_measure = FancyCheckButtons(ax__checkbutton_measure, ['Indicator', 'Selector'])
        ### Connection to callback functions.
        self.checkbutton_measure.on_clicked(self._display_tools)
        ## - Cursor -
        self.cursor_image = Cursor(self.ax_image, useblit=True, color='w', lw=0.5)
        ## - Indicator -
        self.indicator_spectrum = Indicator(self.ax_spectrum, color='k', lw=0.5, ls='-.', stick_to_data=False)
        self.indicator_spectrum.set_active(False)
        ## - RectangleSelector -
        rectprops = dict(facecolor='white', edgecolor='green', alpha=0.7, fill=True)
        self.rect_selector = RectangleSelector(self.ax_image, self._rectselect_action, drawtype='box', useblit=True,
                                               button=[1, 3], spancoords='data', interactive=True, rectprops=rectprops)
        self.rect_selector.set_active(False)
        self.fig.canvas.mpl_connect('draw_event', self._rectselect_persist)

        # ---------- --- ---------- ------- ----- --- ---------
        # Attributes for connection between image and spectrum.
        # ---------- --- ---------- ------- ----- --- ---------
        self.indices_pixels_clicked = [[0, 0]]  # The spectrum [0, 0] is displayed at the beginning.
        self.indices_pixels_saved = []
        self.indices_rect_drawn = []
        self.indices_rect_saved = []
        self.cid_image = self.fig.canvas.mpl_connect('button_release_event', self._image_onclick)

        # ---- --- ---- -- ----- ----------- -- ---------
        # Plot the grid of pixel coordinates if provided.
        # ---- --- ---- -- ----- ----------- -- ---------
        if spim.coordinate is not None:
            cxy = spim.coordinate.reshape((spim.coordinate.shape[0] * spim.coordinate.shape[1], 2))
            style = 'mystyle' if 'mystyle' in plt.style.available else 'seaborn-whitegrid'
            with plt.style.context(style):
                fig_c, ax_c = plt.subplots(1, 1, figsize=(6, 6))
                prop = dict(marker='+', color='k', s=30, lw=1)
                ax_c.scatter(cxy[:, 0], cxy[:, 1], **prop)

        # --- --------- ----- ---
        # --- INTENSITY IMAGE ---
        # --- --------- ----- ---
        image_data = spim.intensity_map(center=self.slider_center.valinit, width=self.slider_width.valinit)
        norm = mpl.colors.Normalize(vmin=np.min(image_data), vmax=np.max(image_data))
        if spim.coordinate is None:
            extent = (spim.xpmin[1] - spim.xstep_value, spim.xpmax[1], spim.ypmin[1] - spim.ystep_value, spim.ypmax[1])
        else:
            extent = (spim.xpmin[1] - 0.5, spim.xpmax[1] + 0.5, spim.ypmin[1] - 0.5, spim.ypmax[1] + 0.5)
        # Plot the intensity image.
        self.image = self.ax_image.imshow(
            image_data,
            cmap='inferno',
            norm=norm,
            interpolation=None,
            origin='lower',
            extent=extent,
            clim=(0, 1),
            aspect='equal',
            # 'auto': image will fit the axes limits but pixels will not be square. 'equal': for square pixels.
        )
        # self.ax_image.grid(lw=0.2, color='k') # Not pleasing when zooming, lines superimpose with pixel borders
        self.ax_image.set_title(self._get_integration_range(), pad=15)
        self.ax_image.set_xlabel("x [{}]".format(self.scan_unit))
        self.ax_image.set_ylabel("y [{}]".format(self.scan_unit))
        # Set ticks and their properties.
        self.ax_image.tick_params(axis='both', direction='in', length=4.3, width=1.2)
        self.colorbar = plt.colorbar(self.image, cax=self.ax_colorbar)
        self.ax_colorbar.tick_params(direction='inout')

        # --- ----------- ---
        # --- PL SPECTRUM ---
        # --- ----------- ---
        self.spectrum, = self.ax_spectrum.plot(spim.xaxis_values, spim.matrix[0, 0, :], lw=1, c='darkviolet')
        self.ax_spectrum.set_xlim(spim.xaxis_values.min(), spim.xaxis_values.max())
        self.ax_spectrum.set_ylim(np.min(spim.matrix[0, 0, :]), np.max(spim.matrix[0, 0, :]))
        # Define vertical marker lines for the spectrum - attributes.
        prop1 = dict(ls='-', color='k', lw=0.7)
        prop2 = dict(ls='--', color='k', lw=0.7)
        self.marker_center = self.ax_spectrum.axvline(x=lambda_init, ymin=0, ymax=1, **prop1)
        self.marker_left = self.ax_spectrum.axvline(x=lambda_init - self.slider_width.val / 2, ymin=0, ymax=1, **prop2)
        self.marker_right = self.ax_spectrum.axvline(x=lambda_init + self.slider_width.val / 2, ymin=0, ymax=1, **prop2)

        self.ax_spectrum.set_title(self._get_spectrum_location(), pad=15)
        self.ax_spectrum.set_xlabel("Wavelength [nm]")
        self.ax_spectrum.set_ylabel("PL intensity [a.u]")
        # Set ticks and their properties.
        self.ax_spectrum.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        self.ax_spectrum.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        self.ax_spectrum.tick_params(axis='both', which='major', direction='in', length=4.3, width=1.2)
        self.ax_spectrum.tick_params(axis='both', which='minor', direction='in', length=3, width=0.6)
        self.ax_spectrum.tick_params(axis='x', which='both', top=True)
        self.ax_spectrum.tick_params(axis='y', which='both', right=True)
        """
        # Create a second x axis in eV
        self.axtwin = self.ax_spectrum.twiny()
        self.axtwin.xaxis.set_major_locator(self.ax_spectrum.xaxis.get_major_locator())
        self.axtwin.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: round(1239.84193/x, 3)))
        self.axtwin.set_xbound(self.ax_spectrum.get_xbound())
        self.axtwin.tick_params(axis='x', which='major', direction='in', length=4.3, width=1.2)
        self.axtwin.set_xlabel("energy [eV]", labelpad=10)
        #self.indicator_spectrum = Indicator(self.axtwin, color='k', lw=1, ls='-.')
        """

        # Finally.
        plt.show()

    def _get_integration_range(self):
        """Return the integration range of the intensity image as a string in nm and eV."""
        int_lambda_min = self.spim.int_lambda_min[1]
        int_lambda_max = self.spim.int_lambda_max[1]
        title_pattern = 'Integration range : [{:.2f}, {:.2f}]nm | [{:.3f}, {:.3f}]eV'
        title = title_pattern.format(
            int_lambda_min, int_lambda_max, fct.nm_eV(int_lambda_max), fct.nm_eV(int_lambda_min))
        return title

    def _get_spectrum_location(self):
        """Return the indices and spatial position of the last clicked spectrum as a string."""
        title_pattern = "PL spectrum : [{}, {}] --- [x={}, y={}]{}"
        idx1 = self.indices_pixels_clicked[-1][0]
        idx2 = self.indices_pixels_clicked[-1][1]
        if self.spim.coordinate is None:
            return title_pattern.format(idx1, idx2, str(self.spim.x_range[idx2]), str(self.spim.y_range[idx1]),
                                        self.scan_unit)
        else:
            return title_pattern.format(idx1, idx2, str(round(self.spim.coordinate[idx1, idx2][0], 3)),
                                        str(round(self.spim.coordinate[idx1, idx2][1], 3)), self.scan_unit)

    def _update_image(self, _):
        """Update the image and spectrum markers when sliders 'center' or 'width' are changed."""
        # Update sliders value.
        center = self.slider_center.val
        width = self.slider_width.val
        # Update image.
        self.image.set_data(self.spim.intensity_map(center, width))
        # Update integration range.
        self.ax_image.set_title(self._get_integration_range(), pad=15)
        # Update markers position.
        func_unit = fct.nm_eV if self.indicator_spectrum.unit_eV else lambda x: x
        self.marker_left.set_xdata(func_unit(self.slider_center.val - self.slider_width.val / 2))
        self.marker_center.set_xdata(func_unit(self.slider_center.val))
        self.marker_right.set_xdata(func_unit(self.slider_center.val + self.slider_width.val / 2))

    def _apply_clim(self, _):
        self.image.set_clim(vmin=self.slider_clim_min.val, vmax=self.slider_clim_max.val)

    def _reset_sliders(self, _):
        self.slider_center.reset()
        self.slider_width.reset()
        self.slider_clim_min.reset()
        self.slider_clim_max.reset()

    def _set_full_range(self, _):
        center = self.spim.xaxis_values[int(len(self.spim.xaxis_values) / 2)]
        width = self.spim.xaxis_values.max() - self.spim.xaxis_values.min()
        self.slider_center.set_val(center)
        self.slider_width.set_val(width)
        self.image.set_data(self.spim.intensity_map(center, width))
        self.ax_image.set_title(self._get_integration_range(), pad=15)
        plt.draw()

    def _delta_minus(self, _):
        # Calculate the spectrum average increment in wavelength.
        increment = np.mean(np.diff(self.spim.xaxis_values))
        center = self.slider_center.val - increment
        width = self.slider_width.val
        self.slider_center.set_val(center)
        self.image.set_data(self.spim.intensity_map(center, width))

    def _delta_plus(self, _):
        # Calculate the spectrum average increment in wavelength.
        increment = np.mean(np.diff(self.spim.xaxis_values))
        center = self.slider_center.val + increment
        width = self.slider_width.val
        self.slider_center.set_val(center)
        self.image.set_data(self.spim.intensity_map(center, width))

    def _plot_image(self, _):
        """Plot the intensity image in a separate figure with same parameters as in the main figure."""
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(
            self.spim.intensity_map(self.slider_center.val, self.slider_width.val),
            cmap=self.image.get_cmap(),
            norm=self.image.norm,
            interpolation=self.image.get_interpolation(),
            origin='lower',
            extent=self.image.get_extent(),
            clim=self.image.get_clim()
        )
        ax.set_title("{} \n {}".format(self._get_integration_range(), self.spim.filename), pad=15, fontsize=8)
        ax.tick_params(axis='both', direction='in', length=4.3, width=1.2)
        # Line below is not necessary with the magic command ``%matplotlib``. 
        fig.canvas.manager.show()

    def _save_spect(self, _):
        """Save the last clicked pixel in the list of pixels saved ``indices_pixels_saved``."""
        if self.rect_selector.get_active():
            self.indices_rect_saved.append(self.indices_rect_drawn[-1])
        elif self.indices_pixels_clicked[-1] not in self.indices_pixels_saved:
            self.indices_pixels_saved.append(self.indices_pixels_clicked[-1])

    def _plot_spect(self, _):
        """
        Plot all saved spectra (i.e contained in ``indices_pixels_saved``).
        The yscale (linear or log) and ylim mode (autoscale or lock) is set by the radiobuttons.
        
        """
        x = []
        xunit = self.radiobutton_xunit.value_selected
        if xunit == 'nm':
            x = self.spim.xaxis_values
        elif xunit == 'eV':
            x = fct.nm_eV(self.spim.xaxis_values)

        # Rectangular selection.
        if self.rect_selector.get_active():
            if not self.indices_rect_saved:
                return

            style = 'mystyle' if 'mystyle' in plt.style.available else 'seaborn-whitegrid'
            with plt.style.context(style):
                fig, ax = plt.subplots(figsize=(7, 6))
                for idx in self.indices_rect_saved:
                    y = np.mean(self.spim.matrix[idx[1]:idx[3] + 1, idx[0]:idx[2] + 1, :], axis=(0, 1))
                    ax.plot(x, y, lw=1)
                    self.cursor = Cursor(ax, color='k', lw=1, ls='-')
                    ax.set_ylim(bottom=min(y))
                    if self.radiobutton_yscale.value_selected == 'Log':
                        ax.set_yscale('log')
                    plt.show()
                ax.set_xlim(left=min(x), right=max(x))
                ax.set_xlabel("Wavelength [nm]") if xunit == 'nm' else ax.set_xlabel("Energy [eV]")

            self._plot_image(None)
            ax = plt.gca()
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for i, idx in enumerate(self.indices_rect_saved):
                rectangle = mpl.patches.Rectangle(
                    (self.spim.x_range[idx[0]], self.spim.y_range[idx[1]]),
                    self.spim.x_range[idx[2]] - self.spim.x_range[idx[0]],
                    self.spim.y_range[idx[3]] - self.spim.y_range[idx[1]],
                    fill=False,
                    color=colors[i],
                    transform=ax.transData,
                )
                ax.add_patch(rectangle)

        # Normal pixel selection.
        else:
            if not self.indices_pixels_saved:
                return
            fig, axes = plt.subplots(
                len(self.indices_pixels_saved), 1, figsize=(14, 7), sharex=True, gridspec_kw=dict(hspace=0))
            axes = [axes] if len(self.indices_pixels_saved) == 1 else axes
            axes[-1].set_xlabel("Wavelength [nm]") if xunit == 'nm' else axes[-1].set_xlabel("Energy [eV]")

            ylim = self.radiobutton_ylim.value_selected
            yscale = self.radiobutton_yscale.value_selected
            ymin_all = np.min([np.min(self.spim.matrix[idx[0], idx[1], :]) for idx in self.indices_pixels_saved])
            ymax_all = np.max([np.max(self.spim.matrix[idx[0], idx[1], :]) for idx in self.indices_pixels_saved])
            # Plot spectra vertically and set ylim and yscale.
            for (ax, idx) in zip(axes, self.indices_pixels_saved):
                ax.plot(x, self.spim.matrix[idx[0], idx[1], :], lw=1, c='tab:red')
                if ylim == 'Autoscale':
                    if yscale == 'Linear':
                        ymin = np.min(self.spim.matrix[idx[0], idx[1], :])
                    elif yscale == 'Log':
                        ymin = max(0.1, np.min(self.spim.matrix[idx[0], idx[1], :]))
                    ymax = np.max(self.spim.matrix[idx[0], idx[1], :])
                elif ylim == 'Lock':
                    if yscale == 'Linear':
                        ymin = ymin_all
                    elif yscale == 'Log':
                        ymin = max(0.1, ymin_all)
                    ymax = ymax_all
                ax.set_yscale(self.ax_spectrum.get_yscale())
                ax.set_ylim(bottom=ymin, top=ymax)
                # Add the indexes of the spectrum.
                ax.text(1, 1, "Pixel [{:d}, {:d}]".format(idx[0], idx[1]),
                        ha='right', va='top', transform=ax.transAxes)
            # A reference to the Multicursor must be kept in the general context to work.
            # Either we declare it as a global object ">> global multicursor" or we add it to the attribute of ``self``.
            self.multicursor = MultiCursorAml(fig.canvas, axes, color='k', lw=1, ls='-.', horizOn=True, vertOn=True)
            # Line below is not necessary with the magic command ``%matplotlib``.
            fig.canvas.manager.show()

    def _reset_spect(self, _):
        """Reset the list of saved pixels."""
        self.indices_pixels_saved = []
        self.indices_rect_saved = []

    def _image_onclick(self, event):
        """
        This function connects the intensity image to the spectrum.
        The spectrum corresponding to the pixel clicked is displayed.
        
        """
        if event.inaxes != self.ax_image:
            return
        # Get the positional indices in ``matrix`` of the pixel clicked.
        # - ``idx1`` is searched with ``event.ydata``,
        # - ``idx2`` with ``event.xdata``.
        # This is because the spim is always formatted as if the scan was taken by successive horizontal lines
        # from bottom to top, by increasing y coordinates (see method ``Spim._shape3Dmatrix``).
        if self.spim.coordinate is None:
            idx1 = np.searchsorted(self.spim.y_range, event.ydata)
            idx2 = np.searchsorted(self.spim.x_range, event.xdata)
        else:
            idx1 = fct.find_nearest(self.spim.y_range, event.ydata)[0]
            idx2 = fct.find_nearest(self.spim.x_range, event.xdata)[0]
        self.indices_pixels_clicked.append([idx1, idx2])
        # Update the spectrum.
        self.spectrum.set_ydata(self.spim.matrix[idx1, idx2, :])
        self._apply_ylim(self)
        self.ax_spectrum.set_title(self._get_spectrum_location(), pad=15)
        plt.draw()

    def _apply_yscale(self, label):
        """Set the yscale of the axis. Special care is taken to manage ylim depending on
        the ``Autoscale`` or ``Lock`` mode."""
        ylim = self.radiobutton_ylim.value_selected
        if label == 'Linear':
            self.ax_spectrum.set_yscale(label)
            if ylim == 'Autoscale':
                self.ax_spectrum.set_ylim(np.min(self.spectrum.get_data()[1]), np.max(self.spectrum.get_data()[1]))
            elif ylim == 'Lock':
                self.ax_spectrum.set_ylim(min(np.min(self.spectrum.get_data()[1]), self.ax_spectrum.get_ybound()[0]))
        elif label == 'Log':
            if ylim == 'Autoscale':
                self.ax_spectrum.set_ylim(
                    max(0.1, np.min(self.spectrum.get_data()[1])), np.max(self.spectrum.get_data()[1]))
            elif ylim == 'Lock':
                self.ax_spectrum.set_ylim(bottom=max(0.1, self.ax_spectrum.get_ybound()[0]))
            self.ax_spectrum.set_yscale(label)
        plt.draw()

    def _apply_ylim(self, label):
        """Set the ylim mode of the axis. Special care is taken to manage ylim depending on
        the ``Linear`` or ``Log`` yscale."""
        # If the method is not called by the radiobutton we set ``label`` to the value of the radiobutton.
        if not isinstance(label, str):
            label = self.radiobutton_ylim.value_selected
        yscale = self.radiobutton_yscale.value_selected
        if label == 'Autoscale':
            if yscale == 'Linear':
                self.ax_spectrum.set_ylim(
                    bottom=np.min(self.spectrum.get_data()[1]), top=np.max(self.spectrum.get_data()[1]))
            elif yscale == 'Log':
                self.ax_spectrum.set_ylim(
                    bottom=max(0.1, np.min(self.spectrum.get_data()[1])), top=np.max(self.spectrum.get_data()[1]))
        elif label == 'Lock':
            pass
        plt.draw()

    def _set_xunit(self, label):
        if label == 'nm':
            x_nm = self.spim.xaxis_values
            self.spectrum.set_xdata(x_nm)
            self.ax_spectrum.set_xlim(np.min(x_nm), np.max(x_nm))
            self.ax_spectrum.set_xlabel('Wavelength [nm]')
            self.indicator_spectrum.unit_eV = False
            # Update markers positions.
            self._update_image(self)
        elif label == 'eV':
            x_eV = fct.nm_eV(self.spim.xaxis_values)
            self.spectrum.set_xdata(x_eV)
            self.ax_spectrum.set_xlim(np.min(x_eV), np.max(x_eV))
            self.ax_spectrum.set_xlabel('Energy [eV]')
            self.indicator_spectrum.unit_eV = True
            # Update markers positions.
            self._update_image(self)
        plt.draw()

    def _display_tools(self, label):
        if label == 'Indicator':
            self.indicator_spectrum.set_active(not self.indicator_spectrum.get_active())
            if not self.indicator_spectrum.get_active():
                self.indicator_spectrum._remove_artists()
                self.fig.canvas.draw_idle()
        elif label == 'Selector':
            self.rect_selector.set_active(not self.rect_selector.get_active())
            if self.rect_selector.get_active():
                plt.disconnect(self.cid_image)
                self.cursor_image.set_active(False)
            else:
                self.cid_image = self.fig.canvas.mpl_connect('button_release_event', self._image_onclick)
                self.cursor_image.set_active(True)

    def _rectselect_action(self, eclick, erelease):
        """``eclick`` and ``erelease`` are the press and release events."""
        if self.spim.coordinate is None:
            idx_00 = np.searchsorted(self.spim.x_range, eclick.xdata)
            idx_01 = np.searchsorted(self.spim.y_range, eclick.ydata)
            idx_10 = np.searchsorted(self.spim.x_range, erelease.xdata)
            idx_11 = np.searchsorted(self.spim.y_range, erelease.ydata)
        else:
            idx_00 = fct.find_nearest(self.spim.x_range, eclick.xdata)[0]
            idx_01 = fct.find_nearest(self.spim.y_range, eclick.ydata)[0]
            idx_10 = fct.find_nearest(self.spim.x_range, erelease.xdata)[0]
            idx_11 = fct.find_nearest(self.spim.y_range, erelease.ydata)[0]

        # Reorder the indices in order to have (idx_00, idx_01) be the coordinates of the bottom left corner
        # and (idx_10, idx_11) the coordinates of the top right corner.
        idx_00, idx_10 = min(idx_00, idx_10), max(idx_00, idx_10)
        idx_01, idx_11 = min(idx_01, idx_11), max(idx_01, idx_11)

        self.indices_rect_drawn.append([idx_00, idx_01, idx_10, idx_11])

        # Compute the average of all spectra contained inside the drawn rectangle
        mean_spectrum = np.mean(self.spim.matrix[idx_01:idx_11 + 1, idx_00:idx_10 + 1, :], axis=(0, 1))
        self.spectrum.set_ydata(mean_spectrum)
        self._apply_ylim(self)
        plt.draw()

    def _rectselect_persist(self, event):
        if self.rect_selector.active:
            self.rect_selector.update()


class Indicator(Cursor):
    """Create a cursor with ability to measure horizontal and vertical distances between two points."""

    def __init__(self, ax=None, unit_eV=False, stick_to_data=False, useblit=True, **cursorprops):
        """
        Parameters
        ----------
        unit_eV : whether the xaxis is in eV or nm.
    
        stick_to_data : whether to measure distances between actual data points or between exact cursor positions.
        
        """
        default_esthetic = {'color': 'tab:red', 'ls': '--', 'lw': 1}
        for key, value in default_esthetic.items():
            cursorprops[key] = cursorprops.pop(key, value)
        ax = plt.gca() if ax is None else ax

        Cursor.__init__(self, ax, useblit=useblit, **cursorprops)

        # Necessary. Otherwise I had the ``hline`` & ``vline`` extending the xlim & ylim indefinitely.
        self.ax.set_xlim(self.ax.get_xlim()[0], self.ax.get_xlim()[1])
        self.ax.set_ylim(self.ax.get_ylim()[0], self.ax.get_ylim()[1])

        self.stick_to_data = stick_to_data
        self.unit_eV = unit_eV

        self.xs = []
        self.ys = []
        self.count = 0

        self.first_click = True
        self.move = False

        self.reftext = None
        self.refarrow = None
        self.refvline = None
        self.refhline = None

        self.connect_event('button_release_event', self._onclick)
        self.connect_event('motion_notify_event', self._onthemove)
        plt.draw()

    def _onclick(self, event):
        if (event.inaxes != self.ax) or (not self.active):
            return
        if not self.canvas.widgetlock.available(self):
            return
        if self.first_click:
            self._remove_artists(reftext=True, refarrow=False, refvline=False, refhline=False)
            self._append_xy_data_clicked(event)
            self.count += 1
            self.first_click = False
            self.move = True
        else:
            self._append_xy_data_clicked(event)
            self._draw_arrow(x0=self.xs[self.count - 1], y0=self.ys[self.count - 1], x=self.xs[-1], y=self.ys[-1])
            self._remove_artists(reftext=False, refarrow=False, refvline=True, refhline=True)
            prop = dict(boxstyle='round', facecolor='gainsboro', alpha=1, lw=1.5)
            # left, bottom, width, height = self.ax.bbox.bounds
            # left, bottom, width, height = self.ax.get_position().bounds
            self.reftext = self.ax.text(
                1, 1, r'$\Delta$x = {} meV | {} nm'.format(*self._distance_x())
                      + '\n' + r'$\Delta$y = {}'.format(self._distance_y()),
                ha='right', va='top', transform=self.ax.transAxes, fontdict=dict(weight='bold'), bbox=prop)
            self.count += 1
            self.first_click = True
            self.move = False
            plt.draw()

    def _onthemove(self, event):
        if (event.inaxes != self.ax) or (not self.move) or (not self.active):
            return
        if not self.canvas.widgetlock.available(self):
            return
        self._draw_arrow(x0=self.xs[self.count - 1], y0=self.ys[self.count - 1], x=event.xdata, y=event.ydata)
        self._remove_artists(reftext=False, refarrow=False, refvline=True, refhline=True)
        self.refvline = self.ax.vlines(event.xdata, self.ax.get_ylim()[0], self.ax.get_ylim()[1],
                                       transform=self.ax.transData, ls=self.linev.get_ls(), lw=self.linev.get_lw(),
                                       color=self.linev.get_color())
        self.refhline = self.ax.hlines(event.ydata, self.ax.get_xlim()[0], self.ax.get_xlim()[1],
                                       transform=self.ax.transData, ls=self.lineh.get_ls(), lw=self.lineh.get_lw(),
                                       color=self.lineh.get_color())
        plt.draw()

    def _append_xy_data_clicked(self, event):
        """
        Two cases: 
        - no data are plotted on the axes --> the only ``lines`` present are the two from the cursor,
        thus we select the ``event.xdata & event.ydata`` that correspond to the precise position of the cursor;
        - data are plotted --> data corresponds to the first line in ``ax.lines``, we search for the nearest points
        in data from the cursor position.
        
        """
        if len(self.ax.lines) == 2 or not self.stick_to_data:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
        else:
            # Catch the last x & y data plotted on the axes.
            x_data = self.ax.lines[0].get_data()[0]
            y_data = self.ax.lines[0].get_data()[1]
            # Get the index and value of the nearest x from ``event.xdata``.
            x_clicked = fct.find_nearest(x_data, event.xdata)
            self.xs.append(x_clicked[1])
            self.ys.append(y_data[x_clicked[0]])

    def _draw_arrow(self, x0, y0, x, y):
        """Draw an arrow between specified points, and clean the previous drawn arrow."""
        self._remove_artists(reftext=False, refarrow=True, refvline=False, refhline=False)
        pos = self.ax.transData.transform([(x0, y0), (x, y)])
        pos = self.ax.transAxes.inverted().transform([(pos[0, 0], pos[0, 1]), (pos[1, 0], pos[1, 1])])
        self.refarrow = self.ax.arrow(pos[0, 0], pos[0, 1], pos[1, 0] - pos[0, 0], pos[1, 1] - pos[0, 1], fc='k',
                                      length_includes_head=True,
                                      width=0.002,
                                      transform=self.ax.transAxes
                                      )

    def _distance_x(self):
        """Compute the horizontal distance between the two last clicked positions - in nm and eV."""
        delta_x_eV, delta_x_nm = None, None
        if not self.unit_eV:
            delta_x_eV = np.abs(
                np.around(1239.84193 / self.xs[self.count] * 1000 - 1239.84193 / self.xs[self.count - 1] * 1000,
                          decimals=2))  # meV
            delta_x_nm = np.abs(np.around(self.xs[self.count] - self.xs[self.count - 1], decimals=2))
        elif self.unit_eV:
            delta_x_eV = np.abs(
                np.around(self.xs[self.count] * 1000 - self.xs[self.count - 1] * 1000, decimals=2))  # meV
            delta_x_nm = np.abs(
                np.around(1239.84193 / self.xs[self.count] - 1239.84193 / self.xs[self.count - 1], decimals=2))
        return [delta_x_eV, delta_x_nm]

    def _distance_y(self):
        """Compute the vertical distance between the two last clicked positions."""
        delta_y = np.abs(np.around(self.ys[self.count] - self.ys[self.count - 1], decimals=2))
        return delta_y

    def _remove_artists(self, reftext=True, refarrow=True, refvline=True, refhline=True):
        """Remove the artists with boolean true value."""
        if reftext:
            try:
                self.reftext.remove()
            except (ValueError, AttributeError):
                pass
        if refarrow:
            try:
                self.refarrow.remove()
            except (ValueError, AttributeError):
                pass
        if refvline:
            try:
                self.refvline.remove()
            except (ValueError, AttributeError):
                pass
        if refhline:
            try:
                self.refhline.remove()
            except (ValueError, AttributeError):
                pass
        plt.draw()


class FancyButton(Button):
    """This class is a simple wrapper for the ``matplotlib.widgets.Button`` with more control on the appearance."""

    indianred = dict(color='gainsboro', labelcolor='indianred', spinecolor='indianred', hovercolor='dimgrey')
    darkviolet = dict(color='gainsboro', labelcolor='darkviolet', spinecolor='darkviolet', hovercolor='dimgrey')
    darkviolet2 = dict(color='darkviolet', labelcolor='k', spinecolor='k', hovercolor='dimgrey')
    all_style = {'indianred': indianred, 'darkviolet': darkviolet, 'darkviolet2': darkviolet2}

    def __init__(self, ax, label, style='darkviolet', color=None, labelcolor=None, fontsize=12,
                 spine_lw=2, spinecolor=None, hovercolor=None):

        if style is None and None in [color, labelcolor, spinecolor, hovercolor]:
            raise ValueError("If ``style`` is None, all other color type arguments must be given.")

        dic_args = dict(color=color, labelcolor=labelcolor, spinecolor=spinecolor, hovercolor=hovercolor)
        esthetic = {}
        if style is not None:
            esthetic = copy.deepcopy(FancyButton.all_style[style])
            for key, value in dic_args.items():
                esthetic[key] = value if value is not None else esthetic[key]
        elif style is None:
            esthetic = dic_args

        Button.__init__(self, ax, label, color=esthetic['color'], hovercolor=esthetic['hovercolor'])
        # Label
        self.label.set_fontsize(fontsize)
        self.label.set_color(esthetic['labelcolor'])
        # Spines
        for spine in self.ax.spines.values():
            spine.set_lw(spine_lw)
            spine.set_color(esthetic['spinecolor'])


class FancySlider(Slider):
    """This class is a simple wrapper for the ``matplotlib.widgets.Slider`` with more control on the appearance."""

    indianred = dict(color='lightcoral', facecolor='gainsboro', labelcolor='k', spinecolor='indianred', barcolor='k')
    darkviolet = dict(color='darkviolet', facecolor='gainsboro', labelcolor='k', spinecolor='k', barcolor='k')
    all_style = {'indianred': indianred, 'darkviolet': darkviolet}

    def __init__(self, ax, label, valmin, valmax, style='darkviolet', color=None, facecolor=None, labelcolor=None,
                 fontsize=10, spine_lw=2, spinecolor=None, barcolor=None, **kwargs):

        if style is None and None in [color, facecolor, labelcolor, spinecolor, barcolor]:
            raise ValueError("If ``style`` is None, all other color type arguments must be given.")

        dic_args = dict(color=color, facecolor=facecolor, labelcolor=labelcolor, spinecolor=spinecolor,
                        barcolor=barcolor)
        esthetic = {}
        if style is not None:
            esthetic = copy.deepcopy(FancySlider.all_style[style])
            for key, value in dic_args.items():
                esthetic[key] = value if value is not None else esthetic[key]
        elif style is None:
            esthetic = dic_args

        Slider.__init__(self, ax, label, valmin, valmax, color=esthetic['color'], **kwargs)
        # Axes
        self.ax.set_facecolor(esthetic['facecolor'])
        # The line for valinit
        if self.orientation == 'horizontal':
            self.vline.set_color(esthetic['barcolor'])
        elif self.orientation == 'vertical':
            self.hline.set_color(esthetic['barcolor'])
        # Label
        self.label.set_fontsize(fontsize)
        self.label.set_color(esthetic['labelcolor'])
        # Spines
        for spine in self.ax.spines.values():
            spine.set_lw(spine_lw)
            spine.set_color(esthetic['spinecolor'])


class FancyRadioButtons(RadioButtons):
    """This class is a simple wrapper for the ``matplotlib.widgets.RadioButtons`` with more control on the
    appearance."""

    indianred = dict(facecolor='gainsboro', labelcolor='indianred', activecolor='indianred', spinecolor='indianred')
    darkviolet = dict(facecolor='gainsboro', labelcolor='darkviolet', activecolor='darkviolet', spinecolor='darkviolet')
    all_style = {'indianred': indianred, 'darkviolet': darkviolet}

    def __init__(self, ax, labels, active=0, style='darkviolet', facecolor=None, labelcolor=None, activecolor=None,
                 spinecolor=None, fontsize=10, spine_lw=2):

        if style is None and None in [facecolor, labelcolor, activecolor, spinecolor]:
            raise ValueError("If ``style`` is None, all other color type arguments must be given.")

        dic_args = dict(facecolor=facecolor, labelcolor=labelcolor, activecolor=activecolor, spinecolor=spinecolor)
        esthetic = {}
        if style is not None:
            esthetic = copy.deepcopy(FancyRadioButtons.all_style[style])
            for key, value in dic_args.items():
                esthetic[key] = value if value is not None else esthetic[key]
        elif style is None:
            esthetic = dic_args

        RadioButtons.__init__(self, ax, labels, active=active, activecolor=esthetic['activecolor'])
        # Axes
        self.ax.set_facecolor(esthetic['facecolor'])
        # Label
        for label in self.labels:
            label.set_fontsize(fontsize)
            label.set_color(esthetic['labelcolor'])
        # Circles
        for i, circle in enumerate(self.circles):
            circle.set_facecolor(esthetic['activecolor'] if i == active else esthetic['facecolor'])
        # Spines
        for spine in self.ax.spines.values():
            spine.set_lw(spine_lw)
            spine.set_color(esthetic['spinecolor'])


class FancyCheckButtons(CheckButtons):
    """This class is a simple wrapper for the ``matplotlib.widgets.CheckButtons`` with more control on the
    appearance."""

    indianred = dict(facecolor='gainsboro', labelcolor='indianred', spinecolor='indianred')
    darkviolet = dict(facecolor='gainsboro', labelcolor='darkviolet', spinecolor='darkviolet')
    all_style = {'indianred': indianred, 'darkviolet': darkviolet}

    def __init__(self, ax, labels, actives=None, style='darkviolet', facecolor=None, labelcolor=None, spinecolor=None,
                 fontsize=10, spine_lw=2):

        if style is None and None in [facecolor, labelcolor, spinecolor]:
            raise ValueError("If ``style`` is None, all other color type arguments must be given.")

        dic_args = dict(facecolor=facecolor, labelcolor=labelcolor, spinecolor=spinecolor)
        esthetic = {}
        if style is not None:
            esthetic = copy.deepcopy(FancyCheckButtons.all_style[style])
            for key, value in dic_args.items():
                esthetic[key] = value if value is not None else esthetic[key]
        elif style is None:
            esthetic = dic_args

        CheckButtons.__init__(self, ax, labels, actives=actives)
        # Axes
        self.ax.set_facecolor(esthetic['facecolor'])
        # Label
        for label in self.labels:
            label.set_fontsize(fontsize)
            label.set_color(esthetic['labelcolor'])
        # Rectangles
        for rectangle in self.rectangles:
            rectangle.set_facecolor(esthetic['facecolor'])
        # Spines
        for spine in self.ax.spines.values():
            spine.set_lw(spine_lw)
            spine.set_color(esthetic['spinecolor'])


class MultiCursorAml(MultiCursor):
    """This class is a simple wrapper for the ``matplotlib.widgets.Multicursor`` for clearing the Multicursor when
    outside the axes."""

    def __init__(self, canvas, axes, useblit=True, horizOn=False, vertOn=True, clear_out_of_axes=True, **lineprops):
        MultiCursor.__init__(self, canvas, axes, useblit=useblit, horizOn=horizOn, vertOn=vertOn, **lineprops)
        if clear_out_of_axes:
            canvas.mpl_connect('motion_notify_event', self._clear_cursor)

    def _clear_cursor(self, event):
        if event.inaxes not in self.axes:
            self.visible = False
            plt.draw()
        else:
            self.visible = True
            plt.draw()


class Datum:
    colorin = mpl.colors.to_rgba("tab:red")
    colorout = mpl.colors.to_rgba("tab:blue")

    def __init__(self, x, y, include=False):
        self.x = x
        self.y = y
        if include:
            self.color = self.colorin
        else:
            self.color = self.colorout


from scipy.optimize import curve_fit


class LassoSelectorFit:
    def __init__(self, ax, x, y, errors=None, splitting=True):
        """
        Parameters
        ----------
        splitting : True if the calculation is on zeeman splitting, False if on full peak position
        """
        self.axes = ax
        self.canvas = ax.figure.canvas
        data = [Datum(x, y) for (x, y) in zip(x, y)]
        self.data = data
        self.Nxy = len(data)
        self.splitting = splitting

        self.facecolors = [d.color for d in data]
        self.xys = [(d.x, d.y) for d in data]
        self.xs = [d.x for d in data]
        self.ys = [d.y for d in data]
        
        self.fit = None
        self.g = None
        
        self.collection = RegularPolyCollection(4, rotation=np.pi/4, sizes=(30,), facecolors=self.facecolors,
                                                offsets=self.xys, transOffset=ax.transData)
        ax.add_collection(self.collection)
        if errors is not None:
            self.axes.errorbar(self.xs, self.ys, yerr=np.asarray(errors)/2, ls='none', marker='s',
                               ms=0, ecolor='tab:red', elinewidth=1, capsize=3)
        ax.autoscale(True)

        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)
        
    def onpress(self, event):
        if self.canvas.widgetlock.locked():
            return
        if event.inaxes != self.axes:
            return
        self.lasso = Lasso(event.inaxes, (event.xdata, event.ydata), self.callback)
        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)

    def callback(self, verts):
        if self.g:
            self.g.remove()
        if self.fit:
            self.fit.remove()
        facecolors = self.collection.get_facecolors()
        p = path.Path(verts)
        ind = p.contains_points(self.xys)
        for i in range(len(self.xys)):
            if ind[i]:
                facecolors[i] = Datum.colorin
            else:
                facecolors[i] = Datum.colorout
        self.collection.set_facecolors(facecolors)
        
        self.innercall(ind)

        self.canvas.draw_idle()
        self.canvas.widgetlock.release(self.lasso)
        del self.lasso
        
    def innercall(self, ind):
        xs = np.asarray(self.xs)[ind]
        ys = np.asarray(self.ys)[ind]
        # Fit with linear function
        popt, pcov = curve_fit(fct.linear, xs, ys)
        xfit = np.linspace(min(self.xs), max(self.xs), 1000)
        self.fit, = self.axes.plot(xfit, fct.linear(xfit, *popt), '--', color='tab:orange')
        # Extract and display the abs value of valley magnetic coupling g factor : this has to go away to have a more general use
        bohr_magneton = 5.788381e-5  # eV.T-1
        if self.splitting:
            g_factor = round(popt[0]*1e-3 / bohr_magneton, 2)
        else:
            g_factor = round(popt[0] / bohr_magneton, 2) * 2
        self.g = self.axes.text(0.5, 1.03, "|g| = {}".format(abs(g_factor)), ha='center', transform=self.axes.transAxes,
                                   size=14, fontweight='heavy')
        self.axes.relim()

# Cursor that works for both directions
class CursorDragHV(AxesWidget):
    """
    A modified version of matplotlib cursor that spans the axes and moves when dragged.

    For the cursor to remain responsive you must keep a reference to it.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        The `~.axes.Axes` to attach the cursor to.
    horizOn : bool, default: True
        Whether to draw the horizontal line.
    vertOn : bool, default: True
        Whether to draw the vertical line.
    useblit : bool, default: False
        Use blitting for faster drawing if supported by the backend.

    Other Parameters
    ----------------
    **lineprops
        `.Line2D` properties that control the appearance of the lines.
        See also `~.Axes.axhline`.

    """

    def __init__(self, ax, horizOn=True, vertOn=True, useblit=False, **lineprops):
        AxesWidget.__init__(self, ax)
        
        self.connect_event('button_press_event', self.onpress)
        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('button_release_event', self.release)
        self.connect_event('draw_event', self.clear)

        self.visible = True
        self.horizOn = horizOn
        self.vertOn = vertOn
        self.useblit = useblit and self.canvas.supports_blit
        self.picked = False

        if self.useblit:
            lineprops['animated'] = True
            
        self.lineh = ax.axhline(np.mean((ax.get_ybound()[0], ax.get_ybound()[1])), visible=horizOn, **lineprops)
        self.linev = ax.axvline(np.mean((ax.get_xbound()[0], ax.get_xbound()[1])), visible=vertOn, **lineprops)

        self.background = None
        self.needclear = False
        

    def clear(self, event):
        """Internal event handler to clear the cursor."""
        if self.ignore(event):
            return
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.linev.set_visible(True)
        self.lineh.set_visible(True)
        
    def onpress(self, event):
        if event.inaxes != self.ax:
            return
        if str(event.button) != 'MouseButton.LEFT':
            return
        self.picked = True
    
    def release(self, event):
        if event.inaxes != self.ax:
            return
        if str(event.button) != 'MouseButton.LEFT':
            return
        self.picked = False

    def onmove(self, event):
        """Internal event handler to draw the cursor when the mouse moves."""
        if self.ignore(event):
            return
        if not self.canvas.widgetlock.available(self):
            return
        if event.inaxes != self.ax:
            if self.needclear:
                self.canvas.draw()
                self.needclear = False
            return
        if not self.picked:
            return
        self.needclear = True
        if not self.visible:
            return
        
        self.linev.set_xdata((event.xdata, event.xdata))
        self.lineh.set_ydata((event.ydata, event.ydata))
        
        
        self.linev.set_visible(self.visible and self.vertOn)
        self.lineh.set_visible(self.visible and self.horizOn)
        
        self._update()

    def _update(self):
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.linev)
            self.ax.draw_artist(self.lineh)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()
        return False

# optimized for vertical cursor
class CursorDrag(AxesWidget):
    """
    A modified version of matplotlib cursor that spans the axes and moves when dragged.

    For the cursor to remain responsive you must keep a reference to it.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        The `~.axes.Axes` to attach the cursor to.
    useblit : bool, default: False
        Use blitting for faster drawing if supported by the backend.

    Other Parameters
    ----------------
    **lineprops
        `.Line2D` properties that control the appearance of the lines.
        See also `~.Axes.axhline`.

    """

    def __init__(self, ax, useblit=False, **lineprops):
        AxesWidget.__init__(self, ax)
        
        self.connect_event('button_press_event', self.onpress)
        self.connect_event('button_release_event', self.release)
        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('draw_event', self.clear)

        self.visible = True
        self.useblit = useblit and self.canvas.supports_blit
        self.picked = False

        if self.useblit:
            lineprops['animated'] = True
            
        if len(ax.get_lines()) == 0:
            init_pos = np.mean((ax.get_xbound()[0], ax.get_xbound()[1]))
        else:
            init_pos = ax.get_lines()[0].get_xdata()[0]
        self.linev = ax.axvline(init_pos, visible=True, **lineprops)

        self.background = None
        self.needclear = False
        

    def clear(self, event):
        """Internal event handler to clear the cursor."""
        if self.ignore(event):
            return
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        #self.linev.set_visible(False)
        
    def onpress(self, event):
        if event.inaxes != self.ax:
            return
        if str(event.button) != 'MouseButton.LEFT':
            return
        self.picked = True
    
    def release(self, event):
        if event.inaxes != self.ax:
            return
        if str(event.button) != 'MouseButton.LEFT':
            return
        self.picked = False

    def onmove(self, event):
        """Internal event handler to draw the cursor when the mouse moves."""
        if self.ignore(event):
            return
        if not self.canvas.widgetlock.available(self):
            return
        if event.inaxes != self.ax:
            if self.needclear:
                self.canvas.draw()
                self.needclear = False
            return
        if not self.picked:
            return
        self.needclear = True
        if not self.visible:
            return
        
        self.linev.set_xdata((event.xdata, event.xdata))        
        self.linev.set_visible(self.visible)
        
        self._update()

    def _update(self):
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.linev)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()
        return False

# Crusor that snap to data - in progress
class CursorSnapData(AxesWidget):
    """
    A modified version of matplotlib cursor that spans the axes and moves when dragged.

    For the cursor to remain responsive you must keep a reference to it.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        The `~.axes.Axes` to attach the cursor to.
    horizOn : bool, default: True
        Whether to draw the horizontal line.
    vertOn : bool, default: True
        Whether to draw the vertical line.
    useblit : bool, default: False
        Use blitting for faster drawing if supported by the backend.

    Other Parameters
    ----------------
    **lineprops
        `.Line2D` properties that control the appearance of the lines.
        See also `~.Axes.axhline`.

    """

    def __init__(self, ax, horizOn=True, vertOn=True, useblit=False, snap_to_data=False, **lineprops):
        AxesWidget.__init__(self, ax)
        
        self.connect_event('button_press_event', self.onpress)
        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('button_release_event', self.release)
        self.connect_event('draw_event', self.clear)
        #
        """
        if snap_to_data:
            self.x, self.y = ax.get_lines()[-1].get_data()
        else:
            self.x, self.y = None, None

        self.old_x = None
        self.old_y = None
        """
        #    
        self.visible = True
        self.horizOn = horizOn
        self.vertOn = vertOn
        self.useblit = useblit and self.canvas.supports_blit
        self.picked = False

        if self.useblit:
            lineprops['animated'] = True
            
        self.lineh = ax.axhline(np.mean((ax.get_ybound()[0], ax.get_ybound()[1])), visible=horizOn, **lineprops)
        self.linev = ax.axvline(np.mean((ax.get_xbound()[0], ax.get_xbound()[1])), visible=vertOn, **lineprops)

        self.background = None
        self.needclear = False
        

    def clear(self, event):
        """Internal event handler to clear the cursor."""
        if self.ignore(event):
            return
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.linev.set_visible(False)
        self.lineh.set_visible(False)
        
    def onpress(self, event):
        if event.inaxes != self.ax:
            return
        if str(event.button) != 'MouseButton.LEFT':
            return
        self.picked = True
    
    def release(self, event):
        if event.inaxes != self.ax:
            return
        if str(event.button) != 'MouseButton.LEFT':
            return
        self.picked = False

    def onmove(self, event):
        """Internal event handler to draw the cursor when the mouse moves."""
        if self.ignore(event):
            return
        if not self.canvas.widgetlock.available(self):
            return
        if event.inaxes != self.ax:
            if self.needclear:
                self.canvas.draw()
                self.needclear = False
            return
        if not self.picked:
            return
        self.needclear = True
        if not self.visible:
            return
        
        #
        """
        new_x = np.searchsorted(self.x, event.xdata)
        new_y = np.searchsorted(self.y, event.ydata)
        if new_x != self.old_x:
            self.linev.set_xdata((new_x, new_x))
        if new_y != self.old_y:
            self.lineh.set_ydata((new_y, new_y))
        """
        #
        
        self.linev.set_xdata((event.xdata, event.xdata))
        self.lineh.set_ydata((event.ydata, event.ydata))
        
        
        self.linev.set_visible(self.visible and self.vertOn)
        self.lineh.set_visible(self.visible and self.horizOn)
        
        #if new_x != self.old_x or new_y != self.old_y:
        self._update()

        #
        """
        self.old_x = new_x
        self.old_y = new_y
        """
        #

    def _update(self):
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.linev)
            self.ax.draw_artist(self.lineh)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()
        return False
