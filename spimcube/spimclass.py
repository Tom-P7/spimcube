import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import (Slider, Button, RadioButtons, CheckButtons, Cursor, MultiCursor, RectangleSelector)
import re
import colorcet as cc
import copy
import statistics as stat
import despike
import spimcube.functions as fct



class Spim:
    """This class creates and formats the data cube (SPIM) to be explored and used. 
    
    The data cube is extracted from rawdata files by using one of the 'initialization' methods.
    Main attributes are the shaped data cube ``matrix`` and the list of wavelengths ``tab_lambda``.
    The created object contains also all the metadata declared as attributes.
    A method ``intensity_map`` is defined to compute intensity integrated image from the data cube.

    Example:
    -------
    >> import spimcube.spimclass as sc
    OR an alternative way is to import directly the objects in the global namespace:
    >> from spimcube.spimclass import (Spim, SpimInterface)
    
    >> path     = "/my/path/to/my/data/"
    >> filename = "SPI_C27_004K_370uW_5x1s_30x30um_1800gr_slit20_310nm_221018a"
    
    >> spim = sc.Spim(path, filename)
    >> spim.initialization()
    >> spim.define_space_range(area=(0, 30, 0, 30))
    
    # Alternatively, a coordinate file can be used:
    >> spim.define_space_range(coordinate_file="/my/path/filename.txt") # Which will additionaly plot a grid of the scanned pixel positions

    # Then, the spim can be explored using ``SpimInterface`` explorator.

    >> si = sc.SpimInterface(spim)

    -------

    """
    def __init__(self, path, filename, scan_direction='hl', clean_spike=False):

        """
        Parameters
        ----------
        scan_direction : one of ('hl', 'hr', 'vt', 'vb')
                         Indicate the directions of scanning.
                         Example: 'hl' means scanning by successive horizontal lines from left to right.
                                  'vt' means scanning by successive vertical lines from top to bottom.
                         Default is 'hl'.
        clean_spike : bool (default : False)
                      If True, the method ``intensity_map`` returns an image after cleaning the spikes with
                      ``despike.clean``.
                         
        """
        self.path = path
        self.filename = filename
        self.scan_direction = scan_direction
        self.clean_spike = clean_spike
        
        # Data attributes
        self.matrix = None
        self.tab_lambda = None
        
        # Attributes of the initialization
        self.xstep_value, self.ystep_value = [None]*2
        self.xstep_number, self.ystep_number = [None]*2
        self.CCD_nb_pixel = None
        self.coordinate = None
        self.x_range, self.y_range = [None]*2
        self.xpmin, self.xpmax, self.ypmin, self.ypmax = [None]*4
        
        # Attributes of the intensity map: integration between min & max wavelength.
        self.int_lambda_min = None
        self.int_lambda_max = None
        
    def initialization(self):
        """Use this function to initialize the SPIM with datafile from L2C - Labview."""
        # Create the complete filenames with extension.
        complete_path = self.path + self.filename
        data_file   = complete_path + ".dat"
        lambda_file = complete_path + ".lambda"
        init_file   = complete_path + ".ini"
        
        # Load 'data' file, extract the first three values and create a list without them.
        raw_data = np.fromfile(data_file, dtype='>i4')
        
        self.xstep_number = raw_data[0]
        self.ystep_number = raw_data[1]
        self.CCD_nb_pixel = raw_data[2]

        data = raw_data[3:]

        # Create a 3D matrix in desired dimensions.
        self.matrix = data.reshape(self.xstep_number, self.ystep_number, self.CCD_nb_pixel)
        ## Transform the matrix depending of the scan direction
        if self.scan_direction == 'hl':
            pass
        elif self.scan_direction == 'hr':
            self.matrix = np.flip(self.matrix, axis=1)
        elif self.scan_direction == 'vt':
            self.matrix = np.flip(self.matrix, axis=1).transpose(1, 0, 2)
        elif self.scan_direction == 'vb':
            self.matrix = self.matrix.transpose(1, 0, 2)

        # Load Lambda file in float.
        self.tab_lambda = np.loadtxt(
            lambda_file, encoding='latin1', converters={0: lambda s: s.replace("," , ".")}
        )

        # Fetch metadata with regex.
        self.xstep_value  = fct.get_value("PasX", init_file)
        self.ystep_value  = fct.get_value("PasY", init_file)

    def initialization_winspec32(self, xstep_number=None, ystep_number=None, xstep_value=None, ystep_value=None,
                                 CCD_nb_pixel=1340):
        """Use this function to initialize the SPIM with datafile from LNCMI - Winspec32."""
        complete_path = self.path + self.filename + ".txt"
        self.CCD_nb_pixel = CCD_nb_pixel
 
        # Load data.
        column_1, column_2 = np.loadtxt(complete_path, unpack=True)

        # Fetch metadata with regex.
        if xstep_number is None or ystep_number is None:
            pattern = r"(\w*)x(\w*)" + 'steps' # Specific to how we write the file name.
            values = re.findall(pattern, self.filename)
            if values != []:
                self.xstep_number = int(values[0][0])
                self.ystep_number = int(values[0][1])
            else:
                raise AttributeError("``xstep_number`` and ``ystep_number`` not found in filename, must be provided.")
        else:
            self.xstep_number = xstep_number
            self.ystep_number = ystep_number

        if xstep_value is None or ystep_value is None:
            pattern = r"(\w*)x(\w*)" + 'mum'   # Specific to how we write the file name.
            values = re.findall(pattern, self.filename)
            if values != []:
                self.xstep_value = int(values[0][0])
                self.ystep_value = int(values[0][1])
            else:
                raise AttributeError("``xstep_value`` and ``ystep_value`` not found in filename, must be provided.")
        else:
            self.xstep_value = xstep_value
            self.ystep_value = ystep_value
            
        # Create a 3D matrix in desired dimensions.
        self.matrix = column_2.reshape(self.xstep_number, self.ystep_number, self.CCD_nb_pixel)
        ## Transform the matrix depending of the scan direction
        if self.scan_direction == 'hl':
            pass
        elif self.scan_direction == 'hr':
            self.matrix = np.flip(self.matrix, axis=1)
        elif self.scan_direction == 'vt':
            self.matrix = np.flip(self.matrix, axis=1).transpose(1, 0, 2)
        elif self.scan_direction == 'vb':
            self.matrix = self.matrix.transpose(1, 0, 2)
        
        # Create the tab of wavelengths.
        self.tab_lambda = column_1[0:self.CCD_nb_pixel]
    
    def define_space_range(self, area=None, coordinate_file=None):
        """Define the size of the area in micrometer unit of the intensity image and generate the x & y scanned positions.
        
        Parameters
        ----------
        area : tuple of 4 floats: ``(xmin, xmax, ymin, ymax)`` representing the limits of the intensity image in micrometers.
               If ``None``, then the full area defined by ``x_range`` and ``y_range`` is supposed.
               
        coordinate_file : the full filename (with the path) containing the scanned positions.
                          If provided, take  precedence on ``area``.
        """
        if coordinate_file is None:
            self.coordinate = None
            # Generate X & Y 1D arrays of scanned positions.
            self.x_range = np.linspace(self.xstep_value, self.xstep_value*self.xstep_number, self.xstep_number,
                                       dtype='>f4') # Builds 1D array of X positions
            self.y_range = np.linspace(self.ystep_value, self.ystep_value*self.ystep_number, self.ystep_number,
                                       dtype='>f4') # and Y positions in µm.
            if area is None:
                area = (0, self.x_range.max(), 0, self.y_range.max())
            space_area = area # Define plot area in µm unit: (xmin, xmax, ymin, ymax).
            pixel_area = []   # Define plot area in pixel unit.

            for i, parameter in enumerate(space_area):
                pixel_number = fct.find_nearest(self.x_range if i<2 else self.y_range, parameter)
                pixel_area.append(pixel_number)

            self.xpmin, self.xpmax, self.ypmin, self.ypmax = pixel_area
            
        else:
            # Get the matrix of coordinates.
            coordinate = np.loadtxt(coordinate_file)
            coordinate = coordinate.reshape(self.xstep_number, self.ystep_number, 2)
            self.coordinate = coordinate
            # Generate X & Y 1D arrays of scanned pixels.
            self.x_range = np.arange(0, self.xstep_number, 1)
            self.y_range = np.arange(0, self.ystep_number, 1)
            # Define plot area in pixel.
            self.xpmin, self.xpmax = [0, 0], [len(self.x_range)-1, self.x_range[-1]]
            self.ypmin, self.ypmax = [0, 0], [len(self.y_range)-1, self.y_range[-1]]
        
    def intensity_map(self, center=None, width=None, clean_spike=self.clean_spike):
        """Builds the intensity image data in the desired spectral range.
        
        Generate a 2D array contening intensity of the PL integrated over the chosen range in wavelength.
        Takes ``center`` and ``width`` values for determining the wavelength range.
        Note: in the present form, the intensity is normalized to the maximum into the range considered.
        
        """
        center = self.tab_lambda[int(len(self.tab_lambda)/2)] if center is None else center
        width  = 1/10 * (self.tab_lambda.max()-self.tab_lambda.min()) if width is None else width
        int_lambda_min = center - width/2
        int_lambda_max = center + width/2
        #  The function ``find_nearest`` return the pixel number and the corresponding value in nm.
        self.int_lambda_min = fct.find_nearest(self.tab_lambda, int_lambda_min)
        self.int_lambda_max = fct.find_nearest(self.tab_lambda, int_lambda_max)
        # Integrate intensity over the chosen range of wavelength.
        image_data = np.sum(
            self.matrix[self.ypmin[0]:self.ypmax[0]+1, self.xpmin[0]:self.xpmax[0]+1,
                        self.int_lambda_min[0]:self.int_lambda_max[0]+1],
            axis=2,
        )
        if clean_spike:
            # Clean the image from the spikes.
            image_data = despike.clean(image_data)
        return image_data / np.max(image_data)


class SpimInterface:
    """This class allows the exploration of a SPIM datacube.

    A rich interface with multiple widgets make the exploration easier, with the possibility to
    extract some spectra and filtered image in separate windows.

    Example:
    -------
    # One first has to create an object ```Spim`` to be explored.
    
    >> import spimcube.spimclass as sc
    OR an alternative way is to import directly the objects in the global namespace:
    >> from spimcube.spimclass import (Spim, SpimInterface)

    >> path     = "/my/path/to/my/data/"
    >> filename = "SPI_C27_004K_370uW_5x1s_30x30um_1800gr_slit20_310nm_221018a"
    
    >> spim = sc.Spim(path, filename)
    >> spim.initialization()
    >> spim.define_space_range(area=(0, 30, 0, 30))

    ## Alternatively, a coordinate file can be used:
    >> spim.define_space_range(coordinate_file="/my/path/coordinates_filename.txt") # Which will additionaly plot a grid of the scanned pixel positions

    # Then, the spim can be explored using ``SpimInterface`` explorator.

    >> si = sc.SpimInterface(spim)

    -------
    
    """
    def __init__(self, spim, lambda_init=None, fig_fc='dimgrey', spec_fc='whitesmoke'):
        """
        Parameters
        ----------
        spim : the ``Spim`` object to be explored.
        
        lambda_init : Initial central wavelength to compute the intensity image.
                      If out of bound: set to the closest value.
                      If ``None``: most common intense pixel of the SPIM. If more than one are most common,\
                      default to most common intense pixel of the first spectrum.
        
        """
        if lambda_init is None:
            try:
                lambda_init = spim.tab_lambda[stat.mode(np.argmax(spim.matrix, axis=2).ravel())]
            except stat.StatisticsError:
                lambda_init = spim.tab_lambda[np.argmax(spim.matrix[0, 0, :])]
        # ``spim`` is given as attribute in order to use it in the class methods that define widgets action.
        self.spim = spim
        
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
        self.ax_spectrum  = self.fig.add_axes([l1, b0, w0, h0], facecolor=spec_fc)
        for spine in self.ax_spectrum.spines.values():
            spine.set_lw(1.7)
            spine.set_color('k')#'dimgrey'
        self.ax_image     = self.fig.add_axes([l2, b0, w0, h0])
        for spine in self.ax_image.spines.values():
            spine.set_lw(1.7)
            spine.set_color('k')
        self.ax_colorbar  = self.fig.add_axes([0.915, b0, 0.02, h0])
        
        # Widgets axes positions - not attributes
        widget_color = 'lightgoldenrodyellow'
        ## - Slider -
        ax__slider_center   = self.fig.add_axes([l2, b2, w3, h3], facecolor=widget_color)
        ax__slider_width    = self.fig.add_axes([l2, b4, w3, h3], facecolor=widget_color)
        ax__slider_clim_min = self.fig.add_axes([0.97, b0+vsp1, 0.015, h0/2-vsp1-vsp2/2], fc=widget_color)
        ax__slider_clim_max = self.fig.add_axes([0.97, b0+h0/2+vsp2/2, 0.015, h0/2-vsp1-vsp2/2], fc=widget_color)
        ## - Button -
        ax__button_reset_sliders = self.fig.add_axes([l4+w1+hsp1, b4, w1, h3])
        ax__button_full_range    = self.fig.add_axes([l4, b4, w1, h3])
        ax__button_delta_minus   = self.fig.add_axes([l4, b2, (w1-0.003)/2, h3])
        ax__button_delta_plus    = self.fig.add_axes([l4+(w1+0.003)/2, b2, (w1-0.003)/2, h3])
        ax__button_plot_image    = self.fig.add_axes([l4+w1+hsp1, b2, w1, h3])
        ax__button_save_spect    = self.fig.add_axes([l1, b1, w1, h1])
        ax__button_plot_spect    = self.fig.add_axes([l1+w1+hsp1, b1, w1, h1])
        ax__button_reset_spect   = self.fig.add_axes([l1+2*w1+2*hsp1, b1, w1, h1])
        ## - RadioButtons -
        ax__radiobutton_yscale  = self.fig.add_axes([l3+w2+hsp1, b3, w2, h2], facecolor=widget_color)
        ax__radiobutton_ylim    = self.fig.add_axes([l3+w2+hsp1, b2, w2, h2], facecolor=widget_color)
        ax__radiobutton_xunit   = self.fig.add_axes([l3, b3, w2, h2], facecolor=widget_color)
        ## - Checkbutton -
        ax__checkbutton_measure = self.fig.add_axes([l3, b2, w2, h2], facecolor=widget_color)
        
        # Widgets definitions - attributes
        ## - Slider -
        self.slider_center = FancySlider(
            ax__slider_center, 'Center\n[nm]', spim.tab_lambda.min()+0.01, spim.tab_lambda.max(), valinit=lambda_init)
        self.slider_width  = FancySlider(
            ax__slider_width, 'Width\n[nm]', 0.02, spim.tab_lambda.max()-spim.tab_lambda.min(), valinit=3)
        self.slider_clim_min = FancySlider(
            ax__slider_clim_min, 'Min', 0, 1, valinit=0, orientation='vertical')
        self.slider_clim_max = FancySlider(
            ax__slider_clim_max, 'Max', 0, 1, valinit=1, orientation='vertical', slidermin=self.slider_clim_min)
        self.slider_clim_max.label.set_position((0.5, 1.06))
        self.slider_clim_max.valtext.set_position((0.5, 1.06))
        self.slider_clim_min.label.set_position((0.5, -0.15))
        ### Connection to callback functions.
        self.slider_center.on_changed(self.update_image)
        self.slider_width.on_changed(self.update_image)
        self.slider_clim_min.on_changed(self.set_clim)
        self.slider_clim_max.on_changed(self.set_clim)
        ## - Button -
        #prop = dict(color=widget_color, hovercolor='0.975')
        prop = {}
        self.button_reset_sliders = FancyButton(ax__button_reset_sliders, 'Reset', **prop)
        self.button_full_range  = FancyButton(ax__button_full_range, 'Full', **prop)
        self.button_delta_plus  = FancyButton(ax__button_delta_plus, '+', **prop)
        self.button_delta_minus = FancyButton(ax__button_delta_minus, '-', **prop)
        self.button_plot_image  = FancyButton(ax__button_plot_image, 'Plot', **prop)
        self.button_save_spect  = FancyButton(ax__button_save_spect, 'Save', **prop)
        self.button_plot_spect  = FancyButton(ax__button_plot_spect, 'Plot', **prop)
        self.button_reset_spect = FancyButton(ax__button_reset_spect, 'Reset', **prop)
        ### Connection to callback functions.
        self.button_reset_sliders.on_clicked(self.reset_sliders)
        self.button_full_range.on_clicked(self.set_full_range)
        self.button_delta_minus.on_clicked(self.delta_minus)
        self.button_delta_plus.on_clicked(self.delta_plus)
        self.button_plot_image.on_clicked(self.plot_image)
        self.button_save_spect.on_clicked(self.save_spect)
        self.button_plot_spect.on_clicked(self.plot_spect)
        self.button_reset_spect.on_clicked(self.reset_spect)
        ## - RadioButtons -
        self.radiobutton_yscale  = FancyRadioButtons(ax__radiobutton_yscale, ('Linear', 'Log'))
        self.radiobutton_ylim    = FancyRadioButtons(ax__radiobutton_ylim , ('Autoscale', 'Lock'))
        self.radiobutton_xunit   = FancyRadioButtons(ax__radiobutton_xunit , ('nm', 'eV'))
        ### Connection to callback functions.
        self.radiobutton_yscale.on_clicked(self.set_yscale)
        self.radiobutton_ylim.on_clicked(self.set_ylim)
        self.radiobutton_xunit.on_clicked(self.set_xunit)
        ## - Checkbuttons -
        self.checkbutton_measure = FancyCheckButtons(ax__checkbutton_measure, ['Indicator', 'Selector'])
        ### Connection to callback functions.
        self.checkbutton_measure.on_clicked(self.display_tools)
        ## - Cursor -
        self.cursor_image = Cursor(self.ax_image, useblit=True, color='w', lw=0.5)
        ## - Indicator -
        self.indicator_spectrum = Indicator(self.ax_spectrum, color='k', lw=0.5, ls='-.', stick_to_data=False)
        self.indicator_spectrum.set_active(False)
        ## - RectangleSelector -
        rectprops = dict(facecolor='white', edgecolor ='green', alpha=0.7, fill=True)
        self.rect_selector = RectangleSelector(self.ax_image, self.rectselect_action, drawtype='box', useblit=True,
                                               button=[1,3], spancoords='data', interactive=True, rectprops=rectprops)
        self.rect_selector.set_active(False)
        self.fig.canvas.mpl_connect('draw_event', self.rectselect_persist)
        
        # ---------- --- ---------- ------- ----- --- ---------
        # Attributes for connection between image and spectrum.
        # ---------- --- ---------- ------- ----- --- ---------
        self.indices_pixels_clicked = [[0, 0]] # The spectrum [0, 0] is displayed at the beginning.
        self.indices_pixels_saved   = []
        self.cid_image = self.fig.canvas.mpl_connect('button_release_event', self.image_onclick)

        # ---- --- ---- -- ----- ----------- -- ---------
        # Plot the grid of pixel coordinates if provided.
        # ---- --- ---- -- ----- ----------- -- ---------
        if spim.coordinate is not None:
            cxy = spim.coordinate.reshape((spim.coordinate.shape[0]*spim.coordinate.shape[1], 2))
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
            extent = (spim.xpmin[1]-spim.xstep_value, spim.xpmax[1], spim.ypmin[1]-spim.ystep_value, spim.ypmax[1])
        else:
            extent = (spim.xpmin[1]-0.5, spim.xpmax[1]+0.5, spim.ypmin[1]-0.5, spim.ypmax[1]+0.5)
        # Plot the intensity image.
        self.image = self.ax_image.imshow(
            image_data,
            cmap='inferno',
            norm=norm,
            interpolation=None,
            origin='lower',
            extent=extent,
            clim=(0, 1),
            aspect='auto', # Image will fit the axes limits but pixels will not be square. Change to ``equal`` for square pixels.
        )
        #self.ax_image.grid(lw=0.2, color='k') # Not pleasing when zooming, lines superimpose with pixel borders
        self.ax_image.set_title(self.get_integration_range(), pad=15)
        self.ax_image.set_xlabel("x [µm]")
        self.ax_image.set_ylabel("y [µm]")
        # Set ticks and their properties.
        self.ax_image.tick_params(axis='both', direction='in', length=4.3, width=1.2)
        self.colorbar = plt.colorbar(self.image, cax=self.ax_colorbar)
        self.ax_colorbar.tick_params(direction='inout')

        
        # --- ----------- ---        
        # --- PL SPECTRUM ---
        # --- ----------- ---
        self.spectrum, = self.ax_spectrum.plot(spim.tab_lambda, spim.matrix[0, 0, :], lw=1, c='darkviolet')
        self.ax_spectrum.set_xlim(spim.tab_lambda.min(), spim.tab_lambda.max())
        self.ax_spectrum.set_ylim(np.min(spim.matrix[0, 0, :]), np.max(spim.matrix[0, 0, :]))
        # Define vertical marker lines for the spectrum - attributes.
        prop1 = dict(ls='-', color='k', lw=0.7)
        prop2 = dict(ls='--', color='k', lw=0.7)
        self.marker_center = self.ax_spectrum.axvline(x=lambda_init, ymin=0, ymax=1, **prop1)
        self.marker_left   = self.ax_spectrum.axvline(x=lambda_init-self.slider_width.val/2, ymin=0, ymax=1, **prop2)
        self.marker_right  = self.ax_spectrum.axvline(x=lambda_init+self.slider_width.val/2, ymin=0, ymax=1, **prop2)

        self.ax_spectrum.set_title(self.get_spectrum_location(), pad=15)
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
        
    def get_integration_range(self):
        """Return the integration range of the intensity image as a string in nm and eV."""
        int_lambda_min = self.spim.int_lambda_min[1]
        int_lambda_max = self.spim.int_lambda_max[1]
        title_pattern  = 'Integration range : [{:.2f}, {:.2f}]nm | [{:.3f}, {:.3f}]eV'
        title = title_pattern.format(
            int_lambda_min, int_lambda_max, fct.nm_eV(int_lambda_min), fct.nm_eV(int_lambda_max))
        return title
    
    def get_spectrum_location(self):
        """Return the indices and spatial position of the last clicked spectrum as a string."""
        title_pattern = "PL spectrum : [{}, {}] --- [x={}, y={}]µm"
        idx1 = self.indices_pixels_clicked[-1][0]
        idx2 = self.indices_pixels_clicked[-1][1]
        if self.spim.coordinate is None:
            return title_pattern.format(idx1, idx2, str(self.spim.x_range[idx2]), str(self.spim.y_range[idx1]))
        else:
            return title_pattern.format(idx1, idx2, str(self.spim.coordinate[idx1, idx2][0]), str(self.spim.coordinate[idx1, idx2][1]))
    
    def update_image(self, _):
        """Update the image and spectrum markers when sliders 'center' or 'width' are changed."""
        # Update sliders value.
        center = self.slider_center.val
        width  = self.slider_width.val
        # Update image.
        self.image.set_data(self.spim.intensity_map(center, width))
        # Update integration range.
        self.ax_image.set_title(self.get_integration_range(), pad=15)
        # Update markers position.
        func_unit = fct.nm_eV if self.indicator_spectrum.unit_eV else lambda x: x
        self.marker_left.set_xdata(func_unit(self.slider_center.val-self.slider_width.val/2))
        self.marker_center.set_xdata(func_unit(self.slider_center.val))
        self.marker_right.set_xdata(func_unit(self.slider_center.val+self.slider_width.val/2))
        
    def set_clim(self, _):
        self.image.set_clim(vmin=self.slider_clim_min.val, vmax=self.slider_clim_max.val)
        
    def reset_sliders(self, _):
        self.slider_center.reset()
        self.slider_width.reset()
        self.slider_clim_min.reset()
        self.slider_clim_max.reset()
    
    def set_full_range(self, _):
        center = self.spim.tab_lambda[int(len(self.spim.tab_lambda)/2)]
        width  = self.spim.tab_lambda.max() - self.spim.tab_lambda.min()
        self.slider_center.set_val(center)
        self.slider_width.set_val(width)
        self.image.set_data(self.spim.intensity_map(center, width))
        self.ax_image.set_title(self.get_integration_range(), pad=15)
        plt.draw()
        
    def delta_minus(self, _):
        # Calculate the spectrum average increment in wavelength.
        increment = np.mean(np.diff(self.spim.tab_lambda))
        center = self.slider_center.val - increment
        width  = self.slider_width.val
        self.slider_center.set_val(center)
        self.image.set_data(self.spim.intensity_map(center, width))
        
    def delta_plus(self, _):
        # Calculate the spectrum average increment in wavelength.
        increment = np.mean(np.diff(self.spim.tab_lambda))
        center = self.slider_center.val + increment
        width  = self.slider_width.val
        self.slider_center.set_val(center)
        self.image.set_data(self.spim.intensity_map(center, width))
    
    def plot_image(self, _):
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
        ax.set_title(self.get_integration_range(), pad=15)
        ax.tick_params(axis='both', direction='in', length=4.3, width=1.2)
        # Line below is not necessary with the magic command ``%matplotlib``. 
        fig.canvas.manager.show()  

    def save_spect(self, _):
        """Save the last clicked pixel in the list of pixels saved ``indices_pixels_saved``."""
        if self.indices_pixels_clicked[-1] not in self.indices_pixels_saved:
            self.indices_pixels_saved.append(self.indices_pixels_clicked[-1])
    
    def plot_spect(self, _):
        """
        Plot all saved spectra (i.e contained in ``indices_pixels_saved``).
        The yscale (linear or log) and ylim mode (autoscale or lock) is set by the radiobuttons.
        
        """
        if self.indices_pixels_saved == []:
            return
        fig, axes = plt.subplots(
            len(self.indices_pixels_saved), 1, figsize=(14, 7), sharex=True, gridspec_kw=dict(hspace=0))
        axes = [axes] if len(self.indices_pixels_saved)==1 else axes
        axes[-1].set_xlabel("Wavelength [nm]")
        
        ylim   = self.radiobutton_ylim.value_selected
        yscale = self.radiobutton_yscale.value_selected
        ymin_all = np.min([np.min(self.spim.matrix[idx[0], idx[1], :]) for idx in self.indices_pixels_saved])
        ymax_all = np.max([np.max(self.spim.matrix[idx[0], idx[1], :]) for idx in self.indices_pixels_saved])
        # Plot spectra vertically and set ylim and yscale.
        for (ax, idx) in zip(axes, self.indices_pixels_saved):
            if self.radiobutton_xunit.value_selected == 'nm':
                x = self.spim.tab_lambda
            elif self.radiobutton_xunit.value_selected == 'eV':
                x = fct.nm_eV(self.spim.tab_lambda)
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
        self.multicursor = MultiCursor_aml(fig.canvas, axes, color='k', lw=1, ls='-.', horizOn=True, vertOn=True)
        # Line below is not necessary with the magic command ``%matplotlib``.
        fig.canvas.manager.show()
        
    def reset_spect(self, _):
        """Reset the list of saved pixels."""
        self.indices_pixels_saved = []
        
    def image_onclick(self, event):
        """
        This function connects the intensity image to the spectrum.
        The spectrum corresponding to the pixel clicked is displayed.
        
        """
        if event.inaxes != self.ax_image:
            return
        # Get the positional indices in ``matrix`` of the pixel clicked.
        # - ``idx1`` is searched with ``event.ydata``,
        # - ``idx2`` with ``event.xdata``. 
        # This is because we scan by successive horizontal lines from bottom to top, by increasing y coordinates.
        # WAIT : not always true: at LNCMI ``Mapping2`` scan program does successive vertical lines from top to bottom
        # --> I added the parameter ``horizontal_scan`` in the constructor of SPIM 
        if self.spim.coordinate is None:
            idx1 = np.searchsorted(self.spim.y_range, event.ydata)
            idx2 = np.searchsorted(self.spim.x_range, event.xdata)
        else:
            idx1 = fct.find_nearest(self.spim.y_range, event.ydata)[0]
            idx2 = fct.find_nearest(self.spim.x_range, event.xdata)[0]
        self.indices_pixels_clicked.append([idx1, idx2])
        # Update the spectrum.
        self.spectrum.set_ydata(self.spim.matrix[idx1, idx2, :])
        self.set_ylim(self)
        self.ax_spectrum.set_title(self.get_spectrum_location(), pad=15)
        plt.draw()
        
    def set_yscale(self, label):
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
        
    def set_ylim(self, label):
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
        
    def set_xunit(self, label):
        if label == 'nm':
            x_nm = self.spim.tab_lambda
            self.spectrum.set_xdata(x_nm)
            self.ax_spectrum.set_xlim(np.min(x_nm), np.max(x_nm))
            self.ax_spectrum.set_xlabel('Wavelength [nm]')
            self.indicator_spectrum.unit_eV = False
            # Update markers positions.
            self.update_image(self)
        elif label == 'eV':
            x_eV = fct.nm_eV(self.spim.tab_lambda)
            self.spectrum.set_xdata(x_eV)
            self.ax_spectrum.set_xlim(np.min(x_eV), np.max(x_eV))
            self.ax_spectrum.set_xlabel('Energy [eV]')
            self.indicator_spectrum.unit_eV = True
            # Update markers positions.
            self.update_image(self)
        plt.draw()
    
    def display_tools(self, label):
        if label == 'Indicator':
            self.indicator_spectrum.set_active(not self.indicator_spectrum.get_active())
            if not self.indicator_spectrum.get_active():
                self.indicator_spectrum.remove_artists()
                self.fig.canvas.draw_idle()
        elif label == 'Selector':
            self.rect_selector.set_active(not self.rect_selector.get_active())
            if self.rect_selector.get_active():
                plt.disconnect(self.cid_image)
                self.cursor_image.set_active(False)
            else:
                self.cid_image = self.fig.canvas.mpl_connect('button_release_event', self.image_onclick)
                self.cursor_image.set_active(True)
            
    def rectselect_action(self, eclick, erelease):
        """``eclick`` and ``erelease`` are the press and release events."""
        if self.spim.coordinate is None:
            idx_00 = np.searchsorted(self.spim.y_range, eclick.ydata)
            idx_01 = np.searchsorted(self.spim.x_range, eclick.xdata)
            idx_10 = np.searchsorted(self.spim.y_range, erelease.ydata)
            idx_11 = np.searchsorted(self.spim.x_range, erelease.xdata)
        else:
            idx_00 = fct.find_nearest(self.spim.y_range, eclick.ydata)[0]
            idx_01 = fct.find_nearest(self.spim.x_range, eclick.xdata)[0]
            idx_10 = fct.find_nearest(self.spim.y_range, erelease.ydata)[0]
            idx_11 = fct.find_nearest(self.spim.x_range, erelease.xdata)[0]
        range_pix_x = np.arange(idx_00, idx_10+1, 1) if idx_00 <= idx_10 else np.arange(idx_10, idx_00+1, 1)
        range_pix_y = np.arange(idx_01, idx_11+1, 1) if idx_01 <= idx_11 else np.arange(idx_11, idx_01+1, 1)
        pixels = [[i, j] for i in range_pix_x for j in range_pix_y]
        spectra = np.array([self.spim.matrix[pixel[0], pixel[1], :] for pixel in pixels])
        self.spectrum.set_ydata(np.mean(spectra, axis=0))
        self.set_ylim(self)
        plt.draw()
        
    def rectselect_persist(self, event):
        if self.rect_selector.active:
            self.rect_selector.update()


class Indicator(Cursor):
    """Create a cursor with ability to measure horizontal and vertical distances between two points."""
    
    def __init__(self, ax=None, unit_eV=False, stick_to_data=True, useblit=True, **cursorprops):
        """
        Parameters
        ----------
        unit_eV : whether the xaxis is in eV or nm.
    
        stick_to_data : whether to measure distances between actual data points or between exact cursor positions.
        """
        default_esthetic = {'color':'tab:red', 'ls':'--', 'lw':1}
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
        
        self.reftext  = None
        self.refarrow = None
        self.refvline = None
        self.refhline = None
        
        self.connect_event('button_release_event', self.onclick)
        self.connect_event('motion_notify_event', self.onthemove)
        plt.draw()

    def onclick(self, event):
        if (event.inaxes != self.ax) or (not self.active):
            return
        if self.first_click:
            self.remove_artists(reftext=True, refarrow=False, refvline=False, refhline=False)
            self.append_xy_data_clicked(event)
            self.count += 1
            self.first_click = False
            self.move = True
        else:
            self.append_xy_data_clicked(event)
            self.draw_arrow(x0=self.xs[self.count-1], y0=self.ys[self.count-1], x=self.xs[-1], y=self.ys[-1])
            self.remove_artists(reftext=False, refarrow=False, refvline=True, refhline=True)
            prop = dict(boxstyle='round', facecolor='gainsboro', alpha=1, lw=1.5)
            #left, bottom, width, height = self.ax.bbox.bounds
            #left, bottom, width, height = self.ax.get_position().bounds
            self.reftext = self.ax.text(
                1, 1, r'$\Delta$x = {} eV | {} nm'.format(*self.distance_x())\
                + '\n' + r'$\Delta$y = {}'.format(self.distance_y()),
                ha='right', va='top', transform=self.ax.transAxes, fontdict=dict(weight='bold'), bbox=prop)
            self.count += 1
            self.first_click = True
            self.move = False
            plt.draw()
            
    def onthemove(self, event):
        if (event.inaxes != self.ax) or (not self.move) or (not self.active):
            return
        self.draw_arrow(x0=self.xs[self.count-1], y0=self.ys[self.count-1], x=event.xdata, y=event.ydata)
        self.remove_artists(reftext=False, refarrow=False, refvline=True, refhline=True)
        self.refvline = self.ax.vlines(event.xdata, self.ax.get_ylim()[0], self.ax.get_ylim()[1],
                                       transform=self.ax.transData, ls=self.linev.get_ls(), lw=self.linev.get_lw(),
                                       color=self.linev.get_color())
        self.refhline = self.ax.hlines(event.ydata, self.ax.get_xlim()[0], self.ax.get_xlim()[1],
                                       transform=self.ax.transData, ls=self.lineh.get_ls(), lw=self.lineh.get_lw(),
                                       color=self.lineh.get_color())
        plt.draw()

    def append_xy_data_clicked(self, event):
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
    
    def draw_arrow(self, x0, y0, x, y):
        """Draw an arrow between specified points, and clean the previous drawn arrow."""
        self.remove_artists(reftext=False, refarrow=True, refvline=False, refhline=False)
        pos = self.ax.transData.transform([(x0, y0), (x, y)])
        pos = self.ax.transAxes.inverted().transform([(pos[0, 0], pos[0, 1]), (pos[1, 0], pos[1, 1])])
        self.refarrow = self.ax.arrow(pos[0, 0], pos[0, 1], pos[1, 0]-pos[0, 0], pos[1, 1]-pos[0, 1], fc='k',
                         length_includes_head=True,
                         width=0.002,
                         transform=self.ax.transAxes)
    
    def distance_x(self):
        if not self.unit_eV:
            delta_x_eV = np.abs(np.around(1239.84193/self.xs[self.count]-1239.84193/self.xs[self.count-1], decimals=4))
            delta_x_nm = np.abs(np.around(self.xs[self.count]-self.xs[self.count-1], decimals=2))
        elif self.unit_eV:
            delta_x_eV = np.abs(np.around(self.xs[self.count]-self.xs[self.count-1], decimals=4))
            delta_x_nm = np.abs(np.around(1239.84193/self.xs[self.count]-1239.84193/self.xs[self.count-1], decimals=2))
        return [delta_x_eV, delta_x_nm]

    def distance_y(self):
        delta_y = np.abs(np.around(self.ys[self.count]-self.ys[self.count-1], decimals=2))
        return delta_y

    def remove_artists(self, reftext=True, refarrow=True, refvline=True, refhline=True):
        """Remove the artists with ``True`` value."""
        if reftext == True:
            try: self.reftext.remove()
            except: pass
        if refarrow == True:
            try: self.refarrow.remove()
            except: pass
        if refvline == True:
            try: self.refvline.remove()
            except: pass
        if refhline == True:
            try: self.refhline.remove()
            except: pass
        plt.draw()


class FancyButton(Button):
    """This class is a simple wrapper for the ``matplotlib.widgets.Button`` with more control on the appearance."""
    
    indianred = dict(color='gainsboro', labelcolor='indianred', spinecolor='indianred', hovercolor='dimgrey')
    darkviolet = dict(color='gainsboro', labelcolor='darkviolet', spinecolor='darkviolet', hovercolor='dimgrey')
    darkviolet2 = dict(color='darkviolet', labelcolor='k', spinecolor='k', hovercolor='dimgrey')
    all_style = {'indianred':indianred, 'darkviolet':darkviolet, 'darkviolet2':darkviolet2}
    
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
    
    indianred  = dict(color='lightcoral', facecolor='gainsboro', labelcolor='k', spinecolor='indianred', barcolor='k')
    darkviolet = dict(color='darkviolet', facecolor='gainsboro', labelcolor='k', spinecolor='k', barcolor='k')
    all_style = {'indianred':indianred, 'darkviolet':darkviolet}
    
    def __init__(self, ax, label, valmin, valmax, style='darkviolet', color=None, facecolor=None, labelcolor=None,
                 fontsize=10, spine_lw=2, spinecolor=None, barcolor=None, **kwargs):
            
        if style is None and None in [color, facecolor, labelcolor, spinecolor, barcolor]:
            raise ValueError("If ``style`` is None, all other color type arguments must be given.")
            
        dic_args = dict(color=color, facecolor=facecolor, labelcolor=labelcolor, spinecolor=spinecolor, barcolor=barcolor)
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
    """This class is a simple wrapper for the ``matplotlib.widgets.RadioButtons`` with more control on the appearance."""
    
    indianred  = dict(facecolor='gainsboro', labelcolor='indianred', activecolor='indianred', spinecolor='indianred')
    darkviolet = dict(facecolor='gainsboro', labelcolor='darkviolet', activecolor='darkviolet', spinecolor='darkviolet')
    all_style = {'indianred':indianred, 'darkviolet':darkviolet}
    
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
            circle.set_facecolor(esthetic['activecolor'] if i==active else esthetic['facecolor'])
        # Spines
        for spine in self.ax.spines.values():
            spine.set_lw(spine_lw)
            spine.set_color(esthetic['spinecolor'])


class FancyCheckButtons(CheckButtons):
    """This class is a simple wrapper for the ``matplotlib.widgets.CheckButtons`` with more control on the appearance."""
    
    indianred  = dict(facecolor='gainsboro', labelcolor='indianred', spinecolor='indianred')
    darkviolet = dict(facecolor='gainsboro', labelcolor='darkviolet', spinecolor='darkviolet')
    all_style = {'indianred':indianred, 'darkviolet':darkviolet}
    
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


class MultiCursor_aml(MultiCursor):
    """This class is a simple wrapper for the ``matplotlib.widgets.Multicursor`` for clearing the Multicursor when outside the axes."""
    
    def __init__(self, canvas, axes, useblit=True, horizOn=False, vertOn=True, clear_out_of_axes=True, **lineprops):
        MultiCursor.__init__(self, canvas, axes, useblit=useblit, horizOn=horizOn, vertOn=vertOn, **lineprops)
        if clear_out_of_axes:
            canvas.mpl_connect('motion_notify_event', self.clear_cursor)
    
    def clear_cursor(self, event):
        if event.inaxes not in self.axes:
            self.visible = False
            plt.draw()
        else:
            self.visible = True
            plt.draw()
