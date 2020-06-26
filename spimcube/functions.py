import os
import time as time
import glob
import re
import math as math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt
from scipy.special import erfc
import pandas as pd


#######################################################################################################################
##################################################### ARRAY PROCESSING ################################################
#######################################################################################################################


def find_nearest(array, value):
    """Return the index and corresponding value of the nearest element of an 'array' from a given 'value'.

    The array is sorted before operation so this is useful only for unique value array.

    """
    array = np.sort(array)
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return [int(idx - 1), array[idx - 1]]
    else:
        return [int(idx), array[idx]]


def generate_boundaries(L, method='extrema', fit_function='gaussian', fit_parameter=None, FWHM_factor=1):
    """Return a tuple with min and max boundaries calculated for ``L`` list.

    Parameters
    ----------
    L : iterable

    method : (string)
       -'extrema': min & max of the list.
       -'extrema_delta': a bit more expanded than 'extrema', by 10% of the difference between min and max values.
       -'FWHM': only available if a histogram of the value of the list has been computed and a gaussian fit of this
       histogram has been done, and result is passed in argument through ``fit_parameter``.
       -'variance': return (mean - variance/2) , (mean + variance/2).

    fit_function : (string) 'gaussian' or 'lorentzian'.
        When ``method`` is 'FWHM', it specifies which functions was used to compute the fit of the histogram.

    fit_parameter : (list)
        List containing the fitted parameters of the ``fit_function``.

    FWHM_factor : default to 1.
        When ``method`` is 'FWHM', factor to determinate the extent around the mean value of the list ``L``.
        Example: if FWHM_factor = 3, the function returns (mean - 3*FWHM), (mean + 3*FWHM)
       
    TO DO: add 'std' method

    """
    mean, min_, max_, var = np.nanmean(L), np.nanmin(L), np.nanmax(L), np.nanvar(L)
    if method == 'extrema':
        return min_, max_
    elif method == 'extrema_delta':
        delta = np.fabs(max_ - min_)
        return min_ - 0.1 * delta, max_ + 0.1 * delta
    elif method == 'variance':
        return mean - 0.5 * var, mean + 0.5 * var
    elif method == 'FWHM':
        if fit_parameter is None:
            raise AttributeError("``fit_parameter`` must be provided with the 'FWHM' method.")
        FWHM = None
        if fit_function == 'gaussian':
            FWHM = abs(2 * np.sqrt(2 * np.log(2)) * fit_parameter[0])
        elif fit_function == 'lorentzian':
            FWHM = abs(fit_parameter[0])
        mean = abs(fit_parameter[1])
        return max(mean - FWHM_factor * FWHM, 0), mean + FWHM_factor * FWHM


def find_peaks(y, x=None, widths=None, decimals=None, **kwargs):
    # Set the good (for me) default value settings for ``find_peaks_cwt``.
    if widths is None:
        widths = np.arange(1, 60)
    kwargs['min_snr'] = kwargs.pop('min_snr', 2)
    # Find the peaks.
    peaks = find_peaks_cwt(vector=y, widths=widths, **kwargs)  # The args can be adjusted.
    # Transform in ``x`` coordinates if provided.
    if x is not None:
        peaks = x[peaks]
    if decimals is not None:
        peaks = np.round(peaks, decimals=decimals)
    return peaks


def sort_x_y(x, y):
    """Return x and y arrays sorted according to the increasing values of x."""
    y = y[np.argsort(x)]
    x = np.sort(x)
    return x, y


def z_score_mean(spectrum):
    mean = np.mean(spectrum)
    sigma = np.std(spectrum)
    return (spectrum - mean) / sigma


def z_score_diff(spectrum):
    return abs(np.diff(spectrum))


def remove_spikes(array, window_size=10, threshold=200):
    """Removes spikes from data contained in ``array`` and return the cleaned array.

    Parameters
    ----------
    array: array-like object.
        Must be of dimension ???

    window_size :

    threshold :

    """
    array_ = np.array(array.copy())  # Don't overwrite the data.
    data = []
    shape = array_.shape

    # Transforming the array in the form of a collection of spectra.
    if array_.ndim == 1:
        data = array_[np.newaxis, :]
    elif array_.ndim > 1:
        new_shape = (np.prod(shape[:-1]), shape[-1])
        data = np.reshape(array_, new_shape)
    else:
        raise TypeError("``array`` must be an array-like object with finite dimension >= 1.")

    # Selecting spikes based on threshold criterion. Note that spikes dimension 1 is one time smaller.
    spikes = z_score_diff(data) > threshold

    # Replace each spike value by the mean value of neighboring pixels, the number of which is determined
    # by ``window_size``.
    for i, spectrum in enumerate(data):
        for j in range(len(spectrum) - 1):
            if spikes[i, j]:
                window = np.arange(max(0, j - window_size), min(j + window_size + 1, len(spectrum) - 1))
                window = window[~spikes[i, window]]  # Eliminate from the window the pixels that are spkikes.
                if len(window) <= 1:
                    # if array_.ndim ==
                    raise ValueError(
                        "Spike at {?} requires larger ``window_size`` to calculate a proper mean value around spike.")
                data[i, j] = np.mean(data[i, window])

    data = np.reshape(data, shape)
    return data


#######################################################################################################################
######################################################## CONVERSION ###################################################
#######################################################################################################################

def nm_eV(*args, decimals=5):
    """Return the converted energy from nm to eV or from eV to nm."""
    if len(args) == 1:
        value = np.array(args[0])
        return np.round(1239.84193 / value, decimals=decimals)
    else:
        values = np.empty(len(args))
        for i, arg in enumerate(args):
            values[i] = np.round((1239.84193 / arg), decimals=decimals)
        return values


def delta_nm2eV(delta_nm, Lambda, decimals=3):
    """Return the energy delta in meV.
    
    Parameters
    ----------
    delta_nm : The difference in energy in nm. Can be either single value, list or numpy.ndarray.
    Lambda : The central wavelength. Can be either single value, list or numpy.ndarray.
    decimals : Number of decimals for the return value/list.

    """
    if isinstance(delta_nm, list):
        delta_nm = np.asarray(delta_nm)
        Lambda = np.asarray(Lambda)
        return list(np.around(np.abs(1000 * 1239.84193 * delta_nm / Lambda ** 2), decimals=decimals))
    elif isinstance(delta_nm, np.ndarray):
        Lambda = np.asarray(Lambda)
        return np.around(np.abs(1000 * 1239.84193 * delta_nm / Lambda ** 2), decimals=decimals)
    else:
        return round(abs(1000 * 1239.84193 * delta_nm / Lambda ** 2), decimals)


def date(representation='literal'):
    """Return the local date.

    Parameters
    ----------
    representation : 'literal', 'number' (default: 'literal')
        Example : 'literal' --> "24 march 2020"
                  'number'  --> "24/03/2020"

    """
    localtime = time.localtime()
    year = localtime[0]
    month = localtime[1]
    day = localtime[2]
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
              'november', 'december']
    if representation == 'literal':
        return '{0} {1} {2}'.format(day, months[month - 1], year)
    elif representation == 'number':
        day = '0' + str(day) if int(day) < 10 else str(day)
        month = '0' + str(month) if int(month) < 10 else str(month)
        return '{0}/{1}/{2}'.format(day, month, year)


#######################################################################################################################
####################################################### FIT FUNCTIONS #################################################
#######################################################################################################################


def linear(x, a, b):
    return a * x + b


def gaussian(x, sigma, mu, A, B):
    return np.exp(-((x - mu) / np.sqrt(2 * sigma ** 2)) ** 2) * A + B


# linear & affine do the same but with different user input, I think that linear is easier to use because one doesn't
# need to calculate the y-intercept (ordonnée à l'origine) ``b``

def gaussian_linear(x, sigma, mu, A, B, a):
    return (np.exp(-((x - mu) / np.sqrt(2 * sigma ** 2)) ** 2) * A + B) + (a * x)


def gaussian_affine(x, sigma, mu, A, a, b):
    return (np.exp(-((x - mu) / np.sqrt(2 * sigma ** 2)) ** 2) * A) + (a * x + b)


def gaussian_polynome2(x, sigma, mu, A, a, b, c):
    return (np.exp(-((x - mu) / np.sqrt(2 * sigma ** 2)) ** 2) * A) + (a * x ** 2 + b * x + c)


def double_gaussian(x, sigma1, mu1, A1, sigma2, mu2, A2, B):
    return (np.exp(-((x - mu1) / np.sqrt(2 * sigma1 ** 2)) ** 2) * A1) + (
                np.exp(-((x - mu2) / np.sqrt(2 * sigma2 ** 2)) ** 2) * A2) + B


def double_gaussian_linear(x, sigma1, mu1, A1, sigma2, mu2, A2, B, a):
    return (np.exp(-((x - mu1) / np.sqrt(2 * sigma1 ** 2)) ** 2) * A1) + (
                np.exp(-((x - mu2) / np.sqrt(2 * sigma2 ** 2)) ** 2) * A2) + B + (a * x)


def triple_gaussian(x, sigma1, mu1, A1, sigma2, mu2, A2, sigma3, mu3, A3, B):
    return (np.exp(-((x - mu1) / np.sqrt(2 * sigma1 ** 2)) ** 2) * A1) + (
                np.exp(-((x - mu2) / np.sqrt(2 * sigma2 ** 2)) ** 2) * A2) \
           + (np.exp(-((x - mu3) / np.sqrt(2 * sigma3 ** 2)) ** 2) * A3) + B


def quadruple_gaussian(x, sigma1, mu1, A1, sigma2, mu2, A2, sigma3, mu3, A3, sigma4, mu4, A4, B):
    return (np.exp(-((x - mu1) / np.sqrt(2 * sigma1 ** 2)) ** 2) * A1) + (
                np.exp(-((x - mu2) / np.sqrt(2 * sigma2 ** 2)) ** 2) * A2) \
           + (np.exp(-((x - mu3) / np.sqrt(2 * sigma3 ** 2)) ** 2) * A3) + (
                       np.exp(-((x - mu4) / np.sqrt(2 * sigma4 ** 2)) ** 2) * A4) + B


def lorentzian(x, gamma, mu, A, B):
    return 1 / (1 + (2 * (x - mu) / gamma) ** 2) * A + B


def lorentzian_linear(x, gamma, mu, A, B, a):
    return (1 / (1 + (2 * (x - mu) / gamma) ** 2) * A + B) + (a * x)


def lorentzian_affine(x, gamma, mu, A, a, b):
    return (1 / (1 + (2 * (x - mu) / gamma) ** 2) * A) + (a * x + b)


def double_lorentzian(x, gamma1, mu1, A1, gamma2, mu2, A2, B):
    return (1 / (1 + (2 * (x - mu1) / gamma1) ** 2) * A1) + (1 / (1 + (2 * (x - mu2) / gamma2) ** 2) * A2) + B


def double_lorentzian_linear(x, gamma1, mu1, A1, gamma2, mu2, A2, B, a):
    return (1 / (1 + (2 * (x - mu1) / gamma1) ** 2) * A1) + (1 / (1 + (2 * (x - mu2) / gamma2) ** 2) * A2) + B + (a * x)


def cosinus(x, f, A, B):
    return A * np.cos(f * x) + B


def sinus(x, f, A, B):
    return A * np.sin(f * x) + B


def g2(x, a=None, lambda1=None, lambda2=None, t0=None):
    # values for a single defect in hBN in the visible ==> article "PRB Martinez et al. 2016"
    a = 0.6 if a is None else a
    lambda1 = 0.51 if lambda1 is None else lambda1
    lambda2 = 0.00014 if lambda2 is None else lambda2
    t0 = 12 if t0 is None else t0

    return 1 - (1 + a) * np.exp(-np.abs(x - t0) * lambda1) + a * np.exp(-np.abs(x - t0) * lambda2)


def g2_irf(x, a=None, lambda1=None, lambda2=None, t0=None):
    """
    The FWHM of the IRF is set in the definition of this function. The value must be changed here because it is a fixed
    parameter that must not be a fitting parameter. It depends on one's optoelectronic devices (APDs, PMT, etc.).

    Note: this function uses the mathematical function ``scipy.special.erfc``.
    """

    # values for a single defect in hBN in the visible ==> article "PRB Martinez et al. 2016"
    a = 0.6 if a is None else a
    lambda1 = 0.51 if lambda1 is None else lambda1
    lambda2 = 0.00014 if lambda2 is None else lambda2
    t0 = 12 if t0 is None else t0
    FWHM = 1.5  # in ns
    sigma = FWHM / 2.355  # in ns

    x_1 = x[x < 100]
    exp1 = np.exp(lambda1 * (x_1 - t0))
    y_1 = 1 - (1 + a) * 0.5 * np.exp(lambda1 ** 2 * sigma ** 2 / 2) \
          * (erfc(((x_1 - t0) + lambda1 * sigma ** 2) / (sigma * np.sqrt(2))) * exp1
             + erfc((-(x_1 - t0) + lambda1 * sigma ** 2) / (sigma * np.sqrt(2))) * np.exp(-lambda1 * (x_1 - t0))
             ) \
          + a * 0.5 * np.exp(lambda2 ** 2 * sigma ** 2 / 2) \
          * (erfc(((x_1 - t0) + lambda2 * sigma ** 2) / (sigma * np.sqrt(2))) * np.exp(lambda2 * (x_1 - t0))
             + erfc((-(x_1 - t0) + lambda2 * sigma ** 2) / (sigma * np.sqrt(2))) * np.exp(-lambda2 * (x_1 - t0))
             )

    x_2 = x[x >= 100]  # 'exp(x*lambda1)' gives overflow but multiply by the term in erfc gives '0' so I set it to '0'
    exp2 = 0
    y_2 = 1 - (1 + a) * 0.5 * np.exp(lambda1 ** 2 * sigma ** 2 / 2) \
          * (erfc(((x_2 - t0) + lambda1 * sigma ** 2) / (sigma * np.sqrt(2))) * exp2
             + erfc((-(x_2 - t0) + lambda1 * sigma ** 2) / (sigma * np.sqrt(2))) * np.exp(-lambda1 * (x_2 - t0))
             ) \
          + a * 0.5 * np.exp(lambda2 ** 2 * sigma ** 2 / 2) \
          * (erfc(((x_2 - t0) + lambda2 * sigma ** 2) / (sigma * np.sqrt(2))) * np.exp(lambda2 * (x_2 - t0))
             + erfc((-(x_2 - t0) + lambda2 * sigma ** 2) / (sigma * np.sqrt(2))) * np.exp(-lambda2 * (x_2 - t0))
             )

    return np.concatenate((y_1, y_2))


#######################################################################################################################
################################################# STRING AND FOLDER RELATED ###########################################
#######################################################################################################################

def get_value(str2search, filename):
    """Return the float value corresponding to the first occurence of 'str2search' in 'filename'."""

    file = open(filename, 'r', encoding='latin-1')
    value = None
    for line in file:
        if str2search in line:
            value = re.findall(r"\d+\.\d+", line.replace(",", "."))
            break
    file.close()
    if not value:
        return []
    else:
        return float(value[0])


def get_filenames_at_location(loc, format_, keyword=None, print_name=True):
    """Return a list of all file names at location matching with the specified format and print the names.
    
    Parameters
    ----------
    loc : path to the folder location.

    format_ : format of the searched files. Do not include the '.' in the format.
    
    keyword : string pattern to match in the file names.

    print_name : print the name of the files in the standard output.

    """
    # we choose the current directory where all the files are located
    os.chdir(loc)
    # glob return a list with all the files with format_
    if keyword is not None:
        filenames = glob.glob("*{0}*.{1}".format(keyword, format_), recursive=False)
    else:
        filenames = glob.glob("*.{}".format(format_), recursive=False)
    length = len('.' + format_)
    for i, file in enumerate(filenames):
        # remove the specified format at the end of the file and the preceding point
        filenames[i] = file[:-length]
    if print_name:
        for i, file in enumerate(filenames):
            print('({})'.format(i), file)
        print("\n")
    return filenames


def get_last_filename_number_at_location(loc='/Users/pelini/L2C/Manip uPL UV 3K/Data/analysis/images en bazar/',
                                         keyword='image_', fmt='pdf'):
    """Return the last number of numbered filename.
    
    Parameters
    ----------
    loc     : the location of the folder to search in
    keyword : the pattern to filter file names
    fmt     : the format of the files

    """
    filenames = get_filenames_at_location(loc, fmt, keyword=keyword, print_name=False)
    numbers = np.empty(len(filenames), dtype=np.int8)
    for i, file in enumerate(filenames):
        number = re.findall(r'\d+', file)
        numbers[i] = int(number[0]) if len(number) > 0 else 0
    numbers = np.sort(numbers)
    if len(numbers) != 0:
        return numbers[-1]
    else:
        return int(0)


#######################################################################################################################
######################################################## PLOT & DRAW ##################################################
#######################################################################################################################


def spectrum_plot(tab_of_lambda, data, pix1, pix2=None, lambda_min=290, lambda_max=330, vline=True, x_vline=0,
                  scale='linear', hspace=0, vspace=0, fontsize=10, **kwargs):
    """
    Plot a spectrum contained in a datacube at [pix1, pix2] or simply [pix1] if the data cube has been flatened,
    and can graphically show the max value by drawing a vertical line with the value.
    
    Parameters
    ----------
    kwargs : passed to ``plt.plot`` function.

    """
    Lmin = find_nearest(tab_of_lambda, lambda_min)
    Lmax = find_nearest(tab_of_lambda, lambda_max)
    x = tab_of_lambda[Lmin[0]:Lmax[0]]
    if pix2 is not None:
        y = data[pix1, pix2, Lmin[0]:Lmax[0]]
    else:
        y = data[pix1, Lmin[0]:Lmax[0]]

    fig = plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    spectrum = plt.plot(x, y, **kwargs)
    ax1.set_yticks([])
    ax1.set_yticklabels('')
    ax1.set_xlabel("wavelength (nm)")
    ax2 = plt.twiny(ax=ax1)
    ax2.set_yticks([])
    ax2.set_yticklabels('')
    ax1Xs = ax1.get_xticks()
    ax2Xs = []
    for X in ax1Xs:
        ax2Xs.append(round(nm_eV(X), 3))  # conversion from nm to eV
    ax2.set_xticks(ax1Xs)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels(ax2Xs)
    ax2.set_xlabel("energy (eV)")
    plt.yscale(scale)
    if vline:
        if x_vline == 0:
            x_unit = find_nearest(tab_of_lambda, x[np.argmax(y)])[1]
            ax1.axvline(x=x_unit, ymin=0, ymax=1, color='k', lw=0.7, linestyle='--')
            ax1.annotate(s='{}eV \n{}nm'.format(round(nm_eV(x_unit), 3), x_unit), xy=(x_unit, np.max(y)), ha='center',
                         xycoords='data', xytext=(x_unit + hspace, np.max(y) + vspace), textcoords='data',
                         fontsize=fontsize)
            # , arrowprops=dict(arrowstyle='->')
        else:
            ax1.axvline(x=x_vline, ymin=0, ymax=1, color='k', lw=0.7, linestyle='--')
            ax1.annotate(s='{}eV \n{}nm'.format(round(nm_eV(x_vline), 3), x_vline), xy=(x_vline, np.max(y)),
                         ha='center',
                         xycoords='data', xytext=(x_vline + hspace, np.max(y) + vspace), textcoords='data',
                         fontsize=fontsize)


def several_spectrum_plot(data, list_of_index, xdata=None, tab_of_lambda=None, lambda_min=290, lambda_max=330,
                          scale='linear', legend=False):
    """
    Plot several spectra contained in a datacube at the positions [pix1, pix2] (contained in ``list_of_index``).
    If ``tab_of_lambda`` is given, it overrides ``xdata``, and the values are fetched up in it between
    ``lambda_min/max``.

    """
    # In that case: 'data' is restricted by 'lambda_min/max' in energy range and 'x' is created using 'tab_of_lambda'.
    if tab_of_lambda is not None:
        Lmin = find_nearest(tab_of_lambda, lambda_min)
        Lmax = find_nearest(tab_of_lambda, lambda_max)
        x = tab_of_lambda[Lmin[0]:Lmax[0]]
        for index in list_of_index:
            plt.plot(x, data[index[1], index[0], Lmin[0]:Lmax[0]], label=str(index))
            if legend: plt.legend()
            plt.yscale(scale)

    # In that case: 'data' should be accompagnied with xdata of the same length.
    elif xdata is not None:
        for index in list_of_index:
            plt.plot(xdata, data[index[1], index[0], :], label=str(index))
            if legend: plt.legend()
            plt.yscale(scale)

    else:
        for index in list_of_index:
            plt.plot(data[index[1], index[0], :], label=str(index))
            if legend:
                plt.legend()
            plt.yscale(scale)


def plot_with_fit(x, y, ax=None, plot_function=None, initial_guess=None, kwarg_for_plot_function={}, label=True,
                  bounds=(-np.inf, np.inf), **fit_functions):
    """
    Plot y versus x with the plot_function given (default to ``plt.plot``), and compute and plot the fits with the
    fit functions given, display the fit parameters in the label.
    For 'gaussian', 'lorentzian' & 'linear' functions, an intelligent guess is automatically computed for the initial
    values.
    
    Parameters
    ----------
    x, y : data to be plotted
    
    ax : axis on which to plot, default is to create a new figure and ax
    
    plot_function : function to use to plot the data. Ex: `plt.plot` (default), `plt.scatter`, `plt.hist2d`, etc.
    
    initial_guess : Must be a dictionnary.
                        key   : name of the fit function
                        value : list of guess values for the parameters of the fit function

    bounds : 2-tuple of array-like or scalar.
        If array: must have length equal to the number of parameters.
        If scalar: same bound for each parameter.
        Default to  no bounds (-np.inf, np.inf).
                        
    **fit_functions : the 'key' must be the name of the fit function and the 'value' the proper function. 
                      Example: >> gaussian=fct.gaussian

    Return a dictionnary in which the key indicates the fit function and the value is an array of fitted parameters.

    """

    def make_label(fit_function_name, popt):

        values = np.concatenate([[j, popt] for j, popt in enumerate(popt, 1)])
        if fit_function_name == 'gaussian':
            label_base = 'fit {}:' + ' P{:1.0f}={:5.4f}, ' * len(popt) + 'FWHM={:5.4f}  '
            FWHM = abs(2 * np.sqrt(2 * np.log(2)) * popt[0])
            values = np.append(values, FWHM)
        elif fit_function_name in ['lorentzian', 'lorentzian_linear', 'lorentzian_affine']:
            label_base = 'fit {}:' + ' P{:1.0f}={:5.4f}, ' * len(popt) + 'FWHM={:5.4f}  '
            FWHM = abs(popt[0])
            values = np.append(values, FWHM)
        else:
            label_base = 'fit {}:' + ' P{:1.0f}={:5.4f}, ' * len(popt)

        return label_base[:-2].format(fit_function_name, *values)

    # I need to manually control the cycle of color because it doesn't change color automatically with different
    # function of plot.
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 6))
    if plot_function == plt.scatter:  # It is good to zoom out a little bit with a scatter plot.
        xleft, xright = generate_boundaries(x, method='extrema_delta')
        ybottom, ytop = generate_boundaries(y, method='extrema_delta')
        ax.set_xlim(left=xleft, right=xright)
        ax.set_ylim(bottom=ybottom, top=ytop)
    x = np.asarray(x) if not isinstance(x, np.ndarray) else x
    y = np.asarray(y) if not isinstance(y, np.ndarray) else y
    xfit = np.linspace(np.nanmin(x), np.nanmax(x), 10 * len(x))

    plot_function = plt.plot if plot_function is None else plot_function
    if plot_function == plt.plot:
        plot_function(x, y, c=colors[0], **kwarg_for_plot_function)
    else:
        plot_function(x, y, **kwarg_for_plot_function)

    import warnings
    warnings.filterwarnings("error")

    result_fit_parameters = {}
    for i, (name, function) in enumerate(fit_functions.items(), 1):
        if (initial_guess is not None) and (name in initial_guess.keys()):
            try:
                popt, pcov = curve_fit(function, x, y, p0=initial_guess[name], bounds=bounds)
                result_fit_parameters[name] = popt
            except Exception:
                print("Fit with {} function did not succeed, try to provide better initial guess.".format(name))
                result_fit_parameters[name] = None
        elif name in ['gaussian', 'lorentzian']:
            wide = 0.1
            mu = x[np.argmax(y)]
            B = np.mean([y[0], y[-1]])
            A = np.max(y) - B
            try:
                popt, pcov = curve_fit(function, x, y, p0=[wide, mu, A, B])
                result_fit_parameters[name] = popt
            except Exception:
                print("Fit with {} function did not succeed, try to provide initial guess.".format(name))
                result_fit_parameters[name] = None
        elif name in ['gaussian_linear', 'lorentzian_linear']:
            wide = 0.1
            mu = x[np.argmax(y)]
            B = np.mean([y[0], y[-1]])
            A = np.max(y) - B
            a = (y[-1] - y[0]) / (x[-1] - x[0])
            try:
                popt, pcov = curve_fit(function, x, y, p0=[wide, mu, A, B, a])
                result_fit_parameters[name] = popt
            except Exception:
                print("Fit with {} function did not succeed, try to provide initial guess.".format(name))
                result_fit_parameters[name] = None
        elif name == 'linear':
            a = (y[-1] - y[0]) / (x[-1] - x[0])
            b = y[0] - a * x[0]
            try:
                popt, pcov = curve_fit(function, x, y, p0=[a, b])
                result_fit_parameters[name] = popt
            except Exception:
                print("Fit with {} function did not succeed, try to provide initial guess.".format(name))
                result_fit_parameters[name] = None
        else:
            raise ValueError("An initial guess for the fit function `{}` must be provided".format(name))

        if result_fit_parameters[name] is not None:
            if label is True:
                ax.plot(xfit, function(xfit, *popt), '--', c=colors[i], label=make_label(name, popt))
                print(name, 'fit standard deviation errors: ', np.sqrt(np.diag(pcov)))
            else:
                ax.plot(xfit, function(xfit, *popt), '--', c=colors[i])
                print(name, "fit parameters: ", result_fit_parameters[name], "\n",
                      name, "fit standard deviation errors: ", np.sqrt(np.diag(pcov))
                      )
        if label is True:
            plt.legend(loc=(0, 1.01))  # loc='upper center', bbox_to_anchor=[0.5, 1.13])

    warnings.filterwarnings('default')

    return result_fit_parameters


def draw_H_bar(x1, y1, x2, y2, ax=None, ratio=0.25, height=None, text=None, text_vpos=None, kw_hline={}, kw_vlines={},
               **kwargs):
    """Draw a H bar between the points of coordinates (x1, y1) & (x2, y2).
    
    Parameters
    ----------
    ax : axes on which the H bar will be plotted. Default value is the current axes instance.
    
    ratio : ratio between the length of the vertical bars and the horizontal bar in pixel coordinates.
    
    height : the length of the vertical bars in data coordinates. If provided, ``ratio`` is ignored.
    
    text : if set to ``True`` the width of the H bar in data coordinate will be added. If a string is passed, it will
    be added instead.
    
    Any additional kwargs passed to the function is assigned both to ``kw_hline`` and ``kw_vlines``.
    
    """
    ax = ax if ax is not None else plt.gca()
    # set the common features
    for key, value in kwargs.items():
        if value is not None:
            kw_hline[key] = value
            kw_vlines[key] = value

    y = np.mean([y1, y2])

    direct_transform = ax.transData  # From data to display (pixel) coordinate system.
    inv_transform = ax.transData.inverted()  # From display to data coordinate system.
    # The width in data coordinates of the horizontal line is: width = ``x2 - x1``.
    # We calculate the height in data coordinate so as to have height (in pixels) be ``ratio`` * width (in pixels).
    pixel_coord = direct_transform.transform([(x1, y1), (x2, y2)])
    x1_pix, y1_pix, x2_pix, y2_pix = pixel_coord[0, 0], pixel_coord[0, 1], pixel_coord[1, 0], pixel_coord[1, 1]
    width_in_pixel = np.abs(x1_pix - x2_pix)
    height_in_pixel = ratio * width_in_pixel
    y_pix = np.mean([y1_pix, y2_pix])

    if height is None:
        y_data_vmin = inv_transform.transform((0, y_pix - height_in_pixel / 2))[1]  # vertical min
        y_data_vmax = inv_transform.transform((0, y_pix + height_in_pixel / 2))[1]  # vertical max
    else:
        y_data_vmin, y_data_vmax = y - height / 2, y + height / 2

    # plot lines
    hline_ = ax.hlines(y, x1, x2, **kw_hline)
    vlines_ = ax.vlines([x1, x2], [y_data_vmin] * 2, [y_data_vmax] * 2, **kw_vlines)

    if text is None:
        print('width of the H bar in data coordinate: {}'.format(np.abs(x2 - x1)))
        return [hline_, vlines_]
    elif not text:
        return [hline_, vlines_]
    else:
        x_text = (x1_pix + x2_pix) / 2
        y_text = y_pix - 0.15 * width_in_pixel
        x_text, y_text = inv_transform.transform((x_text, y_text))
        y_text = y_text if text_vpos is None else text_vpos
        if text:
            text_ = ax.text(x_text, y_text, "{}".format(np.abs(x2 - x1)), ha='center', va='top')
        elif isinstance(text, str):
            text_ = ax.text(x_text, y_text, text, ha='center', va='top')
        else:
            raise ValueError('``text`` must be either the boolean value ``True`` or a string.')
        return [hline_, vlines_, text_]


def plot_shift_to_reference(x, y, reference=None, eV=True):
    mpl.style.use('default')
    mpl.rcParams['font.size'] = 16

    reference = x[np.argmax(y)] if reference is None else reference
    x_shifted = (x - reference) * 1000 if eV else (x - reference)

    # let's plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.plot(x_shifted, y)
    ax.set_xlabel('Energy shift in {} from {:.4} {}'.format('meV' if eV else '?', float(reference), 'eV' if eV else ''))
    ax.set_ylabel('Intensity')


def analyse_spectra(x, y, yscale='log', reference=None, eV=True, second_axis=True, H_bar=False, labelrotation=0,
                    remove_last=1, peak_kws={}):
    plot_shift_to_reference(x, y, reference=reference, eV=eV)
    plt.yscale(yscale)
    reference = x[np.argmax(y)] if reference is None else reference

    # search for peaks
    peaks = find_peaks(y, x=x, decimals=3, **peak_kws)
    peaks = peaks[:-remove_last] if remove_last != 0 else peaks[:]             # The last one is removed by default
    peaks_shifted = (peaks - reference) * 1000 if eV else (peaks - reference)  # because it is fake.
    peaks_shifted = np.round(peaks_shifted, decimals=0)

    # let's plot vertical lines at ``peaks``'s positions
    ax = plt.gca()
    ax.vlines(peaks_shifted, 0, np.max(y), linestyle='--', lw=1, alpha=1)
    ax.set_xticks(peaks_shifted)
    ax.tick_params(axis='x', labelrotation=labelrotation)

    if H_bar:
        if not isinstance(H_bar, dict):
            H_bar = {}
        vpos = H_bar.pop('vpos', np.max(y))
        text_vpos = H_bar.pop('text_vpos', None)
        ratio = H_bar.pop('ratio', 0.1)
        # Let's do a complicated stuff just to be able to keep the specific control on colours from ``draw_H_bar``.
        if ('color' not in H_bar.copy().pop('kw_hline', {}).keys()) and \
                ('color' not in H_bar.copy().pop('kw_vlines', {}).keys()):
            H_bar['color'] = H_bar.pop('color', 'tab:red')

        # Let's add an hbar between each peaks.
        for peak1, peak2 in zip(peaks_shifted[:-1], peaks_shifted[1:]):
            draw_H_bar(peak1, vpos, peak2, vpos, text=True, ratio=ratio, text_vpos=text_vpos, **H_bar)

    if second_axis:
        # Create a second x axis to show the energy differences through top ticklabels.
        ax_twin = ax.twiny()
        ax_twin.set_xlim(ax.get_xlim())
        ax_xticks = ax.get_xticks()
        ax_twin_xticks = []
        for tick1, tick2 in zip(ax_xticks[:-1], ax_xticks[1:]):
            ax_twin_xticks.append(np.mean([tick1, tick2]))
        ax_twin.set_xticks(ax_twin_xticks)
        ax_twin_xticklabels = np.asarray(
            [np.abs(peak1 - peak2) for peak1, peak2 in zip(peaks_shifted[:-1], peaks_shifted[1:])])
        ax_twin.set_xticklabels(ax_twin_xticklabels.astype(int))
        ax_twin.tick_params(axis='x', length=0, pad=0, labelrotation=labelrotation)


def discrete_matshow(data):
    """Create a discrete colorbar for a 'data' 2D array."""
    # get discrete colormap
    cmap = plt.get_cmap('RdBu', np.max(data) - np.min(data) + 1)
    # set limits .5 outside true range
    mat = plt.matshow(data, cmap=cmap, vmin=np.min(data) - .5, vmax=np.max(data) + .5)
    # tell the colorbar to tick at integers
    cax = plt.colorbar(mat, ticks=np.arange(np.min(data), np.max(data) + 1))


def normalize(filename, ydata, tacq_position, power_position):
    list_params = filename[:-4].split('_')
    # print(list_params)
    # Get the acquisition time.
    time = list_params[tacq_position]
    # print(time)
    time = time[:-3]  # remove 'sec'
    # print(time)
    items = time.split('x')
    # print(items)
    acq_time = float(items[1])
    # print(total_time)
    # Get the laser power in µW.
    power = list_params[power_position]
    # print(power)
    power_value = float(power[:-2])
    # print(power_value)
    return ydata / (acq_time * power_value)


#######################################################################################################################
################################### FUNCTIONS FOR PANDA DATAFRAME CONSTRUCTION FROM DATAFILES #########################
################################################## AND PANDA RELATED FUNCTIONS ########################################
#######################################################################################################################

def get_parameters(filename):
    """Return a dictionary with all the parameters contained in the file name given, following the established regex.

    The file extension must be removed from the file name.
    
    Parameters looked for in the file name
    --------------------------------------
    temperature : in kelvin
    
    laser_wavelength : in nanometer
    
    power : in micro watt
    
    wavelength : the central wavelength of the spectro in nm
    
    grooves : number of line per mm on the diffraction grating
    
    tacq : acquisition time in seconds
    
    slit : width of the slit in micrometers

    """
    list_params = filename.split('_')
    # Get the parameters.
    temperature = int(list_params[0][:-1])
    sample = list_params[1]
    laser_wavelength = float(list_params[2][5:-2])
    power = float(list_params[3][:-2])
    wavelength = float(list_params[4][:-2])
    grooves = float(list_params[5][:-7])
    time = list_params[6]
    items = time[:-3].split('x')
    number_acq = int(items[0])
    tacq = float(items[1])
    slit = float(list_params[7][4:])
    filter_ = list_params[8]
    calibration = list_params[9]
    try:
        position = list_params[10][:]  # I keep the 'P' before the number, ex: 'P2' for position 2.
    except Exception:
        position = 0
    return {'temperature': temperature, 'sample': sample, 'laser_wavelength': laser_wavelength,
            'power': power, 'wavelength': wavelength, 'grooves': grooves, 'number_acq': number_acq, 'tacq': tacq,
            'slit': slit, 'filter': filter_, 'calibration': calibration, 'position': position}


def get_parameters_key(filename):
    """Return a dictionary with all the parameters contained in the filename indicated by a key.

    Conventions of name have been fixed by a collective choice. Use regular expressions.

    """
    # Find the keys present in the file name.
    # keys_in_filename = re.findall("[a-zA-Z]+(?=\[)", s)  # matches any string that is followed by '['

    # We could create a list for each parameter in which all the name variations of the related key would be listed so
    # that we can normalized the key name by giving it the 'conventional' name. It would be useful to give this code
    # more flexibility.

    dic_key_parameter = {'T': 'temperature', 'S': 'sample_id', 'P': 'power', 'Lwv': 'laser_wv', 'gr': 'grating',
                         'Gwv': 'grating_wv', 'tacq': 'tacq', 'nacq': 'nacq', 'slit': 'slit', 'F': 'filter_id',
                         'calib': 'calib', 'position': 'position', 'dateID': 'dateID', 'kind': 'kind', 'polar': 'polar'}

    value_parameter = {}
    for key in dic_key_parameter:
        pattern = key + r"\[([^\[\]]*?)\]"     # This regex matches any string between [] that do not contain '[' or ']'
        match = re.findall(pattern, filename)  # and it stops at the first string matching the pattern
                                               # (I set the non greedy mode by placing a '?' after '*').
        # if len(match) == 0:
        #     value_parameter[dic_key_parameter[key]] = 0
        if len(match) == 1:
            value_parameter[dic_key_parameter[key]] = match[0]
        elif len(match) > 1:
            for i, value in enumerate(match):
                if re.match(r"\w+", value) is None:
                    del match[i]
            for value in match:
                error = True if value != match[0] else False
            if not error:
                value_parameter[dic_key_parameter[key]] = match[0]
            else:
                raise ValueError(
                    "The file contains two different value of the same parameter. Please correct the mistake.")
    return value_parameter


def make_DataFrame(list_of_filenames, files_location, format_, version='old', unpack=True):
    """
    Return a dataframe. The number of row correponds to the number of filename & the number of columns to the numbers
    of parameters found in the filenames plus the data.
    
    Important: the extension of the file names given are supposed to be removed.
    Warning: this function is made for the MacroPL experiment and the convention taken for the file names.
    It is necessary to adapt this function for other file names, along with the 'get_parameters' function.
    
    => There is an 'old' <version>: adapted when no keys are used in the file name. The order of the parameter written
    in the file name is primordial.
    
    => There is a 'new' <version>: adapted when keys are used in the file name. Much more adaptable, does not depend on
    the order of the parameter, neither on the existence or not of some parameter.
    
    Parameters looked for in the file names
    ---------------------------------------
    temperature : in kelvin
    
    laser_wavelength : in nanometer
    
    power : in micro watt
    
    wavelength : the central wavelength of the spectro in nm
    
    grooves : number of line per mm on the diffraction grating
    
    tacq : acquisition time in seconds
    
    slit : width of the slit in micrometers

    """
    df = pd.DataFrame({'energies': np.zeros(len(list_of_filenames)), 'intensities': np.zeros(len(list_of_filenames))})
    df['energies'] = df['energies'].astype(object)
    df['intensities'] = df['intensities'].astype(object)

    if version == 'old':
        for i in range(len(list_of_filenames)):
            parameters = get_parameters(list_of_filenames[i])
            df.at[i, 'sample_id'] = parameters['sample']
            df.at[i, 'position'] = parameters['position']
            df.at[i, 'wavelength'] = parameters['wavelength']
            df.at[i, 'power'] = parameters['power']
            df.at[i, 'tacq'] = parameters['tacq']
            df.at[i, 'number_acq'] = parameters['number_acq']
            df.at[i, 'laser_wavelength'] = parameters['laser_wavelength']
            df.at[i, 'temperature'] = parameters['temperature']
            df.at[i, 'filter_id'] = parameters['filter']
            df.at[i, 'slit'] = parameters['slit']
            df.at[i, 'grooves'] = parameters['grooves']
            df.at[i, 'calibration'] = parameters['calibration']
            # get spectrum
            x, y = np.loadtxt(files_location + list_of_filenames[i] + '.' + format_, unpack=unpack)
            # sort by ascending order of x's values
            y = y[np.argsort(x)]
            x = np.sort(x)
            df.at[i, 'energies'] = x
            df.at[i, 'intensities'] = y
            # df['position'] = df['position'].astype(int) # I don't do that anymore because I want to keep the 'P'
            # for MultiIndex
        df['number_acq'] = df['number_acq'].astype(int)
        df['temperature'] = df['temperature'].astype(int)
        # re order the DataFrame
        df = df[['sample_id', 'position', 'wavelength', 'power', 'tacq', 'number_acq', 'laser_wavelength',
                 'temperature', 'filter_id', 'slit', 'grooves', 'calibration', 'energies', 'intensities']]

    elif version == 'new':
        for i in range(len(list_of_filenames)):
            parameters = get_parameters_key(list_of_filenames[i])
            for key in parameters:
                df.at[i, key] = parameters[key]
            # get the spectrum
            x, y = np.loadtxt(files_location + list_of_filenames[i] + '.' + format_, unpack=unpack)
            # sort by ascending order of x's values
            y = y[np.argsort(x)]
            x = np.sort(x)
            df.at[i, 'energies'] = x
            df.at[i, 'intensities'] = y
    return df


def list_values_of_parameter(dataframe, parameter):
    """Return a list of all values of the parameter from the dataframe."""
    index_of_parameter = dataframe['{}'.format(parameter)].value_counts().index
    list_of_parameter = []
    for value in index_of_parameter:
        list_of_parameter.append(value)
    return list_of_parameter


def highlight_row_by_name(s, names, color):
    """This function is intented to be called inside ``pandas.DataFrame.style.apply()`.

    'axis=1' must be specified in the call.
    
    Parameters
    ----------
    'names' & 'color' : must be passed as a list in the kwargs.
    
    Examples
    --------
    df.style.apply(highlight_row_by_name, axis=1, names=['mean', 'std'])
    ==> will highlight the rows named 'mean' and 'std'

    """
    if s.name in names:
        return ['background-color: ' + color] * len(s)
    else:
        return ['background-color: white'] * len(s)
