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
import scipy.constants
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



# From Alrik, see "Data_analysis_tools-AlrikDurand/data.py"

def get_window(x, y, a, b):
    """ Very useful method to get just a window [a, b] of a signal (x,y) """
    mask_1 = a < x
    mask_2 = x < b
    mask = np.logical_and(mask_1, mask_2)
    x = x[mask]
    y = y[mask]
    return x, y


def rebin(data, rebin_ratio, do_average=False):
    """ Rebin a 1D array the good old way.

    @param 1d numpy array data : The data to rebin
    @param int rebin_ratio: The number of old bin per new bin

    @return 1d numpy array : The array rebinned

    The last values may be dropped if the sizes do not match.

    """
    data = np.asarray(data)
    rebin_ratio = int(rebin_ratio)
    length = (len(data) // rebin_ratio) * rebin_ratio
    data = data[0:length]
    data = data.reshape(length//rebin_ratio, rebin_ratio)
    if do_average :
        data_rebinned = data.mean(axis=1)
    else :
        data_rebinned = data.sum(axis=1)
    return data_rebinned


def decimate(data, decimation_ratio):
    """ Decimate a 1D array . This means some value are dropped, not averaged

    @param 1d numpy array data : The data to decimated
    @param int decimation_ratio: The number of old value per new value

    @return 1d numpy array : The array decimated

    """
    decimation_ratio = int(decimation_ratio)
    length = (len(data) // decimation_ratio) * decimation_ratio
    data_decimated = data[:length:decimation_ratio]
    return data_decimated


def rebin_xy(x, y,  ratio=1, do_average=True):
    """ Helper method to decimate x and rebin y, with do_average True as default """
    return decimate(x, ratio), rebin(y, ratio, do_average)

# End of : From Alrik


def get_slope(x, y):
    """Calculate slope by taking first and last values."""
    return (y[-1]-y[0])/(x[-1]-x[0])


def get_intercept(x, y, slope):
    """Calculate intercept by taking first value."""
    return y[0] - slope*x[0]


#######################################################################################################################

                                            #############################
              ################################  HANDLE SPIKES IN DATA  ################################
                                            #############################

#######################################################################################################################


"""
def remove_spikes(array, section=None, window_size=10, threshold=200):
    Removes spikes from data contained in ``array`` and return the cleaned array.

    Parameters
    ----------
    array: array-like object.
        Must be of dimension ???

    section : 2-tuple. The spike removal will be performed only inside the limits
        specified by the 2-tuple.

    window_size :

    threshold :

    Return
    ------
        An array of same shape containing data with spikes removed.

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

    if section is None:
        section = np.arange(data.shape[-1] - 1)
    else:
        section = np.arange(section[0], min(section[1], data.shape[-1] - 1))  

    # Selecting spikes based on threshold criterion. Note that spikes dimension 1 is one time smaller.
    spikes = z_score_diff(data) > threshold

    # Replace each spike value by the mean value of neighboring pixels, the number of which is determined
    # by ``window_size``.
    for i, spectrum in enumerate(data):
        for j in section:
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
"""


def z_score_mean(spectrum):
    mean = np.mean(spectrum)
    sigma = np.std(spectrum)
    return (spectrum - mean) / sigma


def modified_z_score(array):
    """This modified z score used the Median Absolute Deviation (MAD)."""
    array = np.array(array)
    median = np.median(array)
    median_absolute_deviation = np.median(np.abs(array - median))
    # 0.6745 is the 0.75th quartile of the standard normal distribution, to which the MAD converges to.
    modified_z_scores = 0.6745 * (array - median) / median_absolute_deviation
    return modified_z_scores


def modified_z_score_diff(array):
    """Apply a modified z score on a differentiated spectrum and return absolute value, from Whitaker and Hayes."""
    return np.abs(modified_z_score(np.diff(array)))


import copy

def spike_fixer(array, threshold=30, wsize=3):
    """Fix an array by detecting spikes with ``modified_z_score_diff`` method and changing value by taking the mean around spike.

    Parameters
    ----------
    array : 1D iterable.

    threshold : integer.
        Defines the threshold for discriminating spikes.

    wsize : integer.
        Defines the half size of the window centered on a spike to calculate a mean value to replace spike value.
        Precisely: (2 * wsize + 1) points around the spike.

    Return
    ------
    Fixed array of same size as input array. The original array is not modified.
    """
    # Checking for correct inputs.
    if not isinstance(array, (list, np.ndarray)):
        return
    if not isinstance(threshold, int) or not isinstance(wsize, int):
        return
    # Detect spikes based on threshold.
    spikes = modified_z_score_diff(array) > threshold
    fixed = copy.deepcopy(array)  # No overwrite of original array
    # Fix spectrum.
    for i, spike in enumerate(spikes):
        if spike:
            window = np.arange(max(0, i - wsize), min(i + 1 + wsize, len(array)), 1)  # Window around the spike.
            window = window[~spikes[window]]  # Filter out spikes from the window.
            fixed[i] = np.mean(array[window])  # Replace the spike value by the mean value of neighboring pixels.
    return fixed


def flattener(fixed_array, spikes_bool, wsize=3, n_neighbors=2):
    flattened = copy.deepcopy(fixed_array)  # No overwrite of original array
    corrected = np.array([False]*len(flattened))
    # Flatten spectrum in region where spike were detected and fixed.
    for i, spike in enumerate(spikes):
        if spike:
            for j in np.arange(max(0, i - n_neighbors), min(i + 1 + n_neighbors, len(fixed_array)), 1):
                window = np.arange(max(0, j - wsize), min(j + 1 + wsize, len(fixed_array)), 1)  # Window around the spike.
                flattened[j] = np.mean(fixed_array[window])  # Replace the value by the mean value of neighboring pixels.
                corrected[j] = True
    return flattened


def spike_fixer_(array, threshold=30, wsize=3):
    """Fix an array by detecting spikes with ``modified_z_score_diff`` method and changing value by taking the mean around spike.

    Parameters
    ----------
    array : 1D iterable.

    threshold : integer.
        Defines the threshold for discriminating spikes.

    wsize : integer.
        Defines the half size of the window centered on a spike to calculate a mean value to replace spike value.
        Precisely: (2 * wsize + 1) points around the spike.

    Return
    ------
    Fixed array of same size as input array and a boolean spikes array. The original array is not modified.
    """
    # Checking for correct inputs.
    if not isinstance(array, (list, np.ndarray)):
        return
    if not isinstance(threshold, int) or not isinstance(wsize, int):
        return
    # Detect spikes based on threshold.
    spikes = modified_z_score_diff(array) > threshold
    fixed = copy.deepcopy(array)  # No overwrite of original array
    # Fix spectrum.
    for i, spike in enumerate(spikes):
        if spike:
            window = np.arange(max(0, i - wsize), min(i + 1 + wsize, len(array)), 1)  # Window around the spike.
            window = window[~spikes[window]]  # Filter out spikes from the window.
            fixed[i] = np.mean(array[window])  # Replace the spike value by the mean value of neighboring pixels.
    return fixed, spikes


def flattener_(fixed_array, spikes_bool, wsize=3, n_neighbors=2):
    flattened = copy.deepcopy(fixed_array)  # No overwrite of original array
    corrected = np.array([False]*len(flattened))
    # Flatten spectrum in region where spike were detected and fixed.
    for i, spike in enumerate(spikes):
        if spike:
            for j in np.arange(max(0, i - n_neighbors), min(i + 1 + n_neighbors, len(fixed_array)), 1):
                window = np.arange(max(0, j - wsize), min(j + 1 + wsize, len(fixed_array)), 1)  # Window around the spike.
                flattened[j] = np.mean(fixed_array[window])  # Replace the value by the mean value of neighboring pixels.
                corrected[j] = True
    return flattened, corrected

#######################################################################################################################
######################################################## CONVERSION ###################################################
#######################################################################################################################

def nm_eV(*args, decimals=5):
    """Return the converted energy from nm to eV or from eV to nm.

    Note: scalars, list and numpy array can be passed to the function but only one type at a time.
    Example: nm_eV(234.6, 1.75, 1.34)
             nm_eV([233, 1.37, 1.39])
             nm_eV(np.array([666, 665, 664]))
             
    """
    if len(args) == 1:
        value = np.array(args[0])
        return np.round(1239.84193 / value, decimals=decimals)
    else:
        values = np.empty(len(args))
        for i, arg in enumerate(args):
            values[i] = np.round((1239.84193 / arg), decimals=decimals)
        return values


def delta_nm2eV(delta_nm, Lambda, decimals=3):
    """Return an approximation of the energy delta in meV.
    It is more precise to calculate a difference in energy by taking the two wavelength in eV.
    
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


"""def nm_cm(x_nm, laser_wavelength):
    "#"Return the converted energy from nm to cm-1 relatively to the wavelength of the laser."#"
    # We need some constants for the conversion.
    PLANK = scipy.constants.Planck
    EV = scipy.constants.eV
    SPEED_OF_LIGHT = scipy.constants.speed_of_light
    # Factor of conversion.
    factor = (PLANK/EV * SPEED_OF_LIGHT*100)**-1
    return (nm_eV(laser_wavelength) - nm_eV(x_nm)) * factor"""
def nm_cm(x_nm, laser_wavelength):
    """Return the converted energy from nm to raman shoft in cm-1 relatively to the wavelength of the laser."""
    x_nm = np.array(x_nm)
    return np.round(10**7*(1/laser_wavelength - 1/x_nm), 6)


def cm_eV(x_cm, laser_wavelength):
    """Return the converted energy from raman shift cm-1 to eV."""
    # We need some constants for the conversion.
    PLANK = scipy.constants.Planck
    EV = scipy.constants.eV
    SPEED_OF_LIGHT = scipy.constants.speed_of_light
    # Factor of conversion.
    factor = (PLANK/EV * SPEED_OF_LIGHT*100)**-1
    return x_cm/factor


"""def cm_nm(x_cm, laser_wavelength):
    "#""Return the converted energy from raman shift cm-1 to nm relatively to the wavelength of the laser."#""
    delta_eV = cm_eV(x_cm, laser_wavelength)
    return nm_eV((nm_eV(laser_wavelength) - delta_eV))"""
def cm_nm(x_cm, laser_wavelength):
    """Return the converted energy from raman shift cm-1 to nm relatively to the wavelength of the laser."""
    x_cm = np.array(x_cm)
    return np.round(1 / ((1/laser_wavelength) - x_cm/10**7), 6)


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

# In all these functions for fitting it is important to keep "A" as the extra counts above "B" the background because
# when giving min and max bounds for fitting, "B" can change and we always want "A" to be that count above "B".
# Otherwise if we use A = A - B and give A as the absolute counts, when B change sometime A can become negative.

def linear(x, a, b):
    return a * x + b


def gaussian(x, sigma, mu, A, B):
    """The area under the gaussian function without the background is given by: A*sigma*np.sqrt(2*np.pi)."""
    return np.exp(-((x - mu) / np.sqrt(2 * sigma ** 2)) ** 2) * A + B


# linear & affine do the same but with different user input, I think that linear is easier to use because one doesn't
# need to calculate the y-intercept ``b``. Correction: this is not true because we have to calculate ``B``, they are actually
# the same functions.

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


# In this formulae: A = 2/(pi*Gamma) , and B is the background
def lorentzian(x, gamma, mu, A, B):
    """A is the maximum of the peak, mu the center, and gamma the FWHM. B is background."""
    return 1 / (1 + (2 * (x - mu) / gamma) ** 2) * A + B

# For lmfit I create another model, different from the built-in lorentzian model
# this is because in the built-in model the parameter ``amp`` is the area under curve
# and it is not easy to evaluate when giving initial values
from lmfit import Model

def lmfit_lorentzian(x, amplitude, center, sigma):
    """``amplitude`` is the maximum of the peak, ``center`` the center and ``sigma`` the FWHM."""
    return 1 / (1 + (2 * (x - center) / sigma) ** 2) * amplitude
def LorentzianModel(prefix=''):
    """
    This is a function that returns the lmfit Model function with the prefix given.
    ``amplitude`` is the maximum of the peak, ``center`` the center and ``sigma`` the FWHM.
    """
    return Model(lmfit_lorentzian, prefix=prefix)


def lorentzian_linear(x, gamma, mu, A, B, a):
    return (1 / (1 + (2 * (x - mu) / gamma) ** 2) * A + B) + (a * x)


def lorentzian_affine(x, gamma, mu, A, a, b):
    return (1 / (1 + (2 * (x - mu) / gamma) ** 2) * A) + (a * x + b)


def double_lorentzian(x, gamma1, mu1, A1, gamma2, mu2, A2, B):
    return (1 / (1 + (2 * (x - mu1) / gamma1) ** 2) * A1) + (1 / (1 + (2 * (x - mu2) / gamma2) ** 2) * A2) + B


def double_lorentzian_linear(x, gamma1, mu1, A1, gamma2, mu2, A2, B, a):
    return (1 / (1 + (2 * (x - mu1) / gamma1) ** 2) * A1) + (1 / (1 + (2 * (x - mu2) / gamma2) ** 2) * A2) + B + (a * x)

def triple_lorentzian(x, gamma1, mu1, A1, gamma2, mu2, A2, gamma3, mu3, A3, B):
    return (1 / (1 + (2 * (x - mu1) / gamma1) ** 2) * A1) + (1 / (1 + (2 * (x - mu2) / gamma2) ** 2) * A2) + (1 / (1 + (2 * (x - mu3) / gamma3) ** 2) * A3) + B

def triple_lorentzian_linear(x, gamma1, mu1, A1, gamma2, mu2, A2, gamma3, mu3, A3, B, a):
    return (1 / (1 + (2 * (x - mu1) / gamma1) ** 2) * A1) + (1 / (1 + (2 * (x - mu2) / gamma2) ** 2) * A2) + (1 / (1 + (2 * (x - mu3) / gamma3) ** 2) * A3) + B + (a * x)

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


def fit_polarization(x, A, B, phi, C):
    """(A-C) * np.cos(B*x+phi)**2 + C"""
    return (A-C) * np.cos(B*x+phi)**2 + C


def differential_reflectance(signal, reference):
    """Return '(signal - reference) / signal'."""
    return (signal - reference)/(signal)


def contrast_reflectance(signal, reference):
    """Return '(signal - reference) / (signal + reference)'."""
    return (signal - reference)/(signal + reference)


def exponential_negative(x, amplitude=1, decay=1, center=0):
    cut_index = np.searchsorted(x, center)
    if cut_index == len(x):
        return np.zeros(len(x))
    elif cut_index == 0:
        return amplitude * np.exp(-x/decay)
    else:
        first_part = np.zeros(cut_index)
        second_part = amplitude * np.exp(-(x[cut_index:]-center)/decay)
        return np.concatenate((first_part, second_part))


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
    """This void function is used to issue a deprecation warning."""
    raise DeprecationWarning("This function name has changed, use ``get_filenames`` instead.")
def get_filenames(loc, format_='txt', keyword=None, print_name=True, sort_by_date=True, sort_by_end=False, reverse=False):
    """Return a list of all filenames at specified location matching with the given format and keyword.
    
    Parameters
    ----------
    loc : path to the folder location.

    format_ : format of the searched files. (default: 'txt')
        Ex: can be with or without the point. '.txt' and 'txt' are both valid.
    
    keyword : string or sequence of string to match in the filenames.

    print_name : print the name of the files in the standard output.

    sort_by_date : sort by the date at the end of the filename
        It is assumed that the filename end in "201104_01" for the 1st file of the 4th of November 2020.

    sort_by_end : sort the filenames by the end of the string.

    reverse : If ``False`` sort by ascending order. If ``True`` sort by descending order.

    Return a numpy array.

    """
    # we choose the current directory where all the files are located
    os.chdir(loc)
    # Remove the '.' in ``format_``.
    if '.' in format_:
        format_ = format_[1:]
    # glob return a list with all the files with format_
    all_filenames = glob.glob("*.{}".format(format_), recursive=False)
    filenames = []
    if not keyword:
        filenames = all_filenames
    else:
        keyword = [keyword] if not isinstance(keyword, (list, tuple, np.ndarray)) else keyword
        for file in all_filenames:
            if all(map(lambda word: word in file, keyword)):
                filenames.append(file)
    length = len('.' + format_)
    for i, file in enumerate(filenames):
        # remove the specified format at the end of the file and the preceding point
        filenames[i] = file[:-length]
    if sort_by_date and sort_by_end:
        raise ValueError("Only one type of sorting is allowed")
    if sort_by_date:
        filenames = sorted(filenames, key=lambda s: s[-9:], reverse=reverse)
    if sort_by_end:
        filenames = sorted(filenames, key=lambda s: s[::-1], reverse=reverse)
    if not sort_by_date and not sort_by_end:
        # I sort by the beginning.
        filenames = sorted(filenames, reverse=reverse)
    if print_name:
        for i, file in enumerate(filenames):
            print('({:})'.format(i), file)
        print("\n")
    return np.array(filenames)


def get_last_filename_number_at_location(loc='/Users/pelini/L2C/Manip uPL UV 3K/Data/analysis/images en bazar/',
                                         keyword='image_', fmt='pdf'):
    """Return the last number of numbered filename.
    
    Parameters
    ----------
    loc     : the location of the folder to search in
    keyword : the pattern to filter filenames
    fmt     : the format of the files

    """
    filenames = get_filenames(loc, fmt, keyword=keyword, print_name=False)
    numbers = np.empty(len(filenames), dtype=np.int8)
    for i, file in enumerate(filenames):
        number = re.findall(r'\d+', file)
        numbers[i] = int(number[0]) if len(number) > 0 else 0
    numbers = np.sort(numbers)
    if len(numbers) != 0:
        return numbers[-1]
    else:
        return int(0)


def save_image_as_textfile(image: mpl.image.AxesImage, filename, comment=None):
    """
    Saves an image generated by matplotlib.pyplot ``imshow`` method as a one column textfile
    (the data is flatten for that purpose). One can add a comment at the beggining of the file
    to specify the shape of the image --> example: comment='Image 50x50'.
    """
    data = image.get_array().data.ravel()
    np.savetxt(filename, data, header=comment)


def get_files_by_id(list_of_files, date, numbers):
    """
    Select and return filenames specified by ``date`` and ``numbers`` out of ``list_of_files``.

    Parameters
    ----------
    list_of_files : iterable of strings.
        List of filenames in which to search.

    date : string.
        The date for all ids seeked out. Must have the form ``210426`` for April 26th of 2021.

    numbers: iterable of strings/integers.
        The number identifiers of the seeked files. Ex: [01, 02, 10, 23], or ['01', '02', '10', '23'].
    """
    files = [file for file in list_of_files if len(file.split('_')) > 5]  # Filter file that are not pure datafile saved during experiment
    new_list = []
    for file in files:
        if file.split('_')[-2] == date and len(file.split('_')[-1]) == 2 and int(file.split('_')[-1]) in numbers:
            new_list.append(file)
    return new_list


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
                  bounds=None, **fit_functions):
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

    label: bool, None. Whether to plot the legend with the fitted parameters (True), don't plot but print them (False),
        or don't plot them in the legend neither print them (None).
    
    initial_guess : Must be a dictionnary.
        key   : name of the fit function
        value : list of guess values for the parameters of the fit function

    bounds : 2-tuple of array-like or scalar.
        If array: must have length equal to the number of parameters.
        If scalar: same bound for each parameter.
        If 'None' defaults to no bounds (-np.inf, np.inf).
                        
    fit_functions : the 'key' must be the name of the fit function and the 'value' the proper function. 
        Example:  gaussian=fct.gaussian, lorentzian_linear=fct.lorentzian_linear

    Return a 2-tuple of dictionnary. The first one is ``result_fit_parameters`` in which the key indicates
    the fit function and the value is an array of fitted parameters. The second one is ``result_fit_parameters_err``
    for that return "np.sqrt(np.diag(pcov))" where pcov is the covariance matrix return by ``scipy.optimize.curve_fit``.

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
    x = np.array(x) 
    y = np.array(y)
    xfit = np.linspace(np.nanmin(x), np.nanmax(x), 10 * len(x))

    plot_function = plt.plot if plot_function is None else plot_function
    if plot_function == plt.plot:
        plot_function(x, y, c=colors[0], **kwarg_for_plot_function)
    else:
        plot_function(x, y, **kwarg_for_plot_function)

    import warnings
    warnings.filterwarnings("error")

    bounds = (-np.inf, np.inf) if bounds is None else bounds
    result_fit_parameters = {}
    result_fit_parameters_err = {}
    for i, (name, function) in enumerate(fit_functions.items(), 1):
        if (initial_guess is not None) and (name in initial_guess.keys()):
            try:
                popt, pcov = curve_fit(function, x, y, p0=initial_guess[name], bounds=bounds)
                result_fit_parameters[name] = popt
                result_fit_parameters_err[name] = np.sqrt(np.diag(pcov))
            except RuntimeError:
                print("Fit with {} function did not succeed, try to provide better initial guess.".format(name))
                result_fit_parameters[name] = None
                result_fit_parameters_err[name] = None
        elif name in ['gaussian', 'lorentzian']:
            wide = 0.1
            mu = x[np.argmax(y)]
            B = np.mean([y[0], y[-1]])
            A = np.max(y) - B
            try:
                popt, pcov = curve_fit(function, x, y, p0=[wide, mu, A, B], bounds=bounds)
                result_fit_parameters[name] = popt
                result_fit_parameters_err[name] = np.sqrt(np.diag(pcov))
            except RuntimeError:
                print("Fit with {} function did not succeed, try to provide initial guess.".format(name))
                result_fit_parameters[name] = None
                result_fit_parameters_err[name] = None
        elif name in ['gaussian_linear', 'lorentzian_linear']:
            wide = 1  # I have to find a way to let the user define this, as it is sometime needed to use 'bounds'
                        # but not 'initial_guess', but the value 'wide' is out of 'bounds' and is the only one we want to change.
            mu = x[np.argmax(y)]
            B = np.mean([y[0], y[-1]])
            A = np.max(y) - B
            a = (y[-1] - y[0]) / (x[-1] - x[0])
            try:
                popt, pcov = curve_fit(function, x, y, p0=[wide, mu, A, B, a], bounds=bounds)
                result_fit_parameters[name] = popt
                result_fit_parameters_err[name] = np.sqrt(np.diag(pcov))
            except RuntimeError:
                print("Fit with {} function did not succeed, try to provide initial guess.".format(name))
                result_fit_parameters[name] = None
                result_fit_parameters_err[name] = None
        elif name == 'linear':
            a = (y[-1] - y[0]) / (x[-1] - x[0])
            b = y[0] - a * x[0]
            try:
                popt, pcov = curve_fit(function, x, y, p0=[a, b], bounds=bounds)
                result_fit_parameters[name] = popt
                result_fit_parameters_err[name] = np.sqrt(np.diag(pcov))
            except RuntimeError:
                print("Fit with {} function did not succeed, try to provide initial guess.".format(name))
                result_fit_parameters[name] = None
                result_fit_parameters_err[name] = None
        else:
            raise ValueError("An initial guess for the fit function `{}` must be provided".format(name))

        if result_fit_parameters[name] is not None:           
            if label is None:
                ax.plot(xfit, function(xfit, *popt), '--', c=colors[i])
            elif label:
                ax.plot(xfit, function(xfit, *popt), '--', c=colors[i], label=make_label(name, popt))
                print(name, 'fit standard deviation errors: ', np.sqrt(np.diag(pcov)))
            elif not label:
                ax.plot(xfit, function(xfit, *popt), '--', c=colors[i])
                print(name, "fit parameters: ", result_fit_parameters[name], "\n",
                      name, "fit standard deviation errors: ", np.sqrt(np.diag(pcov)),
                      )
    if label:
        plt.legend(loc=(0, 1.01))  # loc='upper center', bbox_to_anchor=[0.5, 1.13])
            
    warnings.filterwarnings('default')
    return result_fit_parameters, result_fit_parameters_err


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


def second_axis(ax, which='x'):
    """Create a second x and/or y axis in eV if initial scaling is in nm, or the other way around.
    
    Parameters
    ----------
    ax : axes to which add the second axis
    
    which : 'x', 'y', or 'both'
    
    Return the new axes instance, either a single axes or a list of two axes.
    
    """
    if which == 'x':
        axtwin = ax.twiny()
        axtwin.xaxis.set_major_locator(ax.xaxis.get_major_locator())
        axtwin.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: round(1239.84193/x, 3)))
        axtwin.set_xbound(ax.get_xbound())
        #axtwin.tick_params(axis='x', which='major', direction='in', length=4.3, width=1.2)
        axtwin.set_xlabel("[eV]", labelpad=10)
        new_axis = axtwin
    elif which == 'y':
        axtwin = ax.twinx()
        axtwin.yaxis.set_major_locator(ax.yaxis.get_major_locator())
        axtwin.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: round(1239.84193/y, 3)))
        axtwin.set_ybound(ax.get_ybound())
        #axtwin.tick_params(axis='x', which='major', direction='in', length=4.3, width=1.2)
        axtwin.set_ylabel("[eV]", labelpad=10)
        new_axis = axtwin
    elif which == 'both':
        # x
        axtwin_x = ax.twiny()
        axtwin_x.xaxis.set_major_locator(ax.xaxis.get_major_locator())
        axtwin_x.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: round(1239.84193/x, 3)))
        axtwin_x.set_xbound(ax.get_xbound())
        axtwin_x.set_xlabel("[eV]", labelpad=10)
        # y
        axtwin_y = ax.twinx()
        axtwin_y.yaxis.set_major_locator(ax.yaxis.get_major_locator())
        axtwin_y.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: round(1239.84193/y, 3)))
        axtwin_y.set_ybound(ax.get_ybound())
        axtwin_y.set_ylabel("[eV]", labelpad=10)
        # new axis
        new_axis = [axtwin_x, axtwin_y]
    # We set the current axes in pyplot back to the original one
    plt.sca(ax)
    return new_axis


from pywinspec import winspec
def plot(file: str, folder: str=None, ax=None, figsize: tuple=(8, 6), transf_func=None, **kwargs):
    """Plot a single file of data by taking the first two columns. Can be .txt or .SPE files. ``transf_func`` is a
    transformation for the x values. kwargs are passed to plt.plot()"""
    def load_txt(folder, file):
        x, y, *_ = np.loadtxt(folder+file, unpack=True)
        return x, y
    def load_spe(folder, file):
        datafile = winspec.SpeFile(folder+file)
        x, y = datafile.xaxis, datafile.data
        y = y.reshape(max(y.shape))
        return x, y

    if folder is None:
        if file.startswith('/'):
            folder = ""
        else:
            folder = "./"
    elif not folder.endswith('/'):
        folder += '/'
    # Extract the data
    if file.endswith('.txt'):
        x, y = load_txt(folder, file)
    elif file.endswith('.SPE'):
        x, y = load_spe(folder, file)
    else:
        try:
            x, y = load_txt(folder, file+'.txt')
        except (FileNotFoundError, OSError):
            try:
                x, y = load_spe(folder, file+'.SPE')
            except (FileNotFoundError, OSError):
                raise ValueError('Only .txt or .SPE file can be processed.')

    if transf_func is not None:
        x = transf_func(x)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax
    ax.plot(x, y, **kwargs)


#######################################################################################################################
################################### FUNCTIONS FOR PANDA DATAFRAME CONSTRUCTION FROM DATAFILES #########################
################################################## AND PANDA RELATED FUNCTIONS ########################################
#######################################################################################################################

def get_parameters(filename):
    """Return a dictionary with all the parameters contained in the filename given, following the established regex.

    The file extension must be removed from the filename.
    
    Parameters looked for in the filename
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
    # Find the keys present in the filename.
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
    
    Important: the extension of the filenames given are supposed to be removed.
    Warning: this function is made for the MacroPL experiment and the convention taken for the filenames.
    It is necessary to adapt this function for other filenames, along with the 'get_parameters' function.
    
    => There is an 'old' <version>: adapted when no keys are used in the filename. The order of the parameter written
    in the filename is primordial.
    
    => There is a 'new' <version>: adapted when keys are used in the filename. Much more adaptable, does not depend on
    the order of the parameter, neither on the existence or not of some parameter.
    
    Parameters looked for in the filenames
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


# I made the ones below during postdoc at Grenoble

def df_from_files(folder, files, fmt='txt', columns=['x', 'y']):
    """Computes a dataframe from two-columns data files, for each file in ``files``.

    Each row corresponds to a datafile, each column to a type of data, each cell to the list of data.
    
    Parameters
    ----------
    folder : complete path to folder. To ignore this parameter just put ``""`` as value, for example
        when the full path is contained in the file name already.
    
    files : sequence of file names to extract data from.
    
    fmt : format of the data file.
    
    columns : column name to use for each column
    
    Return a pandas dataframe.
    
    """
    xx, yy = [], []
    for file in files:
        if fmt is None:
            x, y = np.loadtxt(folder+file, unpack=True, comments='Frame')  # for trivista files
        else:
            x, y = np.loadtxt(folder+file+'.'+fmt, unpack=True, comments='Frame')
        xx.append(x)
        yy.append(y)
        
    df = pd.DataFrame(data={columns[0]: xx, columns[1]: yy})
    # Alternative way:
    #df.insert(0, 'x', xx)
    #df.insert(1, 'y', yy)
    return df

from collections import abc

def df_from_bsweep(folder, file, B_init=None, B_final=None, step=None, bvalues=None, CCD_nb_pixel=1340):
    """Computes a dataframe from a file associated to a sweep in magnetic field and sort dataframe by B values.

    The file should contain several spectra at different field.

    Parameters
    ----------
    folder: complete path to folder.

    file: name of the file. Can be a list of files.

    B_init, B_final: value of initial and final magnetic field.

    step: the step in magnetic field in the sweep.

    bvalues: list of magnetic field values.
        Use this if the data file is a set of spectra from irregular values of magnetic field.
        Ex: 0T, 1T, 2T, 3T, 4T, 6T, 10T, 15T, 20T --> bvalues = [0, 1, 2, 3, 4, 6, 10, 15, 20].

    CCD_nb_pixel: number of pixels on the CCD. default to 1340, standard at LNCMI-G.

    """
    if all((element is not None for element in [B_init, B_final, step])):
        number_of_steps = int(abs((B_final - B_init)/step) + 1)
        B_values = np.linspace(B_init, B_final, number_of_steps)
    elif bvalues is not None:
        number_of_steps = len(bvalues)
        B_values = bvalues
    else:
        raise ValueError("Set either ``B_init``,``B_final`` and ``step` values or provide a ``bvalues`` list.")
    # Extract the spectra.
    if isinstance(file, str):
        col_1, col_2 = np.loadtxt(folder+file+'.txt', unpack=True, comments='Frame')
    elif isinstance(file, abc.Iterable):
        xx, yy = [], []
        for f in file:
            col_1, col_2 = np.loadtxt(folder+f+'.txt', unpack=True, comments='Frame')
            xx.append(col_1)
            yy.append(col_2)
        col_1, col_2 = np.concatenate(xx), np.concatenate(yy)

    col_1 = np.reshape(col_1, (number_of_steps, CCD_nb_pixel))
    col_2 = np.reshape(col_2, (number_of_steps, CCD_nb_pixel))
    # Create the list of the different quantities.
    wavelength = [list(x) for x in col_1]
    intensity = [list(y) for y in col_2]
    energy = [list(nm_eV(x)) for x in wavelength]

    df = pd.DataFrame(data={'B': B_values, 'wavelength': wavelength, 'energy': energy, 'intensity': intensity})
    df.sort_values(by='B', ascending=True, inplace=True, ignore_index=True)
    return df


def df_from_sweep(folder, file, parameter_name, init_val=None, final_val=None, step=None, CCD_nb_pixel=1340):
    """Computes a dataframe from a file associated to a sweep of any parameter.

    The file should contain several spectra at different value of the parameter, each value
    separated by a constant step. It is possible to specify only several parameters or none of them, they will be
    infered from the specified parameters or arbitrary defined.

    Parameters
    ----------
    folder: complete path to folder.

    file: name of the file.

    parameter_name: name of the parameter. Ex: B (for magnetic field), V (for voltage).

    init_val, final_val: value of initial and final parameter.

    step: the step between each value of the parameter.

    CCD_nb_pixel: number of pixels on the CCD. default to 1340, standard at LNCMI-G. Also 1024 for InGaAs camera.
    
    """
    # Extract the spectra.
    col_1, col_2 = np.loadtxt(folder+file+'.txt', unpack=True, comments='Frame')

    number_of_steps = int(np.floor(len(col_1)/CCD_nb_pixel))
    #number_of_steps = int(abs((final_val - init_val)/step) + 1)
    step = 1 if step is None else step
    if init_val is not None and final_val is None:
        init_val = 0 if init_val is None else init_val
        if final_val is None:
            final_val = init_val + (number_of_steps-1)*step
    elif final_val is not None and init_val is None:
        final_val = 0 if final_val is None else final_val
        if init_val is None:
            init_val = final_val - (number_of_steps-1)*step
    elif init_val is None and final_val is None:
        init_val = 0
        final_val = init_val + (number_of_steps-1)*step 

    param_values = np.linspace(init_val, final_val, number_of_steps)
        
    # Reshape.
    col_1 = np.reshape(col_1, (number_of_steps, CCD_nb_pixel))
    col_2 = np.reshape(col_2, (number_of_steps, CCD_nb_pixel))
    # Create the list of the different quantities.
    wavelength = [list(x) for x in col_1]
    intensity = [list(y) for y in col_2]
    energy = [list(nm_eV(x)) for x in wavelength]

    df = pd.DataFrame(data={parameter_name: param_values, 'wavelength': wavelength, 'energy': energy, 'intensity': intensity})
    df.sort_values(by=parameter_name, ascending=True, inplace=True, ignore_index=True)
    return df


def plot_df(df, columns=None, row_selection=None, range_selection=None, ax=None, transf_func=None, **kwargs):
    """A simple routine to plot data from dataframe by selecting rows with ``selection``.
    
    Paramaters
    ----------
    df : dataframe

    columns: list or tuple of integers/strings.
        The two index or label corresponding to the two columns to plot against each other.
    
    row_selection : list or slice of indices of rows.

    range_selection : determines the range in x axis to plot the data. Two ways:
        - list of min and max values in x axis unit. Ex: [700, 720] in nm.
            This works correctly only if all data sets have the same wavelength range.
            Use manually 'plot.xlim()' if this is not the case.
        - slice of index range. Ex: slice(300, 1500) plot between the 600th and 1500th pixel.

    ax : axes on which to plot. Create a new figure and axes if None.

    transf_func : function that apply a transformation on ydata before plotting.

    **kwargs: all additional keyword arguments are passed to the ``plt.plot`` call.
    
    """
    # Defining default value for parameters if unspecified.
    row_selection = slice(None) if row_selection is None else row_selection
    range_selection = slice(None) if range_selection is None else range_selection
    columns = (0, 1) if columns is None else columns
    # Defining the method to access data in dataframe depending on given index or label passed to ``columns``.
    serie_1 = df.iloc[row_selection, columns[0]] if isinstance(columns[0], int) else df.loc[row_selection, columns[0]]
    serie_2 = df.iloc[row_selection, columns[1]] if isinstance(columns[1], int) else df.loc[row_selection, columns[1]]
    # I reset the index of the pandas series so I can adress them by index starting at 0.
    serie_1.reset_index(drop=True, inplace=True)
    serie_2.reset_index(drop=True, inplace=True)
    # If min and max value ox x axis are given for plotting data, we create a slice object
    # with the corresponding index values.
    if isinstance(range_selection, (list, np.ndarray, tuple)):
        min_x = find_nearest(serie_1[0], range_selection[0])[0]
        max_x = find_nearest(serie_1[0], range_selection[1])[0]
        range_selection = slice(min_x, max_x)
    # Let's plot !
    if transf_func is None:
        transf_func = lambda x: x
    if ax == None:
        fig, ax = plt.subplots(figsize=(10, 6))
    for (x, y) in zip(serie_1, serie_2):
        ax.plot(x[range_selection], transf_func(y[range_selection]), **kwargs)


def df_extract(df, **kw_col_values):
    """
    Simple way to get a specific set of rows from dataframe by specifying column values.
    Return a new dataframe containing only the rows that match the specified column values.
    
    Parameters
    ----------
    df : dataframe
    
    kw_col_values : Conditions on column values to select wanted rows of data.
    
    Example
    -------
    1.
    data = df_extract(df, B=12, V=[-4, 4])
    2.
    conditions = {'V': 1, 'B': [0, 2], 'T': 4}
    data = df_extract(df, **conditions)
    
    """
    new_df = df.copy(deep=True)
    for (key, value) in kw_col_values.items():
        if isinstance(value, (list, np.ndarray)):
            new_df = new_df[new_df[key].isin(value)]
        else:
            new_df = new_df[new_df[key]==value]
    return new_df


def df_savetxt(df, columns, location_and_name, fmt='%f'):
    """Save 2 columns from a dataframe to a text file. 
    
    Parameters
    ----------
    df : dataframe
    
    columns : list, sequence.
        The name of the two columns to be saved in the text file.
        
    location_and_name : full path and name for the text file.
    
    fmt : format for writing the data in the text file. Default is float.
    
    Example
    -------
    df_savetxt(df, ['wavelength', 'intensity'], '/path/to/your/folder/filename.txt')
    
    """
    xx = np.asarray([wave for wave in df[columns[0]]]).ravel()
    yy = np.asarray([PL for PL in df[columns[1]]]).ravel()
    np.savetxt(location_and_name, np.column_stack((xx, yy)), fmt=fmt)
    

#######################################################################################################################
############################################# LMFIT HELPER FUNCTIONS ##################################################
#######################################################################################################################

def get_popt(result_fit, prefix=None):
    """Helper function to use with lmfit package.

    Extract the parameters value from a result of a fit with a lmfit model.
    With the function curve_fit of the module scipy.optimize, one is use to retrieve the optimal values of
    a parameters with 'popt, pcov' and use them to plot the result. In lmfit, it is not available directly,
    so this functions address this issue.

    Parameters
    ----------
    result_fit : ``ModelResult``
        The object returned by the method 'model.fit'

    prefix : str
        The prefix given for a single function in the model to fit.


    Return
    ------

    Dictionary of parameter name and value.

    """
    #params_value = {}
    #number_params = len(result_fit.model.param_names)
    #for (pname, param) in result_fit.params.items():
    #    params_value[pname] = param.value
    #params_value = {key: value for i, (key, value) in enumerate(params_value.items()) if i < number_params}
    params_value = result_fit.best_values
    if prefix is not None:
        params_value = {
            key[len(prefix):]: value for (key, value) in params_value.items() if key.startswith(prefix)
        }
    return params_value

#####
# The two functions below I don't use them anymore
def get_param_value(name_param, list_of_results):
    """list of results is a list of lmfit ``ModelResult``."""
    values = []
    stderrs = []
    for result in list_of_results:
        values.append(result.params[name_param].value)
    return values

def get_param_stderr(name_param, list_of_results):
    stderrs = []
    for result in list_of_results:
        stderrs.append(result.params[name_param].stderr)
    return stderrs
#####


from collections.abc import Iterable

def lmfit_get_values(model_result, parameters) -> list:
    if isinstance(parameters, Iterable):
        values = []
        for param in parameters:
            val = model_result.params[param].value
            values.append(val)
    elif isinstance(parameters, str):
        values = [model_result.params[parameters].value]
    return values

def lmfit_get_stderrors(model_result, parameters) -> list:
    if isinstance(parameters, Iterable):
        values = []
        for param in parameters:
            val = model_result.params[param].stderr
            values.append(val)
    elif isinstance(parameters, str):
        values = [model_result.params[parameters].stderr]
    return values

def lmfit_savetxt(list_of_result, list_of_variable, params_to_save, path_and_filename, name_variable=None, save_errors=False, **kwargs):
    """Save to a textfile the selected parameters from lmfit fits.

    Parameters
    ----------
    list_of_result : Iterable[ModelResult]
        Iterable (list) of lmfit result from fit.

    list_of_variable : list
        List of the variable values.
        Example: 10 fit for magnetic field from 1 to 10T, list_of_variable = np.arange(1, 11, 1).

    params_to_save : list
        List of choosen parameters from the fit to be saved.
        Example: A two lorentzian function is used and I want only the center of the lorentzian:
            params_to_save = ['p1center', 'p2center'].

    path_and_filename : str
        Path and name of the file to be saved. If the one wants to save in current directory just give a name.

    name_variable : str
        The name of the variable that changes between each result of ``list_of_result``.
        Example: 10 fit for magnetic field from 1 to 10T, variable = 'B'.

    save_errors : bool
        Weither or not to save the standard errors for each parameter.

    **kwargs : will be passed to ``numpy.savetxt``

    """

    values = [lmfit_get_values(result, params_to_save) for result in list_of_result]  # one list of all peaks value for each pressure
    values = np.transpose(values)  # one list of value for each pressure for each peak

    if save_errors:
        stderrors = [lmfit_get_stderrors(result, params_to_save) for result in list_of_result]
        stderrors = np.transpose(stderrors)
        data_to_save = tuple([list_of_variable]) + tuple([arr for sublist in zip(values, stderrors) for arr in sublist])
        data_to_save = np.column_stack(data_to_save)
    else:
        data_to_save = tuple([list_of_variable]) + tuple(values)
        data_to_save = np.column_stack(data_to_save)

    if not save_errors:
        header = ("{var} , " + ("{} , "*len(params_to_save))[:~2]).format(var=name_variable, *params_to_save)
    else:
        header = ("{var} , " + ("{} , stderr_{} , "*len(params_to_save))[:~2]).format(var=name_variable, *np.concatenate([[item]*2 for item in params_to_save]))

    path_and_filename = path_and_filename if path_and_filename.endswith('.txt') else path_and_filename + '.txt'
    np.savetxt(path_and_filename, data_to_save, header=header, fmt='%10f')




