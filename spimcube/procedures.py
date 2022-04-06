import math as math
import numpy as np
import re
import pandas as pd

import indev.functions as fct

def initialization(path, basename):
    """Return a dictionary with: NStepsX, NStepsY, Npixel, Matrix, tab_of_lambda, Xstep, Ystep, Xrange, Yrange."""
    
    # create the complete file name with extension
    
    data   = path + basename + ".dat"
    Lambda = path + basename + ".lambda"
    init   = path + basename + ".ini"


    #___load 'data' file, extract the first three values and create a list without them___

    Rawfile = np.fromfile(data, dtype='>i4', count=-1, sep="")

    NStepsX = Rawfile[0]
    NStepsY = Rawfile[1]
    CCD_nb_pixels  = Rawfile[2]

    Data = Rawfile[3:]

    #___create a 3D matrix in desired dimensions and fill it with the elements from the 'Data' binary file___

    Matrix = np.reshape(Data, (NStepsY, NStepsX, CCD_nb_pixels))
    #normalization
    #Matrix = Matrix / np.amax(Matrix)

    #___load Lambda file and convert in float___

    LAMBDA  = np.loadtxt(Lambda, dtype=np.str)
    LAMBDAf = [0 for index in range(len(LAMBDA))]
    for index in range(len(LAMBDA)):
        LAMBDAf[index] = float(LAMBDA[index].replace("," , "."))

    #___generate X & Y 1D array of scanned positions___

    Xstep  = fct.get_value("PasX", init)
    Ystep  = fct.get_value("PasY", init)
    Xrange = np.linspace(Xstep, Xstep*NStepsX, NStepsX, dtype='>f4') # building 1D array of X position
    Yrange = np.linspace(Ystep, Ystep*NStepsY, NStepsY, dtype='>f4') # and Y position in µm
    
    return {'NStepsX':NStepsX, 'NStepsY':NStepsY ,'Npixel':CCD_nb_pixels, 'Matrix': Matrix, 'tab_of_lambda':LAMBDAf, 'Xstep':Xstep, 'Ystep':Ystep, 'Xrange':Xrange, 'Yrange':Yrange}

def initialization_winspec32(path, basename, NStepsX=None, NStepsY=None, Xstep=None, Ystep=None, CCD_nb_pixels=1340): 
    """
    Return a dictionary with: NStepsX, NStepsY, Npixel, Matrix, tab_of_lambda, Xstep, Ystep, Xrange, Yrange
    Use Regex for NStepsX/NStepsX/Xstep/Ystep providing that the keywords 'steps' and 'mum' are used in the filename.
    """
    
    filename = path + basename + ".txt"
    
    if NStepsX is None or NStepsY is None:
        pattern = r"(\w*)x(\w*)" + 'steps' # specific to how we write the file name
        values = re.findall(pattern, basename)
        NStepsX = int(values[0][0])
        NStepsY = int(values[0][1])
        
    if Xstep is None or Ystep is None:
        pattern = r"(\w*)x(\w*)" + 'mum'   # specific to how we write the file name
        values = re.findall(pattern, basename)
        Xstep = int(values[0][0])
        Ystep = int(values[0][1])
    
    Xrange = np.linspace(Xstep, Xstep*NStepsX, NStepsX, dtype='>f4') # building 1D array of X position
    Yrange = np.linspace(Ystep, Ystep*NStepsY, NStepsY, dtype='>f4') # and Y position in µm

    column_1, column_2 = np.loadtxt(filename, unpack=True)

    tab_of_lambda = column_1[0:CCD_nb_pixels]
    
    Matrix = column_2.reshape(NStepsX, NStepsY, CCD_nb_pixels)

    return {'NStepsX':NStepsX, 'NStepsY':NStepsY ,'Npixel':CCD_nb_pixels, 'Matrix': Matrix, 'tab_of_lambda':tab_of_lambda, 'Xstep':Xstep, 'Ystep':Ystep, 'Xrange':Xrange, 'Yrange':Yrange}
    

def define_space_range(Xmin, Xmax, Ymin, Ymax, RangeXY):
    """Return a dictionary with: Xpmin, Xpmax, Ypmin, Ypmax."""
    
    #___provide pixel numbers and corresponding values for plotting image in the desired space area___
    
    SpaceArea = [Xmin, Xmax, Ymin, Ymax]  # define plot area in µm
    PixelArea = [Xpmin, Xpmax, Ypmin, Ypmax] = np.zeros((4, 2), dtype='>f4') # define plot area in pixel

    for i, parameter in enumerate(SpaceArea):
        PixelArea[i] = fct.find_nearest(RangeXY[0 if i<2 else 1], parameter) #function defined in package.functions
    
    return {'Xpmin':Xpmin, 'Xpmax':Xpmax, 'Ypmin':Ypmin, 'Ypmax':Ypmax}


def initialization_Bsweep(path, basename, B_init, B_final, B_step, CCD_nb_pixels=1340):
    """Return a dictionary with: Matrix, tab_of_lambda, B_range, B_step, Npixel."""
    
    filename = path + basename + ".txt"
    
    number_Bsweep = int(np.rint((B_final - B_init) / B_step + 1))
    
     # building 1D array of B values

    column_1, column_2 = np.loadtxt(filename, unpack=True)

    tab_of_lambda = column_1[0:CCD_nb_pixels]
    
    Matrix = column_2.reshape(number_Bsweep, CCD_nb_pixels)

    return {'Matrix': Matrix, 'tab_of_lambda':tab_of_lambda, 'B_range':B_range, 'B_step':B_step, 'Npixel':CCD_nb_pixels}


def df_from_bsweep(folder, file, B_init, B_final, step, CCD_nb_pixels=1340):
    """
    Return a dataframe from a sweep in magnetic field, i.e from a file containing several spectra at different field.
    """
    number_of_steps = int((B_final - B_init)/step + 1)
    B_values = np.linspace(B_init, B_final, number_of_steps, dtype='>f4') 
    # Extract the spectra.
    col_1, col_2 = np.loadtxt(folder+file+'.txt', unpack=True)
    col_1 = np.reshape(col_1, (number_of_steps, CCD_nb_pixels))
    col_2 = np.reshape(col_2, (number_of_steps, CCD_nb_pixels))
    # Create the list of the different quantities.
    wavelength = [list(x) for x in col_1]
    intensity = [list(y) for y in col_2]
    energy = [list(fct.nm_eV(x)) for x in wavelength]

    df = pd.DataFrame(data={'B': B_values, 'wavelength': wavelength, 'energy': energy, 'intensity': intensity})
    return df
    
    
#    Goal: detect saturation phenomenon
#    Here we simply check if the spectrum contains a saturated point within the range [pix_min, pix_max].
#    Practically it correponds to have a value > 60000 counts.

def saturation_check(matrix, pix_min, pix_max, threshold=60000):
    """
        Check if there is any saturation in the spectrum within the range [pix_min, pix_max],
        and return a mask array with same shape than the two first dimensions of 
        matrix. Return also a counter of saturations and indexes corresponding to saturated spectra.
    """
    n0, n1, n2,  = matrix.shape
    matrix_ravel = np.reshape(matrix[:, :, :], (n0*n1, n2))
    # create a mask for the 'matrix_ravel' with False value per default
    mask_sat  = np.array([False for i in range(matrix_ravel.shape[0])])
    index_sat = []
    for i in range(matrix_ravel.shape[0]):
        yA = np.amax(matrix_ravel[i, pix_min:pix_max])
        # check whether or not there is saturation
        if yA >= threshold:
            mask_sat[i] = True
            index_sat.append(i) 
        else :
            pass
    #if kill_sat == True:
    #    # filter the matrix_ravel
    #    matrix_ravel_clean = matrix_ravel[~mask_sat]
    # number of saturation: np.sum will count 1 for each True value of the boolean sequence 'mask_sat'
    counter_sat = np.sum(mask_sat)
    mask_sat = np.reshape(mask_sat[:], (n0, n1))
    return {'mask_sat':mask_sat, 'counter_sat':counter_sat, 'index_sat':index_sat}    
    

#    Usually a Max of a peak occurs within few pixels, most often here only one pixel. So detecting a saturation means
#    check if for a pixel corresponding to a Max, the pixels around within a safety interval give lower value.    
    
def flatness_check(matrix, pix_min, pix_max, width_interval_check=0.8, threshold_counter=10, mask_for_index=None):
    """
        Check if there is a overflatened peak in the spectrum within the range [pix_min, pix_max],
        and return a mask array with same shape than the two first dimensions of 
        matrix. Return also a counter of overflatened peaks and indexes corresponding to spectra.
    """
    n0, n1, n2, = matrix.shape
    matrix_ravel = np.reshape(matrix[:, :, :], (n0*n1, n2))
    index = np.arange(0, matrix_ravel.shape[0], 1)
    mask_for_index = list(mask_for_index) # needed to evaluate the existence, otherwise numpy array try to evaluate each element
    if mask_for_index is not None:
        mask_for_index = np.array(mask_for_index)
        index = index[~mask_for_index.ravel()]
    else:
        mask_for_index = np.array(mask_for_index)
    delta = pix_max - pix_min
    check_interval = np.arange(-delta, delta+1, 1)
    mask = check_interval != 0
    check_interval = check_interval[mask]
    # create a mask for the 'matrix_ravel' with False value per default
    mask_flat  = np.array([False for i in range(matrix_ravel.shape[0])])
    index_flat = []
    for i in index:
        yA = np.amax(matrix_ravel[i, pix_min:pix_max])
        xA = np.argmax(matrix_ravel[i, pix_min:pix_max])+pix_min
        interval_check_min = width_interval_check * yA
        interval_check_max = yA
        counter = 0
        for j in check_interval:
            value = matrix_ravel[i, xA+j] 
            if (value >= interval_check_min) and (value <= interval_check_max):
                counter += 1
        if counter > threshold_counter:
            mask_flat[i] = True
            index_flat.append(i)     
        else:
            pass
    # number of flatness: np.sum will count 1 for each True value of the boolean iterable 'mask_flat'
    counter_flat = np.sum(mask_flat)
    mask_flat    = np.reshape(mask_flat[:], (n0, n1))
    return {'mask_flat':mask_flat, 'counter_flat':counter_flat, 'index_flat':index_flat}
