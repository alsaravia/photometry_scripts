######### GRID MAKER
import numpy as np
import regions
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.signal import savgol_filter

def hex_region(center, area, meta=None):
    """
    Return a regular hexagonal PolygonSkyRegion centered at `center` with a
    total area of `area`.

    No sky projection is made; hexagonal regions are assumed to be small enough
    to be flat.
    """

    # A = sqrt(3)/2 * (2r)^2
    inr = np.sqrt(2./np.sqrt(3.) * area) / 2.   
    # A = 3 * sqrt(3) / 8 * (2 * R)^2
    outR = inr * 2./np.sqrt(3.)
    # A = 3/2 * sqrt(3) * a^2
    sideL = np.sqrt(area / ((3./2.) * np.sqrt(3.))) 

    verticies = SkyCoord([center.ra - outR,
                          center.ra - sideL/2.,
                          center.ra + sideL/2.,
                          center.ra + outR,
                          center.ra + sideL/2.,
                          center.ra - sideL/2.],
                         [center.dec,
                          center.dec + inr,
                          center.dec + inr,
                          center.dec,
                          center.dec - inr,
                          center.dec - inr],
                         unit='deg')

    return regions.PolygonSkyRegion(verticies, meta=meta)

def make_bin_grid(grid_center,
                  gridsize=None,
                  kind='hex',
                  area=(1*u.arcsec**2)):
    """
    Assemble a grid of tiled regions (either hexagonal or square).
    The grid is "grown" from a SkyCoord `grid_center` and extends out for
    a total extent of gridsize(x) and gridsize(y). Each grid is returned as
    a PolygonSkyRegion list

    kind can be either 'hex' or 'square'

    Returns a Regions() class with a list of the generated regions
    """

    # list to contain all the regions we generate
    allreg = []
    """
    allreg.append(regions.PointSkyRegion(grid_center,
                                         visual={'color': 'red'}))
    """
    factor = 2./np.sqrt(3.)                                        ############# coefficient defined here
    factor2 = np.sqrt(2./np.sqrt(3.) )*(np.sqrt(2))                ############# coefficients defined here
    
    # 1. determine how many bins of the desired area are required to
    #    cover the requested x and y extents
    if kind == 'square':
        height = np.sqrt(area).to('deg')
        width = np.sqrt(area).to('deg')
        xshift = width
        yshift = height
        Nx = int(np.ceil(gridsize[0].to('deg') / width))
        Ny = int(np.ceil(gridsize[1].to('deg') / height))
    elif kind == 'hex':
        height = np.sqrt(factor * area).to('deg')                  ######### coefficient substituted here     
        width = factor2 * np.sqrt(area).to('deg')                  ######### coefficient substituted here
        xshift = np.sqrt(3./2.) * width
        yshift = height / 2.
        Nx = int(np.ceil(gridsize[0].to('deg') / width))
        Ny = int(np.ceil(gridsize[1].to('deg') / (height/2)))

    # 2. compute the center positions for all the bins
    # list to store all the center locations
    centers = []
    # double-for is slow, but start with this as a basic implementation
    for j in range(min([-1, int(-Ny/2)]), max([1, int(Ny/2)]), 1):
        # start with the center row and move up
        if kind == 'hex' and (j % 2 == 1):
            # if we're doing hex bins and the row number is odd, shift the
            # row to the left
            rowshift = np.sqrt(3) * yshift
        else:
            # if the row number is even or we are using squares, do not
            # shift alternating rows
            rowshift = 0 * u.arcsec

        for i in range(min([-1, int(-Nx/2)]), max([1, int(Nx/2)]), 1):
            # start with the leftmost grid and move to the right
            centers.append(SkyCoord(grid_center.ra.to('deg') + rowshift + i * xshift,
                                    grid_center.dec.to('deg') + j * yshift))

    # 3. loop over all the center positions and generate the appropriate
    #    regions for each mask
    for i, center in enumerate(centers):
        if kind == 'hex':
            allreg.append(hex_region(center,
                                     area=area,
                                     meta={'name': i}))
        elif kind == 'square':
            allreg.append(regions.RectangleSkyRegion(center,
                                                     width=width.to('arcsec'),
                                                     height=height.to('arcsec'),
                                                     meta={'name': i}))

    # return the collection of regions as a single object
    return regions.Regions(allreg)

####### GRID SPLITTER

import numpy as np
import regions
from astropy.coordinates import SkyCoord
import astropy.units as u
import pyregion
from astropy import units as u
import warnings
warnings.filterwarnings('ignore')
import os

name   = 'hex_grid.reg'


def grid_splitter(name):
    hex_regions_split = pyregion.open(name)
    total = len(hex_regions_split)
    segment = int(np.ceil(np.sqrt(total)))

    ###### creates a folder to store the grid's files. CAREFUL to change the name of galaxy
    new_folder     = 'grids'
    parent_dir = str(os.getcwdb())[2:-1]
    path = os.path.join(parent_dir, new_folder)
    try:
        os.mkdir(path)
    except:
        print(' ')

    ##### CAREFUL to change the name of galaxy

    initial = 0
    final   = segment
    m = int(np.ceil(total/segment))
    for n in range(m):
        a = hex_regions_split[initial:final]
        a.write(parent_dir +'/'+ new_folder +'/'+ 'hex_grid_' +str(n)+'.reg')
        initial = final
        final  = final + segment

import matplotlib.pyplot as plt
from scipy import optimize

def gaussian(x, height, center, width):
    return height/np.sqrt(width)*np.exp(-(x - center)**2/(2*width**2)) 

def two_gaussians(x, h1, c1, w1, h2, c2, w2):
    return three_gaussians(x, h1, c1, w1, h2, c2, w2, 0,0,1)

def three_gaussians(x, h1, c1, w1, h2, c2, w2, h3, c3, w3):
    return (gaussian(x, h1, c1, w1) +
        gaussian(x, h2, c2, w2) +
        gaussian(x, h3, c3, w3))

def four_gaussians(x, h1, c1, w1, h2, c2, w2, h3, c3, w3, h4, c4, w4):
    return (gaussian(x, h1, c1, w1) +
        gaussian(x, h2, c2, w2) +
        gaussian(x, h3, c3, w3) + gaussian(x, h4, c4, w4) )

def five_gaussians(x, h1, c1, w1, h2, c2, w2, h3, c3, w3, h4, c4, w4, h5, c5, w5):
    return (gaussian(x, h1, c1, w1) +
        gaussian(x, h2, c2, w2) +
        gaussian(x, h3, c3, w3) + gaussian(x, h4, c4, w4) + gaussian(x, h5, c5, w5))


def six_gaussians(x, h1, c1, w1, h2, c2, w2, h3, c3, w3, h4, c4, w4, h5, c5, w5, h6, c6, w6):
    return (gaussian(x, h1, c1, w1) +
        gaussian(x, h2, c2, w2) +
        gaussian(x, h3, c3, w3) + gaussian(x, h4, c4, w4) + gaussian(x, h5, c5, w5) + gaussian(x, h6, c6, w6))



def individual_fitting_plot(h,thrsh,polyn,rms_avg_1,array,ax): 
    ydata = array
    center   = int(np.array(np.where(ydata==np.max(ydata)))[0])
    last     = int(np.array(np.where(ydata==ydata[-1]))[0])
    interval = last-center
    first    = 0
    if interval<center:
        first = center-interval
    else:
        last = 2*center

    ydata = ydata[first:last]+1
    yhat = savgol_filter(ydata, 51, polyn)
    n_hex = len(yhat)
    xdata = np.linspace(0,n_hex-1,n_hex,dtype=int)
    xdata_r = np.linspace(0,len(ydata)-1,len(ydata),dtype=int)
    
    indices_a = np.array(np.where(yhat -1 >= thrsh*rms_avg_1))[0]
    signo     = []
    extr      = []
    real_extr = []
    
    for i in indices_a[:-1]:
        slope = (yhat[i+1]-yhat[i])/(xdata[i+1]-xdata[i])
        signo.append(slope/np.abs(slope))

    for i in range(1,len(signo)-2,1):
        if signo[i+1]!= signo[i] and signo[i+2] == signo[i+1]:
            extr.append(indices_a[i])

    for i in range(0,len(extr),1):
        if i%2==0:
            real_extr.append(extr[i])
    
    number_gauss = len(real_extr)   

    centers = real_extr
    amp     = yhat[centers]
    
    data = np.zeros([len(yhat),2])
    for i in range(len(yhat)):
        data[i,0] = xdata[i]
        data[i,1] = yhat[i]-1
 

    ax.set_title('Hexagon '+str(h),fontsize = 20)
    ax.plot(data[:,0], data[:,1], lw=1, c='g', linestyle = 'none', marker = '.', label='smoothed hex data')
    ax.axhline(y = thrsh*rms_avg_1,color = 'k',linestyle='dashed', label = str(thrsh)+ 'times the rms') 
    ax.plot(xdata_r, ydata-1, lw=0.01, c='gray', linestyle = 'none', marker = '.', label='raw hex data hex ')

    if number_gauss == 1:
        errfunc1 = lambda p, x, y: (gaussian(x, *p) - y)
        sigma = indices_a[-1]-indices_a[0]
        guess1 = [amp[0], centers[0], sigma]
        optim, cov_matrix, _ , _ , _  = optimize.leastsq(errfunc1, guess1[:], args=(data[:,0], data[:,1]),full_output=True)
        if (len(data[:,1]) > len(guess1)) and cov_matrix is not None:
            s_sq   = (errfunc1(optim, data[:,0], data[:,1])**2).sum()/(len(data[:,1])-len(guess1))

            pcov = cov_matrix * s_sq
        else:
            pcov = np.inf
        error = []
        for i in range(len(optim)):
            try:
                error.append(np.absolute(pcov[i][i])**0.5)
            except:
                error.append( 0.00 )     
        lnwdth = optim[2]
        lnwdth_err = error[2]
        ax.plot(data[:,0], gaussian(data[:,0], *optim),lw=3, c='orange', ls='--', label='fit of 1 Gaussian')

           
    elif number_gauss == 2:
        sigma = []
        for i in range(0,len(extr)-1,1):
            sigm = np.abs(extr[i]-extr[i+1])
            sigma.append(sigm)
        sigma = np.array(sigma)
        
        errfunc2 = lambda p, x, y: (two_gaussians(x, *p) - y)
        guess2 = [amp[0], centers[0], sigma[0], amp[1], centers[1], sigma[1]]
        optim, cov_matrix, _ , _ , _  = optimize.leastsq(errfunc2, guess2[:], args=(data[:,0], data[:,1]),full_output=True)    
        if (len(data[:,1]) > len(guess2)) and cov_matrix is not None:
            s_sq   = (errfunc2(optim, data[:,0], data[:,1])**2).sum()/(len(data[:,1])-len(guess2))
            pcov = cov_matrix * s_sq
        else:
            pcov = np.inf
        error = []
        for i in range(len(optim)):
            try:
                error.append(np.absolute(pcov[i][i])**0.5)
            except:
                error.append( 0.00 )   
        real_sigma = []
        real_error = []
        for k in [2,5]:
            if optim[k] > 0:
                np.array(real_sigma.append(optim[k]))
                np.array(real_error.append(error[k]))
        lnwdth = np.sum(real_sigma)/len(real_sigma)       
        lnwdth_err = np.sum(real_error)/len(real_error)
        ax.plot(data[:,0], two_gaussians(data[:,0], *optim),lw=3, c='orange', ls='--', label='fit of 2 Gaussians')

        
    elif number_gauss ==3:
        sigma = []
        for i in range(0,len(extr)-1,1):
            sigm = np.abs(extr[i]-extr[i+1])
            sigma.append(sigm)
        sigma = np.array(sigma)
        errfunc3 = lambda p, x, y: (three_gaussians(x, *p) - y)
        guess3 = [amp[0], centers[0], sigma[0], amp[1], centers[1], sigma[1],amp[2], centers[2], sigma[2]] 
        optim, cov_matrix, _ , _ , _  = optimize.leastsq(errfunc3, guess3[:], args=(data[:,0], data[:,1]),full_output=True)
        if (len(data[:,1]) > len(guess3)) and cov_matrix is not None:
            s_sq   = (errfunc3(optim, data[:,0], data[:,1])**2).sum()/(len(data[:,1])-len(guess3))
            pcov = cov_matrix * s_sq
        else:
            pcov = np.inf
        error = []
        for i in range(len(optim)):
            try:
                error.append(np.absolute(pcov[i][i])**0.5)
            except:
                error.append( 0.00 )   
        real_sigma = []
        real_error = []
        for k in [2,5,8]:
            if optim[k] > 0:
                np.array(real_sigma.append(optim[k]))
                np.array(real_error.append(error[k]))
        lnwdth = np.sum(real_sigma)/len(real_sigma) 
        lnwdth_err = np.sum(real_error)/len(real_error)
        ax.plot(data[:,0], three_gaussians(data[:,0], *optim),lw=3, c='orange', label='fit of 3 Gaussians')


    elif number_gauss ==4:
        sigma = []
        for i in range(0,len(extr)-1,1):
            sigm = np.abs(extr[i]-extr[i+1])
            sigma.append(sigm)
        sigma = np.array(sigma)
        errfunc4 = lambda p, x, y: (four_gaussians(x, *p) - y)
        guess4 = [amp[0], centers[0], sigma[0], amp[1], centers[1], sigma[1],amp[2], centers[2], sigma[2], amp[3], centers[3], sigma[3]] 
        optim, cov_matrix, _ , _ , _  = optimize.leastsq(errfunc4, guess4[:], args=(data[:,0], data[:,1]),full_output=True)
        if (len(data[:,1]) > len(guess4)) and cov_matrix is not None:
            s_sq   = (errfunc4(optim, data[:,0], data[:,1])**2).sum()/(len(data[:,1])-len(guess4))
            pcov = cov_matrix * s_sq
        else:
            pcov = np.inf
        error = []
        for i in range(len(optim)):
            try:
                error.append(np.absolute(pcov[i][i])**0.5)
            except:
                error.append( 0.00 )     
        real_sigma = []
        real_error = []
        for k in [2,5,8,11]:
            if optim[k] > 0:
                np.array(real_sigma.append(optim[k]))
                np.array(real_error.append(error[k]))
        lnwdth = np.sum(real_sigma)/len(real_sigma) 
        lnwdth_err = np.sum(real_error)/len(real_error)
        ax.plot(data[:,0], four_gaussians(data[:,0], *optim),lw=3, c='orange', label='fit of 4 Gaussians')

    elif number_gauss ==5:
        sigma = []
        for i in range(0,len(extr)-1,1):
            sigm = np.abs(extr[i]-extr[i+1])
            sigma.append(sigm)
        sigma = np.array(sigma)
        errfunc5 = lambda p, x, y: (five_gaussians(x, *p) - y)
        guess5 = [amp[0], centers[0], sigma[0], amp[1], centers[1], sigma[1],amp[2], centers[2], sigma[2], amp[3], centers[3], sigma[3], amp[4], centers[4], sigma[4]] 
        optim, cov_matrix, _ , _ , _  = optimize.leastsq(errfunc5, guess5[:], args=(data[:,0], data[:,1]),full_output=True)
        if (len(data[:,1]) > len(guess5)) and cov_matrix is not None:
            s_sq   = (errfunc5(optim, data[:,0], data[:,1])**2).sum()/(len(data[:,1])-len(guess5))
            pcov = cov_matrix * s_sq
        else:
            pcov = np.inf
        error = []
        for i in range(len(optim)):
            try:
                error.append(np.absolute(pcov[i][i])**0.5)
            except:
                error.append( 0.00 )     
        real_sigma = []
        real_error = []
        for k in [2,5,8,11,14]:
            if optim[k] > 0:
                np.array(real_sigma.append(optim[k]))
                np.array(real_error.append(error[k]))
        lnwdth = np.sum(real_sigma)/len(real_sigma) 
        lnwdth_err = np.sum(real_error)/len(real_error)
        ax.plot(data[:,0], five_gaussians(data[:,0], *optim),lw=3, c='orange', label='fit of 5 Gaussians')

    elif number_gauss >=6:
        sigma = []
        for i in range(0,len(extr)-1,1):
            sigm = np.abs(extr[i]-extr[i+1])
            sigma.append(sigm)
        sigma = np.array(sigma)
        errfunc6 = lambda p, x, y: (six_gaussians(x, *p) - y)
        guess6 = [amp[0], centers[0], sigma[0], amp[1], centers[1], sigma[1],amp[2], centers[2], sigma[2], amp[3], centers[3], sigma[3], amp[4], centers[4], sigma[4], amp[5], centers[5], sigma[5]] 
        optim, cov_matrix, _ , _ , _  = optimize.leastsq(errfunc6, guess6[:], args=(data[:,0], data[:,1]),full_output=True)
        if (len(data[:,1]) > len(guess6)) and cov_matrix is not None:
            s_sq   = (errfunc6(optim, data[:,0], data[:,1])**2).sum()/(len(data[:,1])-len(guess6))
            pcov = cov_matrix * s_sq
        else:
            pcov = np.inf
        error = []
        for i in range(len(optim)):
            try:
                error.append(np.absolute(pcov[i][i])**0.5)
            except:
                error.append( 0.00 )     
        real_sigma = []
        real_error = []
        for k in [2,5,8,11,14,17]:
            if optim[k] > 0:
                np.array(real_sigma.append(optim[k]))
                np.array(real_error.append(error[k]))
        lnwdth = np.sum(real_sigma)/len(real_sigma) 
        lnwdth_err = np.sum(real_error)/len(real_error)
        ax.plot(data[:,0], six_gaussians(data[:,0], *optim),lw=3, c='orange', label='fit of 6 Gaussians')

    ax.legend(loc='best',fontsize = 15)
plt.close()

def linewidth_fit(indices,rms_avg_1,thrsh,polyn,CO_channel_data, chan_width):
    j = 0
    linewidth     = np.zeros(len(indices))
    linewidth_err = np.zeros(len(indices))
    
    for h in indices:        
        ydata = CO_channel_data[h,:]
        center   = int(np.array(np.where(ydata==np.max(ydata)))[0])
        last     = int(np.array(np.where(ydata==ydata[-1]))[0])
        interval = last-center
        first    = 0
        if interval<center:
            first = center-interval
        else:
            last = 2*center
    
        ydata = ydata[first:last]+1
        yhat = savgol_filter(ydata, 51, polyn)
        n_hex = len(yhat)
        xdata = np.linspace(0,n_hex-1,n_hex,dtype=int)
        
        indices_a = np.array(np.where(yhat -1 >= thrsh*rms_avg_1))[0]
        signo     = []
        extr      = []
        real_extr = []
        
        for i in indices_a[:-1]:
            slope = (yhat[i+1]-yhat[i])/(xdata[i+1]-xdata[i])
            signo.append(slope/np.abs(slope))
    
        for i in range(1,len(signo)-2,1):
            if signo[i+1]!= signo[i] and signo[i+2] == signo[i+1]:
                extr.append(indices_a[i])
    
        for i in range(0,len(extr),1):
            if i%2==0:
                real_extr.append(extr[i])
        
        number_gauss = len(real_extr)   
    
        centers = real_extr
        amp     = yhat[centers]
        
        data = np.zeros([len(yhat),2])
        for i in range(len(yhat)):
            data[i,0] = xdata[i]
            data[i,1] = yhat[i]-1
        
        
        if number_gauss == 1:
            errfunc1 = lambda p, x, y: (gaussian(x, *p) - y)
            sigma = indices_a[-1]-indices_a[0]
            guess1 = [amp[0], centers[0], sigma]
            optim, cov_matrix, _ , _ , _  = optimize.leastsq(errfunc1, guess1[:], args=(data[:,0], data[:,1]),full_output=True)
            if (len(data[:,1]) > len(guess1)) and cov_matrix is not None:
                s_sq   = (errfunc1(optim, data[:,0], data[:,1])**2).sum()/(len(data[:,1])-len(guess1))
    
                pcov = cov_matrix * s_sq
            else:
                pcov = np.inf
            error = []
            for i in range(len(optim)):
                try:
                    error.append(np.absolute(pcov[i][i])**0.5)
                except:
                    error.append( 0.00 )     
            lnwdth = optim[2]
            lnwdth_err = error[2]
    
               
        elif number_gauss == 2:
            sigma = []
            for i in range(0,len(extr)-1,1):
                sigm = np.abs(extr[i]-extr[i+1])
                sigma.append(sigm)
            sigma = np.array(sigma)
            
            errfunc2 = lambda p, x, y: (two_gaussians(x, *p) - y)
            guess2 = [amp[0], centers[0], sigma[0], amp[1], centers[1], sigma[1]]
            optim, cov_matrix, _ , _ , _  = optimize.leastsq(errfunc2, guess2[:], args=(data[:,0], data[:,1]),full_output=True)    
            if (len(data[:,1]) > len(guess2)) and cov_matrix is not None:
                s_sq   = (errfunc2(optim, data[:,0], data[:,1])**2).sum()/(len(data[:,1])-len(guess2))
                pcov = cov_matrix * s_sq
            else:
                pcov = np.inf
            error = []
            for i in range(len(optim)):
                try:
                    error.append(np.absolute(pcov[i][i])**0.5)
                except:
                    error.append( 0.00 )   
            real_sigma = []
            real_error = []
            for k in [2,5]:
                if optim[k] > 0:
                    np.array(real_sigma.append(optim[k]))
                    np.array(real_error.append(error[k]))
            lnwdth = np.sum(real_sigma)/len(real_sigma)       
            lnwdth_err = np.sum(real_error)/len(real_error)
    
            
        elif number_gauss ==3:
            sigma = []
            for i in range(0,len(extr)-1,1):
                sigm = np.abs(extr[i]-extr[i+1])
                sigma.append(sigm)
            sigma = np.array(sigma)
            errfunc3 = lambda p, x, y: (three_gaussians(x, *p) - y)
            guess3 = [amp[0], centers[0], sigma[0], amp[1], centers[1], sigma[1],amp[2], centers[2], sigma[2]] 
            optim, cov_matrix, _ , _ , _  = optimize.leastsq(errfunc3, guess3[:], args=(data[:,0], data[:,1]),full_output=True)
            if (len(data[:,1]) > len(guess3)) and cov_matrix is not None:
                s_sq   = (errfunc3(optim, data[:,0], data[:,1])**2).sum()/(len(data[:,1])-len(guess3))
                pcov = cov_matrix * s_sq
            else:
                pcov = np.inf
            error = []
            for i in range(len(optim)):
                try:
                    error.append(np.absolute(pcov[i][i])**0.5)
                except:
                    error.append( 0.00 )   
            real_sigma = []
            real_error = []
            for k in [2,5,8]:
                if optim[k] > 0:
                    np.array(real_sigma.append(optim[k]))
                    np.array(real_error.append(error[k]))
            lnwdth = np.sum(real_sigma)/len(real_sigma) 
            lnwdth_err = np.sum(real_error)/len(real_error)
    
    
        elif number_gauss ==4:
            sigma = []
            for i in range(0,len(extr)-1,1):
                sigm = np.abs(extr[i]-extr[i+1])
                sigma.append(sigm)
            sigma = np.array(sigma)
            errfunc4 = lambda p, x, y: (four_gaussians(x, *p) - y)
            guess4 = [amp[0], centers[0], sigma[0], amp[1], centers[1], sigma[1],amp[2], centers[2], sigma[2], amp[3], centers[3], sigma[3]] 
            optim, cov_matrix, _ , _ , _  = optimize.leastsq(errfunc4, guess4[:], args=(data[:,0], data[:,1]),full_output=True)
            if (len(data[:,1]) > len(guess4)) and cov_matrix is not None:
                s_sq   = (errfunc4(optim, data[:,0], data[:,1])**2).sum()/(len(data[:,1])-len(guess4))
                pcov = cov_matrix * s_sq
            else:
                pcov = np.inf
            error = []
            for i in range(len(optim)):
                try:
                    error.append(np.absolute(pcov[i][i])**0.5)
                except:
                    error.append( 0.00 )     
            real_sigma = []
            real_error = []
            for k in [2,5,8,11]:
                if optim[k] > 0:
                    np.array(real_sigma.append(optim[k]))
                    np.array(real_error.append(error[k]))
            lnwdth = np.sum(real_sigma)/len(real_sigma) 
            lnwdth_err = np.sum(real_error)/len(real_error)
    
        
        elif number_gauss ==5:
            sigma = []
            for i in range(0,len(extr)-1,1):
                sigm = np.abs(extr[i]-extr[i+1])
                sigma.append(sigm)
            sigma = np.array(sigma)
            errfunc5 = lambda p, x, y: (five_gaussians(x, *p) - y)
            guess5 = [amp[0], centers[0], sigma[0], amp[1], centers[1], sigma[1],amp[2], centers[2], sigma[2], amp[3], centers[3], sigma[3], amp[4], centers[4], sigma[4]] 
            optim, cov_matrix, _ , _ , _  = optimize.leastsq(errfunc5, guess5[:], args=(data[:,0], data[:,1]),full_output=True)
            if (len(data[:,1]) > len(guess5)) and cov_matrix is not None:
                s_sq   = (errfunc5(optim, data[:,0], data[:,1])**2).sum()/(len(data[:,1])-len(guess5))
                pcov = cov_matrix * s_sq
            else:
                pcov = np.inf
            error = []
            for i in range(len(optim)):
                try:
                    error.append(np.absolute(pcov[i][i])**0.5)
                except:
                    error.append( 0.00 )     
            real_sigma = []
            real_error = []
            for k in [2,5,8,11,14]:
                if optim[k] > 0:
                    np.array(real_sigma.append(optim[k]))
                    np.array(real_error.append(error[k]))
            lnwdth = np.sum(real_sigma)/len(real_sigma) 
            lnwdth_err = np.sum(real_error)/len(real_error)
    
        elif number_gauss >=6:
            sigma = []
            for i in range(0,len(extr)-1,1):
                sigm = np.abs(extr[i]-extr[i+1])
                sigma.append(sigm)
            sigma = np.array(sigma)
            errfunc6 = lambda p, x, y: (six_gaussians(x, *p) - y)
            guess6 = [amp[0], centers[0], sigma[0], amp[1], centers[1], sigma[1],amp[2], centers[2], sigma[2], amp[3], centers[3], sigma[3], amp[4], centers[4], sigma[4], amp[5], centers[5], sigma[5]] 
            optim, cov_matrix, _ , _ , _  = optimize.leastsq(errfunc6, guess6[:], args=(data[:,0], data[:,1]),full_output=True)
            if (len(data[:,1]) > len(guess6)) and cov_matrix is not None:
                s_sq   = (errfunc6(optim, data[:,0], data[:,1])**2).sum()/(len(data[:,1])-len(guess6))
                pcov = cov_matrix * s_sq
            else:
                pcov = np.inf
            error = []
            for i in range(len(optim)):
                try:
                    error.append(np.absolute(pcov[i][i])**0.5)
                except:
                    error.append( 0.00 )     
            real_sigma = []
            real_error = []
            for k in [2,5,8,11,14,17]:
                if optim[k] > 0:
                    np.array(real_sigma.append(optim[k]))
                    np.array(real_error.append(error[k]))
            lnwdth = np.sum(real_sigma)/len(real_sigma) 
            lnwdth_err = np.sum(real_error)/len(real_error)    
            
        linewidth[j]  = lnwdth*chan_width
        linewidth_err[j] = lnwdth_err*chan_width
        j = j + 1

    return linewidth , linewidth_err, thrsh, polyn


def flux_extractor(mask_cube_data, fits_file, beam_size_in_pix):
    if len(fits_file.shape) ==2:
        results = np.zeros(mask_cube_data.shape[0])
        for n in range(mask_cube_data.shape[0]):    
            image_S = np.multiply(mask_cube_data[n,:,:],fits_file)
            results[n] = np.sum(image_S)/beam_size_in_pix
        
    elif len(fits_file.shape) ==3:
        v, x, y = fits_file.shape
        results = np.zeros((mask_cube_data.shape[0], v))
        for n in range(mask_cube_data.shape[0]):
            masked_cube = np.multiply(mask_cube_data[n, :, :], fits_file)
            results[n, :] = np.sum(masked_cube, axis = (1,2))/beam_size_in_pix
    else:
        results = print('Error: Check your FITS file dimensions')
        
    return results

