######## GENERAL IMPORTS
from astropy.io import fits
import pandas as pd
import numpy as np
import pyregion
from astropy import units as u
from astropy.coordinates import SkyCoord
import os
import multiprocessing
import scipy
import scipy.integrate as integrate
import warnings
warnings.filterwarnings('ignore')

######## LOCAL IMPORTS
from all_functions import flux_extractor


if __name__ == "__main__":

    ### Reading the input file
    variables = {}
    with open('input_file.txt', 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                var_name, var_value = line.split('=')
                var_name = var_name.strip()
                var_value = var_value.strip()
                exec(f'{var_name} = {var_value}', {}, variables)
    name          = variables['name']
    vla_image     = variables['vla_image']
    cube_image    = variables['cube_image']
    mask_template = variables['mask_cube']
    
    print('\nReading files...')
    
    #######radio  
    base_S        = fits.open(vla_image)[0]
    base_S_data0  = base_S.data                  
    base_S_header = base_S.header
    ####### CO cube

    cube          = fits.open(cube_image)[0] 
    cube_data0    = cube.data
    cube_data0[np.isnan(cube_data0)] = 0
    cube_header   = cube.header

    ####### Template Mask Cube
    mask_cube         = fits.open(mask_template)[0] 
    mask_cube_data0    = mask_cube.data

    ####### Preparing arrays
    base_S_data     = base_S_data0.copy().astype('float64')
    cube_data       = cube_data0.copy().astype('float64')
    mask_cube_data  = mask_cube_data0 .copy().astype('float64')

    ####### Calculations
    pxscale          = -1*cube_header['CDELT1']*3600 
    bmaj             = cube_header['BMAJ']*3600
    bmin             = cube_header['BMIN']*3600
    beam_size_in_pix =np.pi*bmaj*bmin/(pxscale**2*4*np.log(2))

    V_d        = (1-cube_header['CRVAL3']/cube_header['RESTFRQ'])*2.99999e5                                                  
    V_u        = (1-(cube_header['CRVAL3']+cube_header['CDELT3']*cube_header['NAXIS3'])/cube_header['RESTFRQ'])*2.99999e5    
    chan_width = (V_u-V_d)/cube_header['NAXIS3']                                                                              

    print('\nExtracting fluxes...')
    radio_val = flux_extractor(mask_cube_data, base_S_data, beam_size_in_pix)
    hex_flux = flux_extractor(mask_cube_data, cube_data, beam_size_in_pix)

    print('\nCreating txt files to store fluxes data...')
    ######## saving radio flux data in txt file 
    try:
        os.remove('master_radio_flux_'+name+'.txt')
    except:
        print('')
    hex_radio_flux = {}
    hx_df          = pd.DataFrame(hex_radio_flux)
    hx_df['flux']  = radio_val
    hx_df.to_csv(r'master_radio_flux_'+name+'.txt', header=True, index=True, sep='\t', mode='a')

    ######## saving CO flux data in txt file 
    try:
        os.remove('master_CO_'+name+'.txt')
    except:
        print('')
    hex_cube_flux = {}
    hx_CO         = pd.DataFrame(hex_cube_flux)
    for nn in range(cube_data.shape[0]):
        hx_CO['channel_'+str(nn)] = hex_flux[:,nn]

    hx_CO.to_csv(r'master_CO_'+name+'.txt', header=True, index=True, sep='\t', mode='a')

    ######### Integrating grid over frequency to create Moment-zero values per hex
    total_region    = hex_flux.shape[0]
    v = hex_flux.shape[1]
    x_axis      = np.linspace(0,v-1,v,dtype=int)
    profile_int = np.zeros(total_region)
    peaks       = np.zeros(total_region)
    for f in range(total_region):
        profile_int[f] = scipy.integrate.simps(hex_flux[f,:],x_axis,dx=1,axis=-1)*chan_width
        
    ######## saving integrated-intensity (moment-0) data in txt file
    try:
        os.remove('master_mom0_'+name+'.txt')
    except:
        print('')
    hex_mom0             = {}
    hx_mom0              = pd.DataFrame(hex_mom0)
    hx_mom0['mom0_flux'] = profile_int
    hx_mom0.to_csv(r'master_mom0_'+name+'.txt', header=True, index=True, sep='\t', mode='a')

    print('Done!')


