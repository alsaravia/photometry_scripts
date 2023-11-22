######## GENERAL IMPORTS
from astropy.io import fits
import numpy as np
import pyregion
from astropy import units as u
from astropy.coordinates import SkyCoord
import os
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

######## LOCAL IMPORTS
from all_functions import make_bin_grid
from all_functions import grid_splitter

def mask_creator(pool_args):
    n,slice_2d, segment, vla_image = pool_args
    base = fits.open(vla_image)[0]
    base_data  = base.data
    base_header= base.header
    k = n % segment  
    m = int(np.floor(n/segment))
    parent_dir = str(os.getcwdb())[2:-1]
    hex_regions = pyregion.open(parent_dir+'/grids/hex_grid_'+str(m)+'.reg')
    del hex_regions[0:k]
    del hex_regions[1:segment]
    aperture_mask = hex_regions.get_mask(shape=(base_data.shape[0], base_data.shape[1]),header = base_header)
    aperture_mask_hdu = fits.PrimaryHDU(aperture_mask.astype('short'), base_header)
    processed_slice = slice_2d + aperture_mask_hdu.data
    return processed_slice



if __name__ == "__main__":

    ##### Reading the imput file
    variables = {}
    with open('input_file.txt', 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                var_name, var_value = line.split('=')
                var_name = var_name.strip()
                var_value = var_value.strip()
                exec(f'{var_name} = {var_value}', {}, variables)
    dec = variables['dec']
    RA = variables['RA']
    hex_area = variables['hex_area']
    vla_image = variables['vla_image']
    cube_image = variables['cube_image']
    grid_dim = variables['grid_dim']


    print('\nReading files...')
    
    #######radio  
    base_S       = fits.open(vla_image)[0]
    base_S_data  = base_S.data                  
    base_S_header= base_S.header 

    ####### CO cube

    cube         = fits.open(cube_image)[0] 
    cube_data    = cube.data
    cube_header  = cube.header
    x = cube_data.shape[1]
    y = cube_data.shape[2]

    ###### Making the hex grid
    grid_center = SkyCoord(RA,dec, unit = u.deg) 
    grid_size   = (grid_dim[0]*u.arcsec, grid_dim[1]*u.arcsec)
    
    print('Done \n Now creating grids...')
    grid = make_bin_grid(grid_center, gridsize = grid_size, kind='hex', area=hex_area*u.arcsec**2)
    grid.write('hex_grid.reg',format='ds9',overwrite=True)
    grid_splitter('hex_grid.reg')
   
    ###### reading main hex grid
    hex_regions           = pyregion.open('hex_grid.reg')
    total_region          = len(hex_regions)
    segment               = int(np.ceil(np.sqrt(total_region)))
    mask_header           = cube_header.copy()
    mask_header['NAXIS3'] = total_region


    
    print('Done \n Now creating template mask...')
    masked_cube      =  np.zeros([total_region,x,y])
    pool_args = [(n,masked_cube[n,:,:], segment, vla_image) for n in range(total_region)]
    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    results = pool.map(mask_creator, pool_args)
    mask_cube = np.array(results)
    mask = fits.PrimaryHDU(mask_cube.astype('int'), mask_header)
    mask_file = 'template_mask_cube.fits'
    mask.writeto(mask_file,overwrite=True)
    pool.close()
    pool.join()
    print('Done!')