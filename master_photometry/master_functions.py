import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from regions import Regions
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from numba import njit, prange

#############################  plot_fits #############################################

def plot_fits(fits_file, region_file=None, figsize=(10, 10), font_size=10):
    """
    Plots a FITS image and overlays DS9 regions if a region file is provided.
    
    :param fits_file: Path to the FITS file to be displayed.
    :param region_file: Optional; path to the DS9 region file to overlay on the FITS image.
    :param figsize: Optional; a tuple indicating the size of the figure (width, height).
    :param font_size: Optional; font size for the text annotations.
    """
    # Load the FITS file
    with fits.open(fits_file) as hdul:
        data = hdul[0].data
        wcs_info = WCS(hdul[0].header)

    # Set the font size globally
    plt.rcParams.update({'font.size': font_size})

    # Plot the FITS image
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': wcs_info})
    im = ax.imshow(data, origin='lower', cmap='viridis')
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')

    # Add a colorbar 
    cbar_ax = fig.add_axes([0.92, 0.1, 0.03, 0.8])  
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Intensity')  # You can change 'Intensity' to match what the data represents

    # Overlay regions if a region file is provided
    if region_file:
        hex_regions = Regions.read(region_file, format='ds9')
        for region in hex_regions:
            # Convert to pixel coordinates and plot
            region_pix = region.to_pixel(wcs_info)
            vertices = region_pix.vertices.xy
            xs, ys = vertices
            if xs[0] != xs[-1] or ys[0] != ys[-1]:
                xs = np.append(xs, xs[0])
                ys = np.append(ys, ys[0])
            ax.plot(xs, ys, 'r-')

    plt.show()

# Example 
#plot_fits('NGC_5258_cube_1arcsec_unclipped.mom0.pbcor_inregion_hcorr.fits', 'new_regions.reg', figsize=(15, 15), font_size=30)



#############################  plot_fits_with_contours ######################################


def plot_fits_with_contours(fits_file, rms=None, num_contours=0, figsize=(10, 10), font_size=10):
    """
    Plots a FITS image and overlays contours based on the provided RMS value.

    :param fits_file: Path to the FITS file to be displayed.
    :param rms: The RMS value to base the contours on. If None, no contours are plotted.
    :param num_contours: The number of additional contour levels to plot, incremented by 1 * rms.
    :param figsize: Optional; a tuple indicating the size of the figure (width, height).
    :param font_size: Optional; font size for the text annotations.
    """
    # Load the FITS file
    with fits.open(fits_file) as hdul:
        data = hdul[0].data
        wcs_info = WCS(hdul[0].header)

    # Set the font size globally
    plt.rcParams.update({'font.size': font_size})

    # Plot the FITS image
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': wcs_info})
    im = ax.imshow(data, origin='lower', cmap='viridis')
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')

    # Add a colorbar 
    cbar_ax = fig.add_axes([0.92, 0.1, 0.03, 0.8]) 
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Intensity')  # You can change 'Intensity' to match what the data represents

    # Plot contours if RMS is provided
    if rms is not None:
        levels = [3 * rms + i * rms for i in range(num_contours + 1)]
        ax.contour(data, levels=levels, colors='white', alpha=0.5, transform=ax.get_transform(wcs_info))

    plt.show()

# Example
#plot_fits_with_contours('NGC_5258_cube_1arcsec_unclipped.mom0.pbcor_inregion_hcorr.fits', rms=0.8, num_contours=1, figsize=(8, 8), font_size=20)


#############################  filter_regions ######################################


def filter_regions(input_region_file, indices, output_region_file):
    """
    Filters regions from an input DS9 region file based on given indices 
    and writes the selected regions to a new region file.

    :param input_region_file: Path to the input DS9 region file.
    :param indices: An array or list of indices specifying which regions to keep.
    :param output_region_file: Path to the output DS9 region file.
    """
    # Read the input region file
    all_regions = Regions.read(input_region_file, format='ds9')

    # Filter the regions based on the provided indices
    selected_regions = [all_regions[i] for i in indices if i < len(all_regions)]

    # Create a Regions object from the selected regions
    filtered_regions = Regions(selected_regions)

    # Write the selected regions to the new region file
    filtered_regions.write(output_region_file, format='ds9', overwrite=True)

# Example
#filter_regions('hex_grid.reg', indices , 'new_regions.reg')

################################## region to mask ##############################

def region_to_mask(fits_file, region_file):
    """
    Create a mask from a DS9 region file, where pixels inside the region are True.

    :param fits_file: Path to the FITS file.
    :param region_file: Path to the DS9 region file.
    :return: A 2D numpy array (mask) with the same shape as the FITS image.
    """
    # Load the FITS file to get the shape and WCS
    with fits.open(fits_file) as hdul:
        data = hdul[0].data
        wcs_info = WCS(hdul[0].header)

    # Read the region file
    regions = Regions.read(region_file, format='ds9')

    # Initialize the mask with the same shape as the FITS image, filled with False
    mask = np.zeros(data.shape, dtype=bool)

    # Convert regions to pixel coordinates and create the mask
    for region in regions:
        region_pix = region.to_pixel(wcs_info)
        mask_region = region_pix.to_mask(mode='center').to_image(data.shape)
        
        # Ensure the mask is boolean
        mask_region_bool = mask_region.astype(bool)

        # Combine the region mask with the main mask
        mask = mask | mask_region_bool

    return mask

#example
#region_to_mask('NGC_5258_cube_1arcsec_unclipped.mom0.pbcor_inregion_hcorr.fits', 'new_regions.reg')

#################################### region_to_mask_frmDnH ########################################

def region_to_mask_frmDnH(data, header, region_file):
    """
    Create a mask from a DS9 region file, where pixels inside the region are True,
    using FITS data and header provided separately.

    :param data: Numpy array of the FITS data (can be 2D or 3D).
    :param header: Header of the FITS file.
    :param region_file: Path to the DS9 region file.
    :return: A numpy array (mask) with the same shape as the FITS data.
    """
    # Check if the data is 2D or 3D and set up WCS accordingly
    if data.ndim == 2:
        wcs_info = WCS(header, naxis=2)
    elif data.ndim == 3:
        wcs_info = WCS(header, naxis=2)  # Use only spatial axes for WCS

    # Read the region file
    regions = Regions.read(region_file, format='ds9')

    # Initialize the mask with the same shape as the FITS data, filled with False
    mask = np.zeros(data.shape[-2:], dtype=bool)  # Mask should match spatial dimensions

    # Convert regions to pixel coordinates and create the mask
    for region in regions:
        region_pix = region.to_pixel(wcs_info)
        mask_region = region_pix.to_mask(mode='center').to_image(mask.shape)

        # Combine the region mask with the main mask
        mask |= mask_region.astype(bool)

    return mask


################################# create_sky_region ##############################################
def create_sky_region(fits_file, coords, output_region_file='sky_region.reg'):
    """
    Create a sky region file from pixel coordinates based on the WCS of a given FITS file.

    :param fits_file: The FITS file to use as a template for WCS.
    :param coords: A list of coordinates defining the polygon in pixel space.
    :param output_region_file: The output region file name.
    """
    # Create the DS9 region string
    region_str = f"image\npolygon({','.join(map(str, coords))}) # color=green"
    regions = Regions.parse(region_str, format='ds9')

    # Load WCS from FITS header
    with fits.open(fits_file) as hdul:
        wcs = WCS(hdul[0].header)

    # Convert each region to sky coordinates
    sky_regions = [region.to_sky(wcs) for region in regions]

    # Encapsulate the list of sky regions into a Regions object before writing
    sky_regions_container = Regions(sky_regions)

    # Write to region file
    sky_regions_container.write(output_region_file, format='ds9', overwrite=True)

# Example 
#coords = [72, 379, 72, 212, 228, 8, 452, 8, 452, 245, 215, 489, 121, 489]
#fits_file = 'NGC_5258_cube_1arcsec_unclipped.mom0.pbcor_inregion_hcorr.fits'
#create_sky_region(fits_file, coords, 'outsidemask.reg')



####################### flux_extractor #####################################################
import concurrent.futures
def calculate_flux_for_region(data, wcs_info, region, beam_size_in_pix):
    region_pix = region.to_pixel(wcs_info)
    mask = region_pix.to_mask(mode='center').to_image(data.shape[-2:]).astype(bool)

    if data.ndim == 2:
        masked_data = mask * data
        flux = np.sum(masked_data) / beam_size_in_pix
        return np.array([flux])  # Return as a 1-element array
    elif data.ndim == 3:
        expanded_mask = np.repeat(mask[np.newaxis, :, :], data.shape[0], axis=0)
        masked_data = expanded_mask * data
        flux = np.sum(masked_data, axis=(1, 2)) / beam_size_in_pix
        return flux  # This is already an array

def flux_extractor(data, header, beam_size_in_pix=1, regions_file=None, batch_size=10):
    wcs_info = WCS(header, naxis=2)
    regions = Regions.read(regions_file, format='ds9')
    
    # Determine the shape of the results array based on the dimensionality of the data
    if data.ndim == 3:
        results = np.zeros((len(regions), data.shape[0]))  # For 3D data, store flux for each channel
    else:
        results = np.zeros(len(regions))  # For 2D data, store total flux

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        for i, region in enumerate(regions):
            # Pass the index to store the result directly in the correct position
            futures[executor.submit(calculate_flux_for_region, data, wcs_info, region, beam_size_in_pix)] = i

        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            result = future.result()
            if data.ndim == 3:
                results[index, :] = result  # Assign the array of flux values for each channel
            else:
                results[index] = result  # Assign the total flux for 2D data

    return results

################################## create_map_from_region ######################################
def create_map_from_regions(header, regions_file, results, indices=None):
    """
    Create an image where specified region's pixels are set to the corresponding value from the results array.
    
    :param header: The FITS header to use for WCS information.
    :param regions_file: Path to the DS9 region file defining the regions.
    :param results: An array of values where each value corresponds to a region in the region file.
    :param indices: Optional; list of indices indicating which regions to fill. If None, all regions are filled.
    :return: A 2D numpy array with specified regions filled with their corresponding flux values.
    """
    wcs_info = WCS(header)
    all_regions = Regions.read(regions_file, format='ds9')
    
    if indices is not None:
        regions = [all_regions[i] for i in indices]
        results = [results[i] for i in indices]
    else:
        regions = all_regions

    data_shape = (header['NAXIS2'], header['NAXIS1'])
    output_image = np.zeros(data_shape)

    for value, region in zip(results, regions):
        region_pix = region.to_pixel(wcs_info)
        mask = region_pix.to_mask(mode='center')

        bbox = mask.bbox
        slice_x = slice(max(0, bbox.ixmin), min(bbox.ixmax, data_shape[1]))
        slice_y = slice(max(0, bbox.iymin), min(bbox.iymax, data_shape[0]))

        mask_data = mask.data
        region_slice = output_image[slice_y, slice_x]

        if mask_data.shape != region_slice.shape:
            mask_data = mask_data[:min(slice_y.stop - slice_y.start, mask_data.shape[0]),
                                  :min(slice_x.stop - slice_x.start, mask_data.shape[1])]

        region_slice[mask_data.astype(bool)] = value

    return output_image

# Usage example
#header = base_S_header  # Ensure this is a header object, not data
#region = 'hex_grid.reg'
#indices = [0, 1, 2, 3]  # Example indices to specify which regions to fill
#array = create_image_from_regions(header, region, radio_data, indices=indices)


############################### array_plotter ########################################

import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS
from matplotlib.patches import Ellipse
import astropy.units as u

def array_plotter(array, figsize=(8, 8), font_size=25, cbar_label='Intensity', header=None):
    """
    Plot the given array with a color bar, WCS information if provided, and a beam ellipse.

    :param array: 2D numpy array to plot.
    :param figsize: Tuple of figure size (width, height) in inches.
    :param font_size: Font size for labels, titles, and tick labels.
    :param cbar_label: Label for the color bar.
    :param header: Optional; FITS header with WCS information. If provided, the plot will include WCS axes and beam size.
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(projection=WCS(header) if header else None)

    im = ax.imshow(array, origin='lower', cmap='viridis')
    ax.set_xlabel('RA', fontsize=font_size)
    ax.set_ylabel('Dec', fontsize=font_size)
    ax.tick_params(labelsize=font_size)

    # Plot the beam if header information is available
    if header and all(key in header for key in ['BMAJ', 'BMIN', 'BPA', 'CDELT1']):
        bmaj = header['BMAJ'] * 3600  # Beam major axis in arcseconds
        bmin = header['BMIN'] * 3600  # Beam minor axis in arcseconds
        bpa = header['BPA']  # Beam position angle in degrees
        pxscale = -header['CDELT1'] * 3600  # Pixel scale in arcseconds per pixel

        beam_size_in_pix = np.pi * bmaj * bmin / (pxscale**2 * 4 * np.log(2))  # Beam area in pixels

        beam_pos = (header['NAXIS1'] * 0.1, header['NAXIS2'] * 0.1)
        beam = Ellipse(xy=beam_pos, width=bmin / pxscale, height=bmaj / pxscale,
                       angle=bpa, edgecolor='white', facecolor='none', linewidth=2)
        ax.add_patch(beam)

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, size=font_size)
    cbar.ax.tick_params(labelsize=font_size)

    plt.show()

# Usage example
# Assuming 'array' is the 2D numpy array and 'header' contains WCS information
# array_plotter(array, figsize=(12, 12), font_size=14, cbar_label='Flux', header=your_fits_header)