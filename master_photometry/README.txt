This code is intended to extract the flux from VLA and ALMA FITS files.
Along with these scripts, you will need 3 fits files that we describe below:

All images must be of the same spatial size in pixels and same pixel scale, centered on the same coordinates.
VLA image: a 2D fits files with 2 axis.
ALMA cube: a 3D fits files with 3 axis.
ALMA moment-zero map: a 2D image with 2 axis.

If images have more than the specified number of axes, you will be able to run most of the the scripts, with the exception of the one generating the synthetic maps as we used aplpy to plot them.

There are 5 scripts:
hex_mask_creator.py, flux_extractor.py, photometry.py, run_plots.py, run_maps.py

There are two additional txt files: input_file.txt and photometry_default_values.txt

You will need to edit the input file with the information of your target, such as coordinates, conversion factors, name of the FITS files, distance, redshift, etc.

hex_mask_creator.py: Creates a template mask for each tile of the grid. This is later used to extract flux or assign values.

flux_extractor.py: Uses the template mask and applies it to the images to extract the flux values. It creates txt files with the information.

photometry.py: This computed quantities from the fluxes. You can run this script interactively or using default values. It is recommended to do it interactively when you do it for the first time. If done interactively, the code will give you the values you can use to customize the photometry_default_values.txt file. If you customize those values, you can run the code using those values. This is a way to make code more efficient. It creates txt files with the results. It also creates a folder with plots of the line fits.


run_plots.py and run_maps.py: are visualization scripts, they create plots and maps from the photometry txt outputs. They create the folders PLOTS and MAPS respectively. You can skip these two and make your own plots using the photometry outputs. 

Follow the steps to run the code:

1. Customize the input file with information from your target.
2. Run hex_mask_creator.py
3. Run flux_extractor.py
4. Run photometry.py
	You can run this script interactively or opt for the default values.
5. Run run_plots.py
6. Run run_maps.py