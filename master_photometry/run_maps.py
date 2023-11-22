import numpy as np
import pandas as pd
import os
from astropy.io import fits
import matplotlib.pyplot as plt
import aplpy
plt.rcParams["font.family"] = "Times New Roman"
import warnings
warnings.filterwarnings('ignore')


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
mask_template = variables['mask_cube']

base_S         = fits.open(vla_image)[0]
base_S_data    = base_S.data
base_S_header  = base_S.header
mask_cube      = fits.open(mask_template)[0] 
mask_cube_data = mask_cube.data

SFR_n_Gas  = pd.read_csv('Results_SFR_+_GAS_'+name+'.txt', sep="\t")
kinematics = pd.read_csv('kinematics_'+name+'.txt', sep="\t")
indices = np.array(kinematics['index'])

new_folder     = 'MAPS'
parent_dir = str(os.getcwdb())[2:-1]
path = os.path.join(parent_dir, new_folder)

try:
    os.remove('MAPS')
except:
    print('')

try:
    os.mkdir(path)
except:
    print(' ')


S_SFR     = np.array(SFR_n_Gas['$\u03A3$_SFR'][indices])
SFR       = np.array(SFR_n_Gas['SFR'][indices])
SFE       = np.array(SFR_n_Gas['SFE'][indices])
t_dep     = np.array(1/SFR_n_Gas['SFE'][indices])
S_flux    = np.array(SFR_n_Gas['radio_flux'][indices])
CO_flux   = np.array(SFR_n_Gas['CO_flux'][indices])
S_H2      = np.array(SFR_n_Gas['$\u03A3$_H2'][indices])
H2_Mass   = np.array(SFR_n_Gas['Mass_H2'][indices])
Sigma     = np.array(kinematics['sigma'])
P_turb    = np.array(kinematics['P_turb'])
alpha_vir = np.array(kinematics['alpha_vir'])
ff_time   = np.array(kinematics['t_ff'])
E_per_ff  = np.array(kinematics['E_ff'])

background_S_SFR     = np.zeros(base_S_data.shape)
background_SFR       = np.zeros(base_S_data.shape)
background_SFE       = np.zeros(base_S_data.shape)
background_t_dep     = np.zeros(base_S_data.shape)
background_S_flux    = np.zeros(base_S_data.shape)
background_CO_flux   = np.zeros(base_S_data.shape)
background_S_H2      = np.zeros(base_S_data.shape)
background_H2_Mass   = np.zeros(base_S_data.shape)
background_Sigma     = np.zeros(base_S_data.shape)
background_P_turb    = np.zeros(base_S_data.shape)
background_alpha_vir = np.zeros(base_S_data.shape)
background_ff_time   = np.zeros(base_S_data.shape)
background_E_per_ff  = np.zeros(base_S_data.shape)

e = 0
for n in indices:
    background_S_SFR     = background_S_SFR + mask_cube_data[n,:,:]*S_SFR[e]
    background_SFR       = background_SFR + mask_cube_data[n,:,:]*SFR[e]
    background_SFE       = background_SFE + mask_cube_data[n,:,:]*SFE[e]
    background_t_dep     = background_t_dep + mask_cube_data[n,:,:]*t_dep[e]
    background_S_flux    = background_S_flux + mask_cube_data[n,:,:]*S_flux[e]
    background_CO_flux   = background_CO_flux + mask_cube_data[n,:,:]*CO_flux[e]
    background_S_H2      = background_S_H2 + mask_cube_data[n,:,:]*S_H2[e]
    background_H2_Mass   = background_H2_Mass + mask_cube_data[n,:,:]*H2_Mass[e]
    background_Sigma     = background_Sigma + mask_cube_data[n,:,:]*Sigma[e]
    background_P_turb    = background_P_turb + mask_cube_data[n,:,:]*P_turb[e]
    background_alpha_vir = background_alpha_vir + mask_cube_data[n,:,:]*alpha_vir[e]
    background_ff_time   = background_ff_time + mask_cube_data[n,:,:]*ff_time[e]
    background_E_per_ff  = background_E_per_ff + mask_cube_data[n,:,:]*E_per_ff [e]   
    e=e+1

############################################################### SFR surface density ############################################################################

array       = S_SFR
image_array = background_S_SFR

max_ = np.max(np.log10(array))
min_ = np.min(np.log10(array))
hdu = fits.PrimaryHDU(np.log10(image_array).astype('float'), base_S_header)
fig = aplpy.FITSFigure(hdu, figsize=(12, 12))
fig.show_colorscale(cmap='viridis', vmax=max_, vmin = min_)
fig.add_colorbar()
fig.colorbar.set_location('right')
fig.colorbar.set_width(0.3) 
fig.colorbar.set_font(size='25')
fig.colorbar.set_axis_label_text('Log$\u03A3_{SFR}$[$M_{\odot}$ $yr^{-1}$ $kpc^{-2}$] ') 
fig.colorbar.set_axis_label_font(size=25)
fig.axis_labels.set_font(size=20)
fig.tick_labels.set_font(size=20)
fig.ticks.set_length(10) 
fig.add_beam()
fig.beam.set_color('black')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'SFR_surface_density.png',bbox_inches='tight')
plt.close()

############################################################### SFR ##############################################################################################

array       = SFR
image_array = background_SFR

max_ = np.max(np.log10(array))
min_ = np.min(np.log10(array))
hdu = fits.PrimaryHDU(np.log10(image_array).astype('float'), base_S_header)
fig = aplpy.FITSFigure(hdu, figsize=(12, 12))
fig.show_colorscale(cmap='viridis', vmax=max_, vmin = min_)
fig.add_colorbar()
fig.colorbar.set_location('right')
fig.colorbar.set_width(0.3) 
fig.colorbar.set_font(size='25')
fig.colorbar.set_axis_label_text('Log SFR [$M_{\odot}$ $yr^{-1}$] ') 
fig.colorbar.set_axis_label_font(size=25)
fig.axis_labels.set_font(size=20)
fig.tick_labels.set_font(size=20)
fig.ticks.set_length(10) 
fig.add_beam()
fig.beam.set_color('black')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'SFR.png',bbox_inches='tight')
plt.close()

############################################################### SFE ############################################################################

array       = SFE
image_array = background_SFE

max_ = np.max(np.log10(array))
min_ = np.min(np.log10(array))
hdu = fits.PrimaryHDU(np.log10(image_array).astype('float'), base_S_header)
fig = aplpy.FITSFigure(hdu, figsize=(12, 12))
fig.show_colorscale(cmap='viridis', vmax=max_, vmin = min_)
fig.add_colorbar()
fig.colorbar.set_location('right')
fig.colorbar.set_width(0.3) 
fig.colorbar.set_font(size='25')
fig.colorbar.set_axis_label_text('Log SFE [$yr^{-1}$] ') 
fig.colorbar.set_axis_label_font(size=25)
fig.axis_labels.set_font(size=20)
fig.tick_labels.set_font(size=20)
fig.ticks.set_length(10) 
fig.add_beam()
fig.beam.set_color('black')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'SFE.png',bbox_inches='tight')
plt.close()

############################################################### Depletion times ############################################################################

array       = t_dep
image_array = background_t_dep

max_ = np.max(np.log10(array))
min_ = np.min(np.log10(array))
hdu = fits.PrimaryHDU(np.log10(image_array).astype('float'), base_S_header)
fig = aplpy.FITSFigure(hdu, figsize=(12, 12))
fig.show_colorscale(cmap='viridis', vmax=max_, vmin = min_)
fig.add_colorbar()
fig.colorbar.set_location('right')
fig.colorbar.set_width(0.3) 
fig.colorbar.set_font(size='25')
fig.colorbar.set_axis_label_text('Log depletion time [yrs] ') 
fig.colorbar.set_axis_label_font(size=25)
fig.axis_labels.set_font(size=20)
fig.tick_labels.set_font(size=20)
fig.ticks.set_length(10) 
fig.add_beam()
fig.beam.set_color('black')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'depletion_time.png',bbox_inches='tight')
plt.close()

############################################################### Radio Flux ############################################################################

array       = S_flux
image_array = background_S_flux

max_ = np.max(np.log10(array))
min_ = np.min(np.log10(array))
hdu = fits.PrimaryHDU(np.log10(image_array).astype('float'), base_S_header)
fig = aplpy.FITSFigure(hdu, figsize=(12, 12))
fig.show_colorscale(cmap='viridis', vmax=max_, vmin = min_)
fig.add_colorbar()
fig.colorbar.set_location('right')
fig.colorbar.set_width(0.3) 
fig.colorbar.set_font(size='25')
fig.colorbar.set_axis_label_text('Log radio flux [Jy] ') 
fig.colorbar.set_axis_label_font(size=25)
fig.axis_labels.set_font(size=20)
fig.tick_labels.set_font(size=20)
fig.ticks.set_length(10) 
fig.add_beam()
fig.beam.set_color('black')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'radio_flux.png',bbox_inches='tight')
plt.close()


############################################################### CO FLux ############################################################################

array       = CO_flux
image_array = background_CO_flux

max_ = np.max(np.log10(array))
min_ = np.min(np.log10(array))
hdu = fits.PrimaryHDU(np.log10(image_array).astype('float'), base_S_header)
fig = aplpy.FITSFigure(hdu, figsize=(12, 12))
fig.show_colorscale(cmap='viridis', vmax=max_, vmin = min_)
fig.add_colorbar()
fig.colorbar.set_location('right')
fig.colorbar.set_width(0.3) 
fig.colorbar.set_font(size='25')
fig.colorbar.set_axis_label_text('Log CO flux [Jy km $s^{-1}$] ') 
fig.colorbar.set_axis_label_font(size=25)
fig.axis_labels.set_font(size=20)
fig.tick_labels.set_font(size=20)
fig.ticks.set_length(10) 
fig.add_beam()
fig.beam.set_color('black')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'CO_flux.png',bbox_inches='tight')
plt.close()

############################################################### Molecualar gas surface density ############################################################################

array       = S_H2
image_array = background_S_H2

max_ = np.max(np.log10(array))
min_ = np.min(np.log10(array))
hdu = fits.PrimaryHDU(np.log10(image_array).astype('float'), base_S_header)
fig = aplpy.FITSFigure(hdu, figsize=(12, 12))
fig.show_colorscale(cmap='viridis', vmax=max_, vmin = min_)
fig.add_colorbar()
fig.colorbar.set_location('right')
fig.colorbar.set_width(0.3) 
fig.colorbar.set_font(size='25')
fig.colorbar.set_axis_label_text('Log$\u03A3_{mol}$[$M_{\odot}$ $yr^{-1}$ $kpc^{-2}$] ') 
fig.colorbar.set_axis_label_font(size=25)
fig.axis_labels.set_font(size=20)
fig.tick_labels.set_font(size=20)
fig.ticks.set_length(10) 
fig.add_beam()
fig.beam.set_color('black')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'Mol_gas_surface_density.png',bbox_inches='tight')
plt.close()

############################################################### Gas Mass ############################################################################

array       = H2_Mass
image_array = background_H2_Mass

max_ = np.max(np.log10(array))
min_ = np.min(np.log10(array))
hdu = fits.PrimaryHDU(np.log10(image_array).astype('float'), base_S_header)
fig = aplpy.FITSFigure(hdu, figsize=(12, 12))
fig.show_colorscale(cmap='viridis', vmax=max_, vmin = min_)
fig.add_colorbar()
fig.colorbar.set_location('right')
fig.colorbar.set_width(0.3) 
fig.colorbar.set_font(size='25')
fig.colorbar.set_axis_label_text('Log Gas Mass [$M_{\odot}$] ') 
fig.colorbar.set_axis_label_font(size=25)
fig.axis_labels.set_font(size=20)
fig.tick_labels.set_font(size=20)
fig.ticks.set_length(10) 
fig.add_beam()
fig.beam.set_color('black')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'Gas_mass.png',bbox_inches='tight')
plt.close()

############################################################### linewidth ############################################################################

array       = Sigma
image_array = background_Sigma

max_ = np.max(array)
min_ = np.min(array)
hdu = fits.PrimaryHDU(image_array.astype('float'), base_S_header)
fig = aplpy.FITSFigure(hdu, figsize=(12, 12))
fig.show_colorscale(cmap='viridis', vmax=max_, vmin = min_)
fig.add_colorbar()
fig.colorbar.set_location('right')
fig.colorbar.set_width(0.3) 
fig.colorbar.set_font(size='25')
fig.colorbar.set_axis_label_text('Linewidth [km/s] ') 
fig.colorbar.set_axis_label_font(size=25)
fig.axis_labels.set_font(size=20)
fig.tick_labels.set_font(size=20)
fig.ticks.set_length(10) 
fig.add_beam()
fig.beam.set_color('black')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'linewidth.png',bbox_inches='tight')
plt.close()

############################################################### linewidth ############################################################################

array       = Sigma
image_array = background_Sigma

max_ = np.max(np.log10(array))
min_ = np.min(np.log10(array))
hdu = fits.PrimaryHDU(np.log10(image_array).astype('float'), base_S_header)
fig = aplpy.FITSFigure(hdu, figsize=(12, 12))
fig.show_colorscale(cmap='viridis', vmax=max_, vmin = min_)
fig.add_colorbar()
fig.colorbar.set_location('right')
fig.colorbar.set_width(0.3) 
fig.colorbar.set_font(size='25')
fig.colorbar.set_axis_label_text('Log linewidth [km/s] ') 
fig.colorbar.set_axis_label_font(size=25)
fig.axis_labels.set_font(size=20)
fig.tick_labels.set_font(size=20)
fig.ticks.set_length(10) 
fig.add_beam()
fig.beam.set_color('black')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'Log linewidth.png',bbox_inches='tight')
plt.close()

############################################################### P_turb ############################################################################

array       = P_turb
image_array = background_P_turb

max_ = np.max(np.log10(array))
min_ = np.min(np.log10(array))
hdu = fits.PrimaryHDU(np.log10(image_array).astype('float'), base_S_header)
fig = aplpy.FITSFigure(hdu, figsize=(12, 12))
fig.show_colorscale(cmap='viridis', vmax=max_, vmin = min_)
fig.add_colorbar()
fig.colorbar.set_location('right')
fig.colorbar.set_width(0.3) 
fig.colorbar.set_font(size='25')
fig.colorbar.set_axis_label_text('Log turbulent Pressure [K $cm^{-3}$] ') 
fig.colorbar.set_axis_label_font(size=25)
fig.axis_labels.set_font(size=20)
fig.tick_labels.set_font(size=20)
fig.ticks.set_length(10) 
fig.add_beam()
fig.beam.set_color('black')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'P_turb.png',bbox_inches='tight')
plt.close()

############################################################### Virial Parameter ############################################################################

array       = alpha_vir
image_array = background_alpha_vir

max_ = np.max(array)
min_ = np.min(array)
hdu = fits.PrimaryHDU(image_array.astype('float'), base_S_header)
fig = aplpy.FITSFigure(hdu, figsize=(12, 12))
fig.show_colorscale(cmap='viridis', vmax=max_, vmin = min_)
fig.add_colorbar()
fig.colorbar.set_location('right')
fig.colorbar.set_width(0.3) 
fig.colorbar.set_font(size='25')
fig.colorbar.set_axis_label_text('Virial Parameter') 
fig.colorbar.set_axis_label_font(size=25)
fig.axis_labels.set_font(size=20)
fig.tick_labels.set_font(size=20)
fig.ticks.set_length(10) 
fig.add_beam()
fig.beam.set_color('black')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'Virial_parameter.png',bbox_inches='tight')
plt.close()

############################################################### Log virial Parameter ############################################################################

array       = alpha_vir
image_array = background_alpha_vir

max_ = np.max(np.log10(array))
min_ = np.min(np.log10(array))
hdu = fits.PrimaryHDU(np.log10(image_array).astype('float'), base_S_header)
fig = aplpy.FITSFigure(hdu, figsize=(12, 12))
fig.show_colorscale(cmap='viridis', vmax=max_, vmin = min_)
fig.add_colorbar()
fig.colorbar.set_location('right')
fig.colorbar.set_width(0.3) 
fig.colorbar.set_font(size='25')
fig.colorbar.set_axis_label_text('Log virial Parameter') 
fig.colorbar.set_axis_label_font(size=25)
fig.axis_labels.set_font(size=20)
fig.tick_labels.set_font(size=20)
fig.ticks.set_length(10) 
fig.add_beam()
fig.beam.set_color('black')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'Log virial_parameter.png',bbox_inches='tight')
plt.close()

############################################################### ff_time ############################################################################

array       = ff_time
image_array = background_ff_time

max_ = np.max(np.log10(array))
min_ = np.min(np.log10(array))
hdu = fits.PrimaryHDU(np.log10(image_array).astype('float'), base_S_header)
fig = aplpy.FITSFigure(hdu, figsize=(12, 12))
fig.show_colorscale(cmap='viridis', vmax=max_, vmin = min_)
fig.add_colorbar()
fig.colorbar.set_location('right')
fig.colorbar.set_width(0.3) 
fig.colorbar.set_font(size='25')
fig.colorbar.set_axis_label_text('Log Free Fall time [Myr] ') 
fig.colorbar.set_axis_label_font(size=25)
fig.axis_labels.set_font(size=20)
fig.tick_labels.set_font(size=20)
fig.ticks.set_length(10) 
fig.add_beam()
fig.beam.set_color('black')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'Free_fall_time.png',bbox_inches='tight')
plt.close()

############################################################### E_per_ff ############################################################################

array       = E_per_ff
image_array = background_E_per_ff

max_ = np.max(np.log10(array))
min_ = np.min(np.log10(array))
hdu = fits.PrimaryHDU(np.log10(image_array).astype('float'), base_S_header)
fig = aplpy.FITSFigure(hdu, figsize=(12, 12))
fig.show_colorscale(cmap='viridis', vmax=max_, vmin = min_)
fig.add_colorbar()
fig.colorbar.set_location('right')
fig.colorbar.set_width(0.3) 
fig.colorbar.set_font(size='25')
fig.colorbar.set_axis_label_text('Log SF efficiency per free fall time') 
fig.colorbar.set_axis_label_font(size=25)
fig.axis_labels.set_font(size=20)
fig.tick_labels.set_font(size=20)
fig.ticks.set_length(10) 
fig.add_beam()
fig.beam.set_color('black')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'Efficiency_per_ff_time.png',bbox_inches='tight')
plt.close()

print('Done!')

