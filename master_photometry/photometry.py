######## GENERAL IMPORTS
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyregion
from astropy import units as u
from astropy.coordinates import SkyCoord
import os
import multiprocessing
import scipy
import scipy.integrate as integrate
from all_functions import linewidth_fit
from all_functions import individual_fitting_plot
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

def ask_user(question):
    while True:
        response = input(question + " (y/n): ").lower()
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        else:
            print("Invalid response. Please answer with 'y' for yes or 'n' for no.")

def SFR_F(T,v,alpha_NT,L):
    return 1.e-27*(2.18*(T/1.e4)**0.45*(v)**(-0.1)+15.1*(v)**(1.*alpha_NT))**(-1)*(L)


def line_finder(thresh,rms_avg_1,number,array):
    line_free_hex = []
    line_detc_hex = []
    for j in range(0, CO_channel_data.shape[0],1):
        det = 0
        for i in range(0,v-number,1):
            subset = CO_channel_data[j,i:i+number]
            if all(k > thresh*rms_avg_1 for k in subset):
                det = det + 1
            else:
                det = det
        if det != 0:
            line_detc_hex.append(j)
        elif det == 0:
            line_free_hex.append(j)     

    return line_detc_hex, line_free_hex

def rms_function(values):
    rms = np.std(values)
    return rms

def get_positive_input(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = int(input(prompt))
            if (min_value is not None and value <= min_value) or (max_value is not None and value >= max_value):
                print(f"Please enter a value greater than {min_value} and less than {max_value}.")
            else:
                return value
        except ValueError:
            print("Please enter a valid number.")


def input_positive_number(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value > 0:
                return value
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":

    ### Reading the input file
    print('\nReading files...\n Cube noise characterization:')
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
    CO_to_H2      = variables['CO_to_H2']
    distance      = variables['distance']
    z             = variables['redshift']


    radio_data      = np.array(pd.read_csv('master_radio_flux_'+name+'.txt', sep="\s+"))[:,0]
    CO_channel_data = np.array(pd.read_csv('master_CO_'+name+'.txt', sep="\s+"))
    mom0_data       = np.array(pd.read_csv('master_mom0_'+name+'.txt', sep="\s+"))[:,0]
    total_region    = CO_channel_data.shape[0]
    v               = CO_channel_data.shape[1]


    # Asking the user whether to run this section of the code
    if ask_user("Do you want to input initial values? \n (if no, we will use default values)"):
        ##### Creating folder to save plots
        new_folder     = 'Inspec_Images'
        parent_dir = str(os.getcwdb())[2:-1]
        path = os.path.join(parent_dir, new_folder)
        try:
            os.mkdir(path)
        except:
            print(' ')
        
        ######### PLOTS AND CALCULATIONS
        print('\nCreating plot of flux per channel of the cube...')

        ###### Creating plots of all non-zero line-free channels
        non_zero_indices = []   
        for i in range(0,CO_channel_data.shape[0],1):
            detection = len([k for k in CO_channel_data[i,:] if all(k == 0 for k in CO_channel_data[i,:])])
            if detection==0:
                non_zero_indices.append(i)
        
        plt.figure(figsize=(20, 9))
        channel_v = np.linspace(0, v-1, v, dtype=int)

        for i in non_zero_indices:
            plt.plot(channel_v, CO_channel_data[i, :], linestyle='none', marker='.')

        plt.title('Select a range of line-free channels before and after emission to compute noise', fontsize=20)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.xlabel('Channels', fontsize=20)
        plt.ylabel('Flux', fontsize=20)
        plt.savefig(parent_dir + '/' + new_folder + '/' + 'all_channels', bbox_inches='tight')
        
        # Ask the user if they want to see the graph
        response = input("Do you want to see the plot? (y/n): ").strip().lower()
        if response in ["y", "yes"]:
            plt.show()
        else:
            print("Graph will not be displayed.")


        ######## calculating the average rms per channel
        max_channel = v  # Example maximum value
        print('\nFrom the graph, enter a range of line-free channels before the emission, then enter another range after the emission. There will be 4 values in total: ')

        channel_1 = get_positive_input('Enter first value: ', min_value=0, max_value=max_channel)
        channel_2 = get_positive_input('Enter second value: ', min_value=channel_1, max_value=max_channel)
        channel_3 = get_positive_input('Enter third value: ', min_value=channel_2, max_value=max_channel)
        channel_4 = get_positive_input('Enter fourth value: ', min_value=channel_3, max_value=max_channel)

        print(f"Values entered: {channel_1}, {channel_2}, {channel_3}, {channel_4}")

        emission_hex = CO_channel_data[non_zero_indices,:]    
        print('\nCalculating average rms per channel...')                                                 
        range_1 = emission_hex[:,channel_1:channel_2]
        range_2 = emission_hex[:,channel_3:channel_4]

        range_x = np.concatenate((range_1,range_2),1)         

        rms_chan = np.zeros(range_x.shape[1])

        for i in range(0,range_x.shape[1],1):               
            rms_chan[i] = rms_function(range_x[:,i])
            
        rms_avg_1 = np.mean(rms_chan)                       

        print('rms per line-free channels = ', rms_avg_1, 'Jy')    

        range_1_x = np.linspace(0, range_x.shape[1]-1, range_x.shape[1], dtype=int)
        plt.figure(figsize=(20, 9))
        # Plotting the data
        for i in range(0, range_x.shape[1], 1):
            plt.scatter(range_1_x, range_x[i, :], s=10)

        # Plotting rms, 3rms, and 5rms lines
        plt.axhline(y=rms_avg_1, color='k', linestyle='dashed')
        plt.axhline(y=3*rms_avg_1, color='k', linestyle='dashed')
        plt.axhline(y=5*rms_avg_1, color='k', linestyle='dashed')

        # Setting labels and font sizes
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.xlabel('Line-free Hexagons', fontsize=20)
        plt.ylabel('Flux in line-free channels', fontsize=20)
        plt.savefig(parent_dir + '/' + new_folder + '/' + 'Noise_and_rms.png', bbox_inches='tight')
        # Display the plot
        user_response = input("Do you want to see the plot? (y/n): ").lower()
        if response in ["y", "yes"]:
            plt.show()
        else:
            print("Graph will not be displayed.")


        ####### Sorting out hexagons according to rms
        # Ask the user for the threshold and the number of channels
        print('\nFiltering hexagons...')
        thresh = input_positive_number("Enter the threshold value (xtimes the rms): ")
        number = int(input_positive_number("Enter the number of consecutive channels \n  above the threshold you want to consider: "))
        
        line_detc_hex, line_free_hex = line_finder(thresh, rms_avg_1, number,CO_channel_data)
        
        line_free_hexagon = mom0_data[line_free_hex]
        rms_hex = np.abs(np.mean(line_free_hexagon))
        print('\naverage flux of line-free hexagons= ',rms_hex)
        print('number of hexagons with line detected= ',len(line_detc_hex))
        print('number of line-free hexagons         =',len(line_free_hex))  

        plt.figure(figsize=(20, 9))
        channel_v = np.linspace(0,v-1,v,dtype=int)

        for i in range(0,CO_channel_data.shape[0]-1,1):
            plt.plot(channel_v, CO_channel_data[i, :], linestyle='none', marker='.')

            
        plt.axhline(y = rms_avg_1, color='k', linestyle ='solid', label = '1 x rms')
        plt.axhline(y = 3*rms_avg_1, color='k', linestyle ='dotted', label = '3 x rms')
        plt.axhline(y = 5*rms_avg_1, color='k', linestyle ='dashed', label = '5 x rms')
        plt.axhline(y = thresh*rms_avg_1, color='k', linestyle ='solid', linewidth = 3, label = 'Selected threshold: ' + str(thresh)+ ' x rms')
        plt.title('Flux in all channels and different thresholds', fontsize = 20)
        plt.legend(fontsize = 15)
        plt.xlabel('Channels',fontsize=20)
        plt.ylabel('Flux',fontsize=20)
        plt.savefig(parent_dir + '/' + new_folder + '/' + 'CO_fluxes_n_thresholds.png', bbox_inches='tight')

        peaks = np.zeros(total_region)
        for f in line_detc_hex:
            peaks[f] = np.max(CO_channel_data[f,:])

        df_peak = {'peaks':peaks}
        df_peak =pd.DataFrame(df_peak)
        df_peak.iloc[line_detc_hex]
        lowest = df_peak.iloc[line_detc_hex].nsmallest(16,'peaks').index

        fig, axs = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle('16 faintest line profiles', fontsize=20, y=0.90)
        w = 0
        for i in range(4):
            for j in range(4):
                ax = axs[i, j]
                ax.plot(channel_v,CO_channel_data[lowest[w],:],marker='.',color='k', linestyle='none', label = 'hexagon '+str(lowest[w]))
                ax.axhline(y = thresh*rms_avg_1, color='k', linestyle='dashed')
                ax.legend()
                w = w + 1
        plt.savefig(parent_dir + '/' + new_folder + '/' + 'faintest_line_profiles.png', bbox_inches='tight')

        print('\nwhat is the threshold of your radio image (get it from CASA) \n and what threshold (xtimes the rms) you want to use?')
        rms           = input_positive_number("Enter the rms): ")
        thres_factor  = input_positive_number("Enter the how many times the rms): ") 

    else:
        print("Using default values...")
        variables = {}
        with open('photometry_default_values.txt', 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    var_name, var_value = line.split('=')
                    var_name = var_name.strip()
                    var_value = var_value.strip()
                    exec(f'{var_name} = {var_value}', {}, variables)
        rms_avg_1     = variables['avg_rms_channel']
        rms           = variables['vla_rms']
        thres_factor  = variables['vla_threshold']
        thresh        = variables['line_threshold']
        number        = variables['number_channels']

    ######### COMPUTING SF QUANTITIES
    line_detc_hex, line_free_hex = line_finder(thresh,rms_avg_1,number,CO_channel_data)
    line_free_hexagon = mom0_data[line_free_hex]
    rms_hex = np.abs(np.mean(line_free_hexagon))
    print('\naverage flux of line-free hexagons= ',rms_hex)
    print('number of hexagons with line detected= ',len(line_detc_hex))
    print('number of line-free hexagons         =',len(line_free_hex)) 

    cube          = fits.open(cube_image)[0] 
    cube_header   = cube.header
    pxscale       = -1*cube_header['CDELT1']*3600 
    bmaj          = cube_header['BMAJ']*3600
    bmin          = cube_header['BMIN']*3600
    resolution    = np.sqrt(bmaj*bmin)
    ang_to_pc     = (resolution*1./3600)*distance*1e6/57.29
    nu_obs        = cube_header['CRVAL3']/1e9

    V_d        = (1-cube_header['CRVAL3']/cube_header['RESTFRQ'])*2.99999e5             
    V_u        = (1-(cube_header['CRVAL3']+cube_header['CDELT3']*cube_header['NAXIS3'])/cube_header['RESTFRQ'])*2.99999e5    
    chan_width     = np.abs((V_u-V_d)/cube_header['NAXIS3'])                                                                              

    ### radio
    alpha    = -0.85   ### spectral index
    area     = np.pi*bmaj*bmin/(4*np.log(2))

    Lum_S    = 4*np.pi*(distance*3.086e24)**2*radio_data*1.e-23
    SFR        = SFR_F(1.e4 ,3. ,alpha , Lum_S)
    Sd_SFR  = SFR/(area*(ang_to_pc**2))*(1000000.)

    ###### radio errors

    N_beam = 1    # number of beams per hex
    d_cal_s = 0.05*radio_data
    d_rms_s = rms*np.sqrt(N_beam)

    error_s      = np.sqrt(d_cal_s**2+d_rms_s**2)
    error_Lum_S  = 4*np.pi*(distance*3.086e24)**2*error_s*1.e-23
    err_SFR      = SFR_F(1.e4,3.,alpha,error_Lum_S)
    error_Sd_SFR = err_SFR/(area*(ang_to_pc**2))*(1000000.)

    ################################################################################################
    #### CO

    Lum_G       = 3.25e7*(mom0_data)*nu_obs**(-2.)*distance**2*(1+z)**(-3.)
    M_G         = CO_to_H2*Lum_G
    Sd_G       = M_G/(area*(ang_to_pc**2))

    N_beam  = 1  # number of beams per hex
    rms_G   = rms_hex
    d_cal_G = 0.05*mom0_data
    d_rms_G = rms_G*np.sqrt(N_beam)

    #### errors in CO
    error_G     = np.sqrt(d_cal_G**2+d_rms_G**2)
    error_Lum_G = 3.25e7*(error_G)*nu_obs**(-2.)*distance**2*(1+z)**(-3.)
    error_M_G   = CO_to_H2*error_Lum_G
    error_Sd_G  = error_M_G/(area*(ang_to_pc**2))

    ######## SFE
    SFE         = Sd_SFR/(Sd_G*1.e6)   ### the factor of 1e6 accounts for the different area units
    SFE_err     = SFE*(error_Sd_G/Sd_G+error_Sd_SFR/Sd_SFR)

    dataframe = {'radio_flux':radio_data,'radio_error':error_s,'SFR':SFR,'SFR_error':err_SFR,'CO_flux':mom0_data,'CO_error':error_G,'$\u03A3$_SFR':Sd_SFR,'$\u03A3$_SFR_err':error_Sd_SFR,'Mass_H2':M_G,'Mass_H2_err':error_M_G,'$\u03A3$_H2':Sd_G,'$\u03A3$_H2_err':error_Sd_G,'SFE':SFE,'SFE_err':SFE_err}
    df = pd.DataFrame(dataframe)

    try:
        os.remove('Results_SFR_+_GAS_'+name+'.txt')
    except:
        print('')

    df.to_csv('Results_SFR_+_GAS_'+name+'.txt', header=True, index=True, sep='\t', mode='a')

    ########## Here we do the sorting by imposing thresholds
    Sd_G_rms    = CO_to_H2*(3.25e7*(rms_hex)*nu_obs**(-2.)*distance**2*(1+z)**(-3.))/(np.pi*bmaj*bmin/(4*np.log(2))*(ang_to_pc**2))
    df_0        = df.iloc[line_detc_hex]                          ##### all CO above threshold
    df_0        = df_0[df_0['$\u03A3$_H2'] > 3*Sd_G_rms]

    df_overlap  = df_0[df_0['radio_flux'] >=  thres_factor*rms]     ##### all hex with both radio and CO > threshold   
    df_CO_only  = df_0[ df_0['radio_flux'] <=  thres_factor*rms]    ##### CO only        
    S_only_index = list(set(df[df['radio_flux'] >=  thres_factor*rms].index) - set(line_detc_hex))
    df_S_only    = df.iloc[S_only_index]

    print ('number of CO only  =',len(df_CO_only))
    print ('number of S only   =',len(df_S_only))
    print ('number of overlaps =',len(df_overlap))

    ########### Now calculating linewidth
    print('\nFitting the linewidth...')
    df_disp       = df_overlap     
    indices       = np.array(df_disp.index)
    thrsh = 3
    polyn = 10
    linewidth , linewidth_err, thrsh, polyn = linewidth_fit(indices,rms_avg_1, thrsh,polyn,CO_channel_data, chan_width)

    dataframe={'index':indices,'sigma':linewidth, 'sigma_err':linewidth_err,'threshold':thrsh,'polynomial':polyn}
    df = pd.DataFrame(dataframe)  
    df.set_index('index', inplace=True)

    linew        = df['sigma']
    linew_err    = df['sigma_err']
    perc95_linew = np.percentile(np.array(linew),95)
    perc95_linew_err = np.percentile(np.array(linew_err ),95)

    outlier_linew     = df[df['sigma']>perc95_linew]
    outlier_linew_err = df[df['sigma_err']>perc95_linew_err]

    for i in outlier_linew_err.index:
        thrsh = 2
        k = 0
        polyn_new = 10
        polyn_list = [10, 9, 8, 5, 3]
        rslts = linewidth_fit([i],rms_avg_1,thrsh,polyn_list[k],CO_channel_data, chan_width)
        lnw  = rslts[0]
        lnwr = rslts[1]
        while ((lnwr > 3*perc95_linew_err and k < len(polyn_list)) or (lnw > 2*perc95_linew and k < len(polyn_list))):
            rslts     = linewidth_fit([i],rms_avg_1,thrsh,polyn_list[k],CO_channel_data, chan_width)
            lnw       = rslts[0]
            lnwr      = rslts[1]
            polyn_new = polyn_list[k]
            k = k + 1
        df['sigma'][i]      = lnw
        df['sigma_err'][i]  = lnwr
        df['threshold'][i]  = thrsh
        df['polynomial'][i] = polyn_new

    print('\nCreating linewidth fits plots...')
    new_folder     = 'profiles'
    parent_dir = str(os.getcwdb())[2:-1]
    path = os.path.join(parent_dir, new_folder)

    try:
        os.remove(parent_dir +'/'+ new_folder)
    except:
        print('')

    try:
        os.mkdir(path)
    except:
        print(' ')

    first = 0
    last  = 4
    limit = len(indices)
    stop = 0
    for k in range(limit):
        selection = indices[first:last]
        sel_lim   = indices[-1]
        w = 0
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        if stop == 0:
            for i in range(2):
                if stop == 0:
                    for j in range(2):
                        ax = axs[i, j]
                        individual_fitting_plot(selection[w],df['threshold'][selection[w]], df['polynomial'][selection[w]] ,rms_avg_1,CO_channel_data[selection[w],:],ax)
                        if selection[w] == sel_lim:
                            stop = 1
                            break
                        else:
                            w = w + 1
                else:
                    break
        else:
            plt.close()
            print('Done')
            break
        plt.savefig(parent_dir +'/'+ new_folder +'/'+ 'From_'+ str(selection[0])+'_to_' + str(selection[-1])+'.png')
        plt.close()
        first = last
        last  = last + 4
        if last > limit:
            last = limit
        if last == first or first == limit:
            print('Done')
            break


    df_dispersion = df

    G = 4.30091e-3 #pc M_sun^-1 (km/s)^2
    f = 10./9.  #numerical factor (value extracted from a paper along with formula)
    df_dispersion

    print('\nCalculating kinematic properties...')
    sigma     = df_dispersion['sigma']
    sigma_err = df_dispersion['sigma_err']
    Mass_CO   = np.array(df_disp['Mass_H2'])
    SD_CO     = np.array(df_disp['$\u03A3$_H2'])
    R         = np.sqrt(area)*ang_to_pc

    alpha_vir        = 5*sigma**2*R/(f*G*Mass_CO)
    alpha_vir_err    = (df_disp['Mass_H2_err']/df_disp['Mass_H2']+ 2*sigma_err/sigma )*alpha_vir
    sigma_sqr_over_R = sigma**2/R
    P_turb           = 61.3*(SD_CO)*sigma**2*(ang_to_pc/40)**(-1)
    p_turb_err       = (df_disp['$\u03A3$_H2_err']/df_disp['$\u03A3$_H2']+ 2*sigma_err/sigma )*P_turb


    t_ff      = np.sqrt(3)/(4*G)*df_dispersion['sigma']/df_disp['$\u03A3$_H2']*3.086e13/3.1557e13
    t_ff_err  = t_ff*((df_disp['$\u03A3$_H2_err']/df_disp['$\u03A3$_H2']+ sigma_err/sigma ))
    E_ff      = t_ff*df_disp['SFR']/df_disp['Mass_H2']*1e6
    E_ff_err  = E_ff*(t_ff_err/t_ff+df_disp['SFR_error']/df_disp['SFR']+df_disp['Mass_H2_err']/df_disp['Mass_H2'])


    df_dispersion['alpha_vir']     = alpha_vir
    df_dispersion['P_turb']        = P_turb
    df_dispersion['P_turb_err']    = p_turb_err
    df_dispersion['alpha_vir_err'] = alpha_vir_err
    df_dispersion['t_ff']          = t_ff
    df_dispersion['t_ff_err']      = t_ff_err
    df_dispersion['E_ff']          = E_ff
    df_dispersion['E_ff_err']      = E_ff_err

    try:
        os.remove('kinematics_'+name+'.txt')
    except:
        print('')
        
    df_dispersion.to_csv('kinematics_'+name+'.txt', header=True, index=True, sep='\t', mode='a')
    print ('Done!')
