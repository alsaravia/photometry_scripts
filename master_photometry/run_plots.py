import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


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

spirals    = pd.read_fwf('spiral_r.txt', delimeter = ' ', header = None)
starbursts = pd.read_fwf('starburst_r.txt', delimeter = ' ', header = None)
phngs_d    = pd.read_csv('phangs_KS.txt', sep='\t')
phangs_kin = pd.read_fwf('Sun_2020.txt', delimiter = ' ', header=None)
SFR_n_Gas  = pd.read_csv('Results_SFR_+_GAS_'+name+'.txt', sep="\t")
kinematics = pd.read_csv('kinematics_'+name+'.txt', sep="\t")
indices = np.array(kinematics['index'])

new_folder     = 'PLOTS'
parent_dir = str(os.getcwdb())[2:-1]
path = os.path.join(parent_dir, new_folder)

try:
    os.remove('PLOTS')
except:
    print('')

try:
    os.mkdir(path)
except:
    print(' ')

############################################################### K-S Law ############################################################################
fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(np.array(phngs_d['SD_GAS']),np.array(phngs_d['SFR']),color='gray',marker = '.',s=50, label = '1.5 kpc individual regions from PHANGS galaxies')
ax.scatter(np.power(10,spirals[12]),np.power(10,spirals[8]),color='k',marker = 's',facecolors='brown',s=50,label = 'Normal Spirals from Kennicutt & de los Reyes 2021 ', alpha = 0.5)
ax.scatter(np.power(10,starbursts[2]),np.power(10,starbursts[3]),color='k',marker = 's',facecolors='orange',s=50,label = 'Starburst from Kennicutt & de los Reyes 2021 ',alpha = 0.5)
ax.scatter(np.array(SFR_n_Gas['$\u03A3$_H2'][indices]), np.array(SFR_n_Gas['$\u03A3$_SFR'][indices]), marker = 'h', color = 'k', facecolors = 'crimson', label = name)

####### depletion time lines

x_1GHz  = np.array([0,1e6])
y_1GHz  = x_1GHz*1e6/(1e9)
x_10GHz  = np.array([0,1e6])
y_10GHz  = x_10GHz*1e6/(10*1e9)
x_01GHz  = np.array([0,1e6])
y_01GHz  = x_01GHz*1e6/(0.1*1e9)
plt. plot(x_01GHz,y_01GHz,linestyle = '--', color = 'k', alpha=.5)
plt. plot(x_1GHz,y_1GHz,linestyle = '--', color = 'k', alpha=.5)
plt. plot(x_10GHz,y_10GHz,linestyle = '--', color = 'k', alpha=.5)

plt.text(15000,100,'$t_{dep}$ = 0.1 Gyr', fontsize=15,rotation=35, alpha =0.5)
plt.text(15000,10,'$t_{dep}$ = 1.0 Gyr', fontsize=15,rotation=35, alpha =0.5)
plt.text(15000,1,'$t_{dep}$ = 10 Gyr', fontsize=15,rotation=35, alpha =0.5)

ax.set_xscale("log")
ax.set_yscale("log")
font_style = {'family': 'Times New Roman','color':  'k','weight': 'normal','size': 25,}
plt.ylabel('$\u03A3_{SFR}$[$M_{\odot}$ $y^{-1}$ $kpc^{-2}$]',fontdict = font_style)
plt.xlabel('$\u03A3_{Mol}$[$M_{\odot}$ $pc^{-2}$]',fontdict = font_style)
plt.legend(fontsize=15)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
ax.minorticks_on()
ax.tick_params('both', length=15, width=2, which='major')
ax.tick_params('both', length=8, width=1, which='minor')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'K-S_Law_comparison.png',bbox_inches='tight')
plt.close()


############################################################### Depletion times #################################################################
fig, ax = plt.subplots(figsize = (10, 10))
t_err = (1/SFR_n_Gas['SFE'][indices])**2*SFR_n_Gas['SFE_err'][indices]
ax.scatter(SFR_n_Gas['$\u03A3$_H2'][indices],1/SFR_n_Gas['SFE'][indices],color='k',marker = 'h',facecolors='crimson',s=100 ,label = name)
ax.errorbar(SFR_n_Gas['$\u03A3$_H2'][indices],1/SFR_n_Gas['SFE'][indices], yerr=t_err,xerr=SFR_n_Gas['$\u03A3$_H2_err'][indices], fmt='.',color ='gray', alpha=.2)

ax.set_xscale("log")
ax.set_yscale("log")
plt.ylabel('Depletion time [$yrs$]',fontsize=20)
plt.xlabel('Gas surface density[$M_{\odot}$ $kpc^{-2}$]',fontsize=20)
plt.title('Timescales',fontsize=20)
plt.legend(fontsize = 15)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

ax.minorticks_on()
ax.tick_params('both', length=15, width=2, which='major')
ax.tick_params('both', length=8, width=1, which='minor')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'depletion_times.png',bbox_inches='tight')
plt.close()

############################################################# SFR-P_turb #########################################################################
fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(kinematics['P_turb'],SFR_n_Gas['SFR'][indices],marker='o',color='k',s=100,facecolor='crimson')
ax.errorbar(kinematics['P_turb'],SFR_n_Gas['SFR'][indices],xerr=kinematics['P_turb_err'],yerr=SFR_n_Gas['SFR_error'][indices], fmt='.', color ='gray', alpha=.5)
ax.set_xscale("log")
ax.set_yscale("log")

plt.rcParams["font.family"] = "Times New Roman"
plt.ylabel('SFR [$M_{\odot}$ $y^{-1}$ $kpc^{-2}$]',fontsize=30)
plt.xlabel('Turbulent Pressure [K $cm^{-3}$]',fontsize=30)

plt.yticks(fontsize=30)
plt.xticks(fontsize=30)

ax.minorticks_on()
ax.tick_params('both', length=15, width=2, which='major')
ax.tick_params('both', length=8, width=1, which='minor')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'SFR-P_turb.png',bbox_inches='tight')
plt.close()

############################################################# SFR-P_turb #########################################################################

fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(kinematics['sigma'],SFR_n_Gas['SFR'][indices],marker='o',color='k',s=100,facecolor='crimson')
ax.errorbar(kinematics['sigma'],SFR_n_Gas['SFR'][indices],xerr=kinematics['sigma_err'],yerr=SFR_n_Gas['SFR_error'][indices], fmt='.', color ='gray', alpha=.2)
#ax.set_xscale("log")
ax.set_yscale("log")


plt.rcParams["font.family"] = "Times New Roman"
plt.ylabel('SFR [$M_{\odot}$ $y^{-1}$ $kpc^{-2}$]',fontsize=30)
plt.xlabel('Linewidth [km $s^{-1}$]',fontsize=30)

plt.yticks(fontsize=30)
plt.xticks(fontsize=30)


ax.minorticks_on()
ax.tick_params('both', length=15, width=2, which='major')
ax.tick_params('both', length=8, width=1, which='minor')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'SFR-linewidth.png',bbox_inches='tight')
plt.close()

############################################################# linewidth-Gas #########################################################################

fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(SFR_n_Gas['$\u03A3$_H2'][indices],kinematics['sigma'],marker='o',color='k',s=100,facecolor='crimson')
ax.errorbar(SFR_n_Gas['$\u03A3$_H2'][indices],kinematics['sigma'],xerr=SFR_n_Gas['$\u03A3$_H2_err'][indices],yerr=kinematics['sigma_err'], fmt='.', color ='gray', alpha=.2)
ax.set_xscale("log")
#ax.set_yscale("log")

plt.rcParams["font.family"] = "Times New Roman"
plt.ylabel('Linewidth [km $s^{-1}$]',fontsize=30)
plt.xlabel('$\u03A3$_$H_2$ [$M_{\odot}$ $pc^{-2}$]',fontsize=30)

plt.yticks(fontsize=30)
plt.xticks(fontsize=30)


ax.minorticks_on()
ax.tick_params('both', length=15, width=2, which='major')
ax.tick_params('both', length=8, width=1, which='minor')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'linewidth-Gas.png',bbox_inches='tight')
plt.close()

############################################################# SFE-P_turb #########################################################################

fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(kinematics['P_turb'],SFR_n_Gas['SFE'][indices],marker='o',color='k',s=100,facecolor='crimson')
ax.errorbar(kinematics['P_turb'],SFR_n_Gas['SFE'][indices],xerr=kinematics['P_turb_err'],yerr=SFR_n_Gas['SFE_err'][indices], fmt='.', color ='gray', alpha=.2)
ax.set_xscale("log")
ax.set_yscale("log")

plt.rcParams["font.family"] = "Times New Roman"
plt.ylabel('SFE [$yrs^{-1}$]',fontsize=30)
plt.xlabel('Turbulent Pressure [K $cm^{-3}$]',fontsize=30)

plt.yticks(fontsize=30)
plt.xticks(fontsize=30)


ax.minorticks_on()
ax.tick_params('both', length=15, width=2, which='major')
ax.tick_params('both', length=8, width=1, which='minor')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'SFE-P_turb.png',bbox_inches='tight')
plt.close()

############################################################# SFE-linewidth #########################################################################

fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(kinematics['sigma'],SFR_n_Gas['SFE'][indices],marker='o',color='k',s=100,facecolor='crimson')
ax.errorbar(kinematics['sigma'],SFR_n_Gas['SFE'][indices],xerr=kinematics['sigma_err'],yerr=SFR_n_Gas['SFE_err'][indices], fmt='.', color ='gray', alpha=.2)
#ax.set_xscale("log")
ax.set_yscale("log")

plt.rcParams["font.family"] = "Times New Roman"
plt.ylabel('SFE [$yrs^{-1}$]',fontsize=30)
plt.xlabel('Linewidth [km$s^{-1}$]',fontsize=30)

plt.yticks(fontsize=30)
plt.xticks(fontsize=30)


ax.minorticks_on()
ax.tick_params('both', length=15, width=2, which='major')
ax.tick_params('both', length=8, width=1, which='minor')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'SFE-linewidth.png',bbox_inches='tight')
plt.close()

############################################################# SFR-Virial Parameter #########################################################################

fig, ax = plt.subplots(figsize = (8, 8))
ax.scatter(kinematics['alpha_vir'],SFR_n_Gas['SFR'][indices],marker='o',color='k',s=100,facecolor='crimson')
ax.errorbar(kinematics['alpha_vir'],SFR_n_Gas['SFR'][indices],xerr=kinematics['alpha_vir_err'],yerr=SFR_n_Gas['SFR_error'][indices], fmt='.', color ='gray', alpha=.2)
#ax.set_xscale("log")
ax.set_yscale("log")


plt.rcParams["font.family"] = "Times New Roman"
plt.ylabel('SFR [$M_{\odot}$ $y^{-1}$ $kpc^{-2}$]',fontsize=30)
plt.xlabel('alpha_vir',fontsize=30)

plt.yticks(fontsize=30)
plt.xticks(fontsize=20)


ax.minorticks_on()
ax.tick_params('both', length=15, width=2, which='major')
ax.tick_params('both', length=8, width=1, which='minor')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'SFR-alpha_vir.png',bbox_inches='tight')
plt.close()

############################################################# Gas-P_turb #########################################################################

fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(kinematics['P_turb'],SFR_n_Gas['$\u03A3$_H2'][indices],marker='o',color='k',s=100,facecolor='crimson')
ax.errorbar(kinematics['P_turb'],SFR_n_Gas['$\u03A3$_H2'][indices],xerr=kinematics['P_turb_err'],yerr=SFR_n_Gas['$\u03A3$_H2_err'][indices], fmt='.', color ='gray', alpha=.2)
ax.set_xscale("log")
ax.set_yscale("log")

plt.rcParams["font.family"] = "Times New Roman"
plt.ylabel('$\u03A3$_$H_2$ [$M_{\odot}$ $pc^{-2}$]',fontsize=30)
plt.xlabel('Turbulent Pressure [K $cm^{-3}$]',fontsize=30)

plt.yticks(fontsize=30)
plt.xticks(fontsize=30)


ax.minorticks_on()
ax.tick_params('both', length=15, width=2, which='major')
ax.tick_params('both', length=8, width=1, which='minor')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'Gas-P_turb.png',bbox_inches='tight')
plt.close()

############################################################# t_ff-Gas #########################################################################

fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(SFR_n_Gas['$\u03A3$_H2'][indices],kinematics['t_ff'],marker='o',color='k',s=100,facecolor='crimson')
ax.errorbar(SFR_n_Gas['$\u03A3$_H2'][indices],kinematics['t_ff'],xerr=SFR_n_Gas['$\u03A3$_H2_err'][indices],yerr=kinematics['t_ff_err'], fmt='.', color ='gray', alpha=.2)
ax.set_xscale("log")
#ax.set_yscale("log")

plt.rcParams["font.family"] = "Times New Roman"
plt.ylabel('Free-Fall Time [Myr]',fontsize=30)
plt.xlabel('$\u03A3$_$H_2$ [$M_{\odot}$ $pc^{-2}$]',fontsize=30)

plt.yticks(fontsize=30)
plt.xticks(fontsize=30)


ax.minorticks_on()
ax.tick_params('both', length=15, width=2, which='major')
ax.tick_params('both', length=8, width=1, which='minor')
plt.savefig(parent_dir + '/' + new_folder + '/' + 't_ff-Gas.png',bbox_inches='tight')
plt.close()

############################################################# E_ff-Gas #########################################################################

fig, ax = plt.subplots(figsize = (8, 8))
ax.scatter(SFR_n_Gas['$\u03A3$_H2'][indices],kinematics['E_ff'],marker='o',color='k',s=100,facecolor='crimson')
ax.errorbar(SFR_n_Gas['$\u03A3$_H2'][indices],kinematics['E_ff'],xerr=SFR_n_Gas['$\u03A3$_H2_err'][indices],yerr = kinematics['E_ff_err'], fmt='.', color ='gray', alpha=.2)

ax.set_xscale("log")
ax.set_yscale("log")


plt.rcParams["font.family"] = "Times New Roman"
plt.ylabel('$\epsilon_{ff}$',fontsize=30)
plt.xlabel('$\u03A3$_$H_2$ [$M_{\odot}$ $pc^{-2}$]',fontsize=30)

plt.yticks(fontsize=30)
plt.xticks(fontsize=30)


ax.minorticks_on()
ax.tick_params('both', length=15, width=2, which='major')
ax.tick_params('both', length=8, width=1, which='minor')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'E_ff-Gas.png',bbox_inches='tight')
plt.close()

############################################################# E_ff-P_turb #########################################################################

fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(kinematics['P_turb'],kinematics['E_ff'],marker='o',color='k',s=100,facecolor='crimson')
ax.errorbar(kinematics['P_turb'],kinematics['E_ff'],xerr= kinematics['P_turb_err'],yerr = kinematics['E_ff_err'] , fmt='.', color ='gray', alpha=.2)
ax.set_xscale("log")
ax.set_yscale("log")

plt.rcParams["font.family"] = "Times New Roman"
plt.ylabel('SFE per free-fall time',fontsize=30)
plt.xlabel('P turb',fontsize=30)

plt.yticks(fontsize=30)
plt.xticks(fontsize=30)


ax.minorticks_on()
ax.tick_params('both', length=15, width=2, which='major')
ax.tick_params('both', length=8, width=1, which='minor')
plt.savefig(parent_dir + '/' + new_folder + '/' + 'E_ff-P_turb.png',bbox_inches='tight')
plt.close()

############################################################# Histogram linewidth #########################################################################

fig, ax = plt.subplots(figsize = (10, 10))
plt.hist(kinematics['sigma'],10,histtype='step', color='k',density = True, label = name)
plt.hist(np.array(phangs_kin[(phangs_kin[8] > 10) & (phangs_kin[8] <= 75)][8]),10,histtype='step', color='r', density = True,linestyle = 'dashed', label = 'PHANGS galaxies (>10km/s) from Sun et al.2020')
plt.hist(np.array(phangs_kin[phangs_kin[8]<+75][8]),10,histtype='step', color='r', density = True, label = 'PHANGS galaxies (all) from Sun et al.2020')

plt.ylabel('Probability Density',fontsize=20)
plt.xlabel('Linewidth [km $s^{-1}$]',fontsize=20)

plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.legend(loc='upper right',fontsize = 20)
plt.savefig(parent_dir + '/' + new_folder + '/' + 'linewidth_hist.png',bbox_inches='tight')
plt.close()

############################################################# Histogram P_turb #########################################################################

fig, ax = plt.subplots(figsize = (10, 10))
plt.hist(np.log10(kinematics['P_turb']),10,histtype='step', color='k', density = True, label = name)
plt.hist(np.log10(np.array(phangs_kin[9])),10,histtype='step', color='r', density = True,label = 'PHANGS galaxies from Sun et al.2020')

plt.ylabel('Probability Density',fontsize=20)
plt.xlabel('Log (Turbulent Pressure) [K $cm^{-3}$]',fontsize=20)

plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.legend(loc='upper left',fontsize = 20)
plt.savefig(parent_dir + '/' + new_folder + '/' + 'P_turb_hist.png',bbox_inches='tight')
plt.close()

############################################################# Histogram Virial Paramter #########################################################################

fig, ax = plt.subplots(figsize = (10, 10))
plt.hist(kinematics['alpha_vir'],10,histtype='step', color='k',density = True, label = name)
plt.hist(np.array(phangs_kin[phangs_kin[10]<40][10]),30,histtype='step', color='r', density = True, label = 'PHANGS galaxies from Sun et al.2020')
plt.ylabel('Probability Density',fontsize=20)
plt.xlabel('alpha_vir',fontsize=20)

plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.legend(loc='best',fontsize = 20)
plt.savefig(parent_dir + '/' + new_folder + '/' + 'alpha_vir_hist.png',bbox_inches='tight')
plt.close()

print('Done!')

