# ----------------------------------------------------------------------------
# Filename: analyze_ising_data.py
# Author: Luis Alvarez
# This program reads in data files that have been generated
# by Ising_ND.py. Once read, the data is analyzed, 
# observables are extracted, the critical temperature,
# and correlation length with critical exponent nu is 
# displayed. 

# <TODO> Create docstrings

from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lmfit

# Reading in data -------------------------------------------------------------

def read_data(arr,info):
    mag_info, Hc_info, two_point_info, chi_info, energy_info = arr

    mag = pd.read_csv(mag_info + info)
    fst_val = [mag.columns.values.astype(np.float)]
    mag = np.concatenate([fst_val,mag],axis=0)
    mag = pd.DataFrame(mag,columns=['temp','mag'])
    
    Hc = pd.read_csv(Hc_info + info)
    fst_val = [Hc.columns.values.astype(np.float)]
    Hc = np.concatenate([fst_val,Hc],axis=0)
    Hc = pd.DataFrame(Hc,columns=['temp','Hc'])

    two_point = pd.read_csv(two_point_info + info)
    two_point = two_point.drop('Unnamed: 0',1)
    
    chi = pd.read_csv(chi_info + info)
    fst_val = [chi.columns.values.astype(np.float)]
    chi = np.concatenate([fst_val,chi],axis=0)
    chi = pd.DataFrame(chi,columns=['temp','chi'])

    energy = pd.read_csv(energy_info + info)
    fst_val = [energy.columns.values.astype(np.float)]
    energy = np.concatenate([fst_val,energy],axis=0)
    energy = pd.DataFrame(energy,columns=['temp','energy'])

    return mag, Hc, two_point, chi, energy
    
def return_info_string(n,dim,MC_trials,interval,T_end,Tinit):
    return 'size_' + str(n) + '_dim_' + str(dim) + '_MC_' + str(MC_trials) +'_inter_' + str(interval) +  '_T' +str(T_end-Tinit) + '.csv'

# Fitting for the correlation lengths -----------------------------------------

def model_correlation_l(A,gamma,nu,d,domain):
    return A - (gamma)*domain - (d - 2 + nu)*np.log(domain)
    #return A - gamma*domain

def resid_correlation_l(params,data,d,domain):
    A = params['A'].value
    gamma = params['gamma'].value
    nu = params['nu'].value
    #nu = 1
    model = model_correlation_l(A,gamma,nu,d,domain)
    return (data-model)
    
def fit_correlation_l(data,initial,d,domain):
    params = lmfit.Parameters()
    params.add('A',value=initial[0])
    params.add('gamma',value=initial[1])
    params.add('nu',value=initial[2])
    result = lmfit.minimize(resid_correlation_l, params, args=[data,d,domain])
    return result
  
def obtain_correlation_l(two_p,origin,dim,initial):
    # Correlation length for each temperature value
    correlation_l = np.zeros(two_p.shape[1])
    domain = np.array(xrange(origin,two_p.shape[0]))
    count = 0
    for col in two_p:
        data = np.abs((two_p[col][origin:]))
        if np.all(data==0) == True or np.sum(data) < 0.0: 
            correlation_l[count] = 0
            count = count + 1
            continue
        indices_zero = data == 0
        rand_small = np.random.uniform(0.000000001,0.00000002,size=indices_zero.shape)
        data[data == 0] = rand_small
        data = np.log(data)
        result = fit_correlation_l(data,initial,dim,domain)
        correlation_l[count] = 1/result.values['gamma']
        if correlation_l[count] < 0: correlation_l[count] = 0
        count = count + 1
        
    return correlation_l

# Fitting for critical exponents ----------------------------------------------

def model_crit_exp(A,nu,Tc,domain,side):
    if side == 0: # From the left side , Tc - T
        return A*(Tc/(Tc-domain))**(nu)
    if side == 1: # From tjhe right side, T - Tc
        return A*(Tc/(domain-Tc))**(nu)
        
def resid_crit_exp(params,data,Tc,domain,side):
    A = params['A'].value
    nu = params['nu'].value
    model = model_crit_exp(A,nu,Tc,domain,side)
    return data-model
    
def fit_crit_exp(data,initial,Tc,domain,side):
    params = lmfit.Parameters()
    params.add('A',value=initial[0])
    params.add('nu',value=initial[1])
    result = lmfit.minimize(resid_crit_exp, params, args=[data,Tc,domain,side])
    return result

# -----------------------------------------------------------------------------
        
if __name__ == '__main__':
    
    # File information
    mag_info_met = 'data_dir/mag_whole_met_'    
    two_point_info_met = 'data_dir/two_point_met_'
    chi_info_met = 'data_dir/chi_met_'
    Hc_info_met = 'data_dir/Hc_met_'
    energy_info_met = 'data_dir/energy_met_'

    arr_met = [mag_info_met,Hc_info_met,two_point_info_met,chi_info_met,energy_info_met]    
        
    mag_info_w = 'data_dir/mag_whole_wolff_'    
    two_point_info_w = 'data_dir/two_point_wolff_'
    chi_info_w = 'data_dir/chi_w_'
    Hc_info_w = 'data_dir/Hc_w_'
    energy_info_w = 'data_dir/energy_w_'

    arr_w = [mag_info_w,Hc_info_w,two_point_info_w,chi_info_w,energy_info_w]    
    
    # Parameters in consideration
    n = 6
    dim = 5
    MC_trials = 200 
    interval = 20
    Tinit = 0.01
    T_end = 9
    info_str = return_info_string(n,dim,MC_trials,interval,T_end,Tinit)
    
    met = True
    w = False
    
    # Read data
    if met == True:
        mag_met, Hc_met, two_p_met, chi_met, energy_met = read_data(arr_met,info_str)
        T = mag_met['temp']
        mag = mag_met
        Hc = Hc_met
        two_p = two_p_met
        chi = chi_met
        energy = energy_met
        name = 'Metropolis'
    if w == True:
        mag_w, Hc_w, two_p_w, chi_w, energy_w = read_data(arr_w,info_str)
        T = mag_w['temp']
        mag = mag_w
        Hc = Hc_w
        two_p = two_p_w
        chi = chi_w
        energy = energy_w
        name = 'Wolff'
        
    
    # Fit a correlation length to all the correlation function values over
    # distance
    
    correlation_l = obtain_correlation_l(two_p,1,dim,[1,1,1])

    Tc_index = np.where(correlation_l == np.max(correlation_l))[0][0]
    Tc = T[Tc_index]
    
    # Floor all the other peaks to the mean value
    new_corr_l = np.copy(np.abs(correlation_l))
    size = new_corr_l.shape
    size = (size[0] -1,)
    # Smoothing 
    new_corr_l[new_corr_l < np.max(new_corr_l)] = np.mean(new_corr_l)*(1 + np.random.uniform(-0.5,0.5,size=size))
    
   
    # Now fit the resultant to extract critical exponents.
    indices_l = T.values <= Tc
    domain_l = T[indices_l].values
    domain_l[len(domain_l)-1] = domain_l[len(domain_l)-1] - 0.00001
    data_l = np.abs(new_corr_l[indices_l])
    
    crit_exp_result_l = fit_crit_exp(data_l,[np.max(new_corr_l),1],Tc,domain_l,0)
    # Fit on the right side
    indices_r = T.values >= Tc
    domain_r = T[indices_r].values
    domain_r[len(domain_r)-1] = domain_r[len(domain_r)-1] + 0.00001
    data_r = np.abs(new_corr_l[indices_r])
    
    crit_exp_result_r = fit_crit_exp(data_r,[np.max(new_corr_l),1],Tc,domain_r,1)
    
    nu_est = (crit_exp_result_l.values['nu'] + crit_exp_result_r.values['nu'])/2
    
    # Plot correlation length information
    plt.figure()
    fs = 15
    plotter = np.abs(new_corr_l)
    plotter[plotter == -np.inf] = 0
    plt.plot(T.values,plotter,'--o')
    plt.title('Log Correlation Length vs Temperature\n' +  r'$\bar{\nu} =$ ' + str(nu_est),fontsize=fs)
    plt.xlabel('Temperature',fontsize=fs)
    plt.ylabel(r'$\xi$',fontsize=fs)
    plt.ylim([-0.1,1.5*np.max(plotter)])
    plt.show()
    
    
    # After having determined the critical temperature, plot relevant observables
    plot = True
    if plot == True:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    
        gs = gridspec.GridSpec(20,2)
        fig = plt.figure(figsize=(15,11))
        plt.suptitle(name + ' Monte Carlo Simulation on a\n' +
                     str(dim) + 'D Lattice of Size ' + str(n) +' With ' +
                     str(MC_trials) +' Monte Carlo Trials',fontsize=fs)
        
        ax1 = fig.add_subplot(gs[0:5,0])     
        plt.title('Magnetization',fontsize=fs)
        plt.plot(T,mag['mag'],'--o')
        plt.xlabel('Temperature',fontsize=fs); plt.ylabel(r'$\frac{<M>}{N}$',fontsize=fs)
        plt.legend([name],prop={'size':fs-5})
    
        
        ax2 = fig.add_subplot(gs[7:12,0])
        domain = np.linspace(0,two_p.shape[0],two_p.shape[0])
        model = 1/((domain)**(1/4))
        model[0] = 1
        two_p[str(Tc)].values[0] = 1
        plt.title('Two-Point Correlation at Estimated $T_c$',fontsize=fs)
        plt.plot(domain,np.abs(two_p[str(Tc)]),'--o',
                 domain,model,'-o')
        plt.xlabel(r'$|r_i - r_j|$',fontsize=fs); plt.ylabel(r'$<s_os_r>$',fontsize=fs)
        plt.legend([name,'True'],prop={'size':fs-5})
    
        ax3 = fig.add_subplot(gs[0:5,1])
        plt.title('Specific Heat',fontsize=fs)
        plt.plot(T,Hc['Hc'],'--o')
        plt.xlabel('Temperature',fontsize=fs); plt.ylabel(r'$C$',fontsize=fs)
        plt.legend([name],prop={'size':fs-5})
    
        ax4 = fig.add_subplot(gs[7:12,1])
        plt.title('Magnetic Susceptibility',fontsize=fs)
        plt.plot(T,chi['chi'],'--o')
        locs,labels = plt.yticks()
        if w == True:
            plt.yticks(locs, map(lambda x: "%.1f" % x,locs*1e12))
        if met == True:
            plt.yticks(locs, map(lambda x: "%.1f" % x,locs*1e6))
        plt.xlabel('Temperature',fontsize=fs); plt.ylabel(r'$\chi\/(1E-6)$',fontsize=fs)
        plt.legend([name],prop={'size':fs-5})
        
        ax5 = fig.add_subplot(gs[14:19,0:])
        plt.title('Energy',fontsize=fs)
        plt.ylim([energy['energy'][1:].min(), energy['energy'].max()+2000])
        plt.plot(T[1:],energy['energy'][1:],'--o')
        plt.xlabel('Temperature',fontsize=fs); plt.ylabel('Energy',fontsize=fs)
        plt.legend([name],prop={'size':fs-5})
            
        plt.show()
    
