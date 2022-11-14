from astropy.table import QTable
import sys, os, contextlib

# show fit parameter for index I source
def pars(roi, I):
           
    name = roi['name'][I]
    print('{:15s}:'.format('Name'), name)
    
    coord = (roi['RAJ2000'][I],roi['DEJ2000'][I])
    print('{:15s}:'.format('RA/DEC'), coord[0],'/',coord[1])
    coord = (roi['GLON'][I],roi['GLAT'][I])
    print('{:15s}:'.format('GLON/GLAT'), f'{coord[0]:.2f}','/',f'{coord[1]:.2f}')
    
    TS = roi['ts'][I]
    print('{:15s}:'.format('TS'), f'{TS:.2f}')
    
    npred = roi['npred'][I]
    print('{:15s}:'.format('Npred'), f'{npred:.2f}')
    
    flux, flux_err = roi['flux'][I], roi['flux_err'][I]
    eflux, eflux_err = roi['eflux'][I], roi['eflux_err'][I]
    print('{:15s}: {:10.4g} +/- {:10.4g}'.format('Flux', flux, flux_err))
    print('{:15s}: {:10.4g} +/- {:10.4g}'.format('EnergyFlux', eflux, eflux_err))
    
    spat = roi['SpatialModel'][I]
    print('{:15s}:'.format('SpatialModel'), f'{spat:s}')
    
    spec = roi['SpectrumType'][I]
    print('{:15s}:'.format('SpectrumType'), f'{spec:s}')
    
    ps = ['Prefactor', 'Index', 'Scale', 'alpha', 'beta', 'norm']
    
    print('Spectral Parameters')
    for i, p in enumerate(roi['param_names'][I]):
        if not p:
            break
        val = roi['param_values'][I][i]
        err = roi['param_errors'][I][i]
        print('{:15s}: {:10.4g} +/- {:10.4g}'.format(ps[i], val, err))
        
        
        
##############################

# get parameter info from the analysis results files
def get_pars(names, extra, loc='./', verb=False):
    
    Ri = []
    # open files with source parameters information
    for n,i in enumerate(names):
        
        if i in extra:
            try:        # get sources analyzed > 500 MeV
                with open(loc+i+'_2/Results_'+i+'_loc.txt') as f:
                    Ri.append(f.readlines())
                    f.close()
            except:
                with open(loc+i+'_2/Results_'+i+'.txt') as f:
                    Ri.append(f.readlines())
                    f.close()
                
        else:
            try:
                with open(loc+i+'/Results_'+i+'_loc.txt') as f:
                    Ri.append(f.readlines())
                    f.close()
            except:
                try:
                    with open(loc+i+'/Results_'+i+'.txt') as f:
                         Ri.append(f.readlines())
                         f.close()
                except:
                    with open(loc+i+'/Resultados_'+i+'.txt') as f:
                     Ri.append(f.readlines())
                     f.close()

    
    ph_i, ph_err = [], [] # photon index and error
    
    Fx, Ferr = [], []    # photon flux and error

    Ex, Eerr = [], []    # energy flux and error
    
    UF, UE = [], []    # upper limits of photon and energy flux
    
    TS = []  # TS

    for l in Ri:
        if verb:
            print(l[0].split(' ')[-1])
            
        pf = l[5].split(' ')[13:16]
        pe = l[5].split(' ')[20:22]

        sf = l[6].split(' ')[7:9]
        se = l[6].split(' ')[14:16]
        
        UF.append(float(l[15].split(' ')[-1]))
        UE.append(float(l[14].split(' ')[-1]))
        
        TS.append(float(l[3].split(' ')[-1]))
        
        for i in l[11].split(' ')[15:]:
            try:
                ph_i.append(float(i))
                break
            except:
                pass

        try:
            ph_err.append(float(l[11].split(' ')[-1]))
        except:
            pass
        
        try:
            Fx.append(float(pf[0]))
        except:
            Fx.append(float(pf[1]))

        try:
            Ferr.append(float(pe[0]))
        except:
            Ferr.append(float(pe[1]))

        try:
            Ex.append(float(sf[0]))
        except:
            Ex.append(float(sf[1]))

        try:
            Eerr.append(float(se[0]))
        except:
            Eerr.append(float(se[1]))
    
    return Fx, Ferr, Ex, Eerr, UF, UE, ph_i, ph_err, TS

##############################
import datetime
import time

# Gets str of Julian date of input given in fermi seconds
def date_day(fermi_s):
    t0 = (2001, 1, 1, 0, 0, 0, 0, 0, 0) # base
    
    if type(fermi_s) == list:
        T = []
        for i in range(len(fermi_s)):
            T.append(f'{datetime.datetime.fromtimestamp(fermi_s[i] + time.mktime(t0)).day}/{datetime.datetime.fromtimestamp(fermi_s[i] + time.mktime(t0)).month}/{datetime.datetime.fromtimestamp(fermi_s[i] + time.mktime(t0)).year}')
                        
    elif type(fermi_s) == int:
        T = f'{datetime.datetime.fromtimestamp(fermi_s + time.mktime(t0)).day}/{datetime.datetime.fromtimestamp(fermi_s + time.mktime(t0)).month}/{datetime.datetime.fromtimestamp(fermi_s + time.mktime(t0)).year}'
    return T
    
# returns number of seconds ater 2001.0
# imput is a tupple (year, month, day, hour-1, ...)
def fermi_time(t, verb=False):
    T = time.mktime(t) - time.mktime((2001, 1, 1, 0, 0, 0, 0, 0, 0)) +5
    if verb:
        print(datetime.datetime.fromtimestamp(time.mktime(t)), '-', T, 'seconds after', datetime.datetime.fromtimestamp(time.mktime((2001, 1, 0, 23, 0, 0, 0, 0, 0))))
    return T

##############################
import numpy as np

# Generate the positions for the mock sources (number, ref. start, slice, minimal distance)
def gen_rd_pos(RA, DEC, n_mocks, thresh):
    ra, dec = [], []
    c = 0
    while c < n_mocks:
        z = 10*np.random.random() - 5
        h = 10*np.random.random() - 5

        # don't include sources too close
        if np.sqrt((z)**2+(h)**2) > thresh:
            ra.append(RA + z)
            dec.append(DEC + h)
            c = c+1            
    
    return ra, dec
    
# get index of target source in the ROI (k)
def roi_i(roi, name):
    K = 'nan'
    # section for printing the source params correctly
    for k in range (len(roi)): 
        if roi['name'][k] == name:
            K = k
    return K

###################
import re

def open_txt(names, loc, par_lines=(7,38), data_lines=(41,165)):
    with open(loc) as f_A:
        all_d = f_A.readlines()
        f_A.close()

    pattern = re.compile(r'\s+')
    par_names = []
    par_desc = []
    for line in all_d[par_lines[0]:par_lines[1]]:
        #print(list(re.sub(pattern, ' ', line.strip()).split()))
        par_names.append(re.sub(pattern, ' ', line.strip()).split()[1])
        par_desc.append(re.sub(pattern, ' ', line.strip()).split()[2])

    # main params
    par_params = {par_names[i]:i for i in range(len(par_names))}
    print(par_params,'\n\n',par_desc,'\n')

    lims = (data_lines[0],data_lines[1])
    
    # all target info (source, parameter)
    target_info = np.empty((len(all_d[lims[0]:lims[1]]),len(par_params)),dtype=object)

    i = 0
    for line in all_d[lims[0]:lims[1]]:
        #print(re.sub(pattern, ' ', line.strip()).split(' ')[0], len(re.sub(pattern, ' ', line.strip()).split(' ')))
        target_info[i] = np.array(list(re.sub(pattern, ' ', line.strip()).split(' ')))
        if re.sub(pattern, ' ', line.strip()).split(' ')[0] != names[i]:
            print(names[i])
        i = i+1
    
    return par_params, par_desc, target_info

###################
from Basic_calc_num import numint

c = 3*10**5 # speed of light in km/s

def dist(z):
    a = 1/(1+z)
    Om = 0.3
    Ol = 0.7
    H0 = 73 # km/s/Mpc
    
    return c/(a**2*H0*np.sqrt(Om*a**-3 + Ol))

# get distance (in Mpc) to galaxy with redshift z0
def Dist(z0):
    return numint.Isimp(dist, 0, z0, 150)

############################
from Basic_calc_num import numint

# assuming gaussian fluctuations
def gauss(x):
    return np.exp(-x**2/(2))/np.sqrt(2*np.pi)

# chance of finding value n sigma away from average due to random fluctuations
def chance(n):
    # integrate using simpson's method
    c = 1 - 2*numint.Isimp(gauss, 0, n, 100)
    
    print('The chance of measuring', np.round(n,3) ,'sigma away from average due to random fluctuations is')
    print(f'{c:.3e}', ', or about 1 in', int(c**-1))
    

##################################
    
