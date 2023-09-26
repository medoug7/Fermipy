##################################################
#	Code to trace TS profile as a function of photon spectrum index
#	for ROIs selected using Fermi-LAT gamma-ray data
#
#	Authors: Douglas F. Carlos & Raniere de Menezes
#
#	Fermitools	v2.0.8
#	Fermipy		v1.0.0
#	
##################################################

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib
import numpy as np
import sys, os, contextlib
from io import StringIO
import warnings
import astropy.io.fits as pyfits
from astropy.table import QTable
import astropy.units as u
from astropy.coordinates import SkyCoord as Coord
import time
from fermipy.gtanalysis import GTAnalysis
from fermi_utils import pars, gen_rd_pos, roi_i
warnings.filterwarnings("ignore")
matplotlib.interactive(True)


# Makes sure the fit quality is 3 with two rounds of fitting:
# one more general for the whole ROI then another more concentrated near the target
def do_fit(gta, source, source_pars=['Prefactor','Index'], free_dist=3, do_opt=True, optimizer='NEWMINUIT', verb=3):
    if verb>1 and do_opt:
        print('Optimizing')


    # Optimize params for all sources in the ROI, except the target
    if do_opt:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                # the bit above this line just supresses the prints of optimize
                gta.optimize(skip=[source])

    gta.free_source('galdiff')
    gta.free_source('isodiff')
    gta.free_sources(distance=10, pars='norm')
    gta.free_source(source, pars=source_pars)
    
    if verb>1:
        print('Fitting')

    fit = gta.fit()
    if fit['fit_quality'] < 3:
        if do_opt:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    # the bit above this line just supresses the prints of optimize
                    gta.optimize(skip=[source])

        gta.free_sources(free=False)
        gta.free_sources(distance=free_dist, pars='norm')
        gta.free_source(source, pars=source_pars)

        fit = gta.fit(min_fit_quality=3, optimizer=optimizer,reoptimize=True)

        if verb>1:
            print('Quality = ', fit['fit_quality'])

    return fit


def trace_profile(Names, n_indexs=12, free_dist=5, opt='MINUIT',
                  main_dir='Stack_dwarfs', config_file='configStack1GeV.yaml',
                  loc_TS = 7, loc_tol=0.8,
                  load=True, trace=True, test_mocks=True, n_mocks=15, verb=3, make_plts=True):
    
    
    ##########################################
    #---Inputs
    #
    #        Names:         (list) List containing names of targets to trace TS profile
    #        
    #        n_index:       (int) Number of photon index values perform fit
    #
    #        free_dist:     (float) Free parameters from sources inside this number of degrees from ROI center
    #
    #        opt:           (str) Fhoice of optimizer ('MINUIT', 'NEWMINUIT')
    #
    #        main_dir:      (str) Name of directory containing the folders for each ROI
    #
    #        config_file:   (str) Standard name of Config yaml file to be used in the analysis
    #    
    #			 loc_TS:			 (float) Threshold to run the localize() method on the target source
    #
    #        loc_tol:       (float) Standard tolerance (in degrees) for distance between target and nearby source
    #                          We use half this value as maximum tolerance for running the localize() method on our targets
    #
    #        load:          (bool) Load previous ROI state from an existing ROI.fits file
    #
    #        trace:         (bool) Actually do the tracing (True), or just verify other functionalities (False) like localize()
    #
    #        test_mocks:    (bool) Actually do the random mock source fitting to later compare with target results
    #
    #        n_mocks:       (int) Number of mock targets to generate per ROI
    #
    #        verb:          (int 0-3) Level of verbosity for printing the output of fermipy and tracing
    #                          recomend 1 for just the important stuff, or 3 for full analysis
    #
    #        make_plts:     (bool) Make png ROI plots like the count and TS maps
    #        
    ###########################################
    
    # number of index values to fit in order to trace
    Index_range = np.linspace(0.5, 5.0, n_indexs)
    
    # we decide to trace the TS profile of a single source if ...
    if trace:
        do_trace = True
    else:
        do_trace = False
        
        
        
    ####### Do the analysis for a given slice of the sample
    for n,i in enumerate(Names):
        # remember if we want to trace all targets or not (may change for a single source)
        if not do_trace:
            trace = False
        else:
            trace = True
            
            
        # load config file and setup the fermipy analysis
        print('\nTime for: ', i)
        gta = GTAnalysis('./'+main_dir+'/'+i+'/'+config_file, logging={'verbosity': verb})
        gta.setup(overwrite=False)
        
        # load ROI?
        if load:
            print('Loading ROI')
            try:
                # load previous ROI state
                gta.load_roi('ROI')
            except:
                # if cant find ROI.fits just move on
                print('Load failed')
                load = False

        # Optimize params for all sources in the ROI, except the target
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                # the bit above this line just supresses the prints of optimize
                gta.optimize(skip=[i])
        
        if not load:        		
            # Search for possible new sources to add to the model
            print('----Finding sources around', i)
            model = {'Index' : 2.0, 'SpatialModel' : 'PointSource'}
            srcs = gta.find_sources(model=model, sqrt_ts_threshold=4.0, min_separation=0.5, multithread=True)
        
        ####### Check if target is inside the galactic plane
        roi = QTable(gta.roi.create_table())
        K = roi_i(roi, i)
        if abs(roi[K]['glat']) < 15:
            print(i, 'is close to galactic plane: b =', np.round(roi[0]['glat'],3))
            
        print(i, 'is', np.round(roi[K]['offset'].value,3), 'deg from the ROI center')
        
        loc_ps = False
        loc_ta = False
        new_ps = False
        ####### Check if close sources are too close to target
        if loc_tol != 0:
            for j in range(len(roi[:2])):
                PS = roi[j]['name'] # name of closest Point Source (PS)
                if 'PS' in roi[j]['name'].split(): # check if it's a known 4FLG source or a potential one found by find_sources() method
                    
                    # check if PS is inside the tolerance
                    if roi[j]['offset'].value < loc_tol:
                        print('Closest source is', PS, ': TS =', int(roi[j]['ts']), ', offset =', np.round(roi[j]['offset'].value,3), 'deg')
                        
                        # run localize() for closest source
                        loc = gta.localize(PS, dtheta_max=0.3, nstep=5, free_radius=1, make_plots=make_plts, update=True, write_fits=False, write_npy=False)
                        loc_coord = Coord(ra = loc['ra'], dec = loc['dec'], unit=(u.deg, u.deg))

                        roi = QTable(gta.roi.create_table())
                        print('Moved', PS,'by: ', np.round(loc['pos_offset'],3),'deg, new offset =', np.round(roi[j]['offset'].value,3),' deg')
                        print('99% positional uncertainty: ', np.round(loc['pos_r99'],3))

                        # if target is inside PS's r99 and substitute it for the new PS
                        if roi[j]['offset'].value < loc['pos_r99']:
                            print('Target is within r99, substituting', i)
                            gta.delete_source(i)
                            gta.delete_source(PS)
                            
                            # variable to know if target has been found by find_sources() method
                            loc_ps = True
                            gta.add_source(i,{ 'ra' : loc['ra'], 'dec' : loc['dec'],
                                'SpectrumType' : 'PowerLaw', 'Index' : roi[j]['dnde_index'],
                            'Scale' : 1000, 'Prefactor' : roi[j]['param_values'][0],
                                'SpatialModel' : 'PointSource' })

                        else:
                            # target and PS are different
                            print('Target is outside r99')
                            
                    else:
                        # PS is outside tolerance for localize()
                        print('----Closest source is', PS, ': TS =', int(roi[j]['ts']), ', offset =', np.round(roi[j]['offset'].value,3))
                        loc_coord = Coord(ra = roi[j]['ra'], dec = roi[j]['dec'], unit=(u.deg, u.deg))
                        
                # PS is a know 4FGL source
                elif '4FGL' in roi[j]['name'].split():
                    print('----Closest source is', PS, ': TS =', int(roi[j]['ts']), ', offset =', np.round(roi[j]['offset'].value,3))
                    loc_coord = Coord(ra = roi[j]['ra'], dec = roi[j]['dec'], unit=(u.deg, u.deg))
                    # Raise a warning
                    if roi[j]['offset'].value < 1:
                        print('WARNING:', PS, 'proximity might interfere with the results for', i)
                    # if it's inside 0.5 deg give up the tracing entirely
                    if roi[j]['offset'].value < 0.15:
                        trace = False
                        
                        
        ############ Fit        
        # Fit the spectral parameters with the chosen optimizer
        fit_results = do_fit(gta=gta, source=i, free_dist=free_dist, optimizer=opt, verb=verb)

        ########### Check for fit problems or convergence issues
        roi = QTable(gta.roi.create_table())
        K = roi_i(roi, i)
        
        if fit_results['fit_quality'] == 3 and abs(roi[K]['dnde_index']) < 4.8 and abs(roi[K]['dnde_index']) > 0.5:
            print('Fit ok, TS =', np.round(roi[K]['ts'], 3))
        else:
            print('WARNING: \n   Fit quality ', fit_results['fit_quality'], 'Photon index = ', roi[K]['dnde_index'])
        #############
        
        # save optimum ROI and plots
        gta.write_roi('ROI', clobber=True, make_plots=make_plts)
        # make TS maps
        if make_plts:
            print('----Making TS Map')
            # load previous ROI dictionary
            c = np.load('./'+main_dir+'/'+i+'/ROI.npy', allow_pickle=True).flat[0]

            # generate TS maps
            model={'SpatialModel' : 'PointSource'}
            maps = []
            for index in [c['sources'][i]['dnde_index']]:
                model['Index'] = index
                maps += [gta.tsmap('ts_map', exclude=[i], model=model, make_plots=True)]
                #maps += [gta.tsmap('fit1', model=model, make_plots=True)]

            gta.plotter.make_tsmap_plots(maps, roi=gta.roi)

        
        if verb > 0:
            gta.print_roi()
            
        ##### get target parameters
        stdout_backup = sys.stdout
        sys.stdout = string_buffer = StringIO()

        pars(roi, K)

        sys.stdout = stdout_backup  # restore old sys.stdout
        string_buffer.seek(0)
        p = string_buffer.read()
        #####
        
        # delete main source
        gta.delete_source(i)
        fit_nresults = gta.fit(optimizer=opt)
        
        ##### Save fit results
        if trace:
            print('----Saving Results')
            f = open('./'+main_dir+'/'+i+'/Results_'+i+'.txt','w')
            f.write(p)
            f.write('\nEnergy flux upper limit (MeV cm-2 s-1): '+str(gta.roi.sources[K]['eflux_ul95']))
            f.write('\nPhoton flux upper limit (cm-2 s-1): '+str(gta.roi.sources[K]['flux_ul95']))
            f.close()
        
        
        ######## perform localize() on the target source if tolerance is not 0
        if loc_tol != 0:
            # re-add target in order to localize it 
            gta.add_source(i, { 'ra' : roi[K]['ra'], 'dec' : roi[K]['dec'],
                            'SpectrumType' : 'PowerLaw', 'Index' : roi[K]['dnde_index'],
                            'Scale' : 1000, 'Prefactor' : roi[K]['param_values'][0],
                            'SpatialModel' : 'PointSource' })

            
            # optimize ROI
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    # (the bit above this line just supresses the prints from optimize
                    gta.optimize()
            
            #print(i,': TS =', int(roi[K]['ts']))
            
            # run localize for target if it's significant enough (TS > 7, or signif > 2.6 sig)
            if roi[K]['ts'] > loc_TS or loc_ps:
                print('----Localizing', i)
                loc_t = gta.localize(i, dtheta_max=0.3, free_radius=2, make_plots=make_plts, update=False, write_fits=False, write_npy=False)
                
                print('Moved', roi[K]['name'],'by: ', np.round(loc_t['pos_offset'],3),'deg')
                print('99% positional uncertainty: ', np.round(loc_t['pos_r99'],3), 'deg')

                # check if new position is inside the tolerance to update model
                if loc_t['pos_offset'] < 0.5 and loc_t['pos_offset'] < loc_t['pos_r99']:
                    
                    loc_ta = True # variable to remember if the localize() method was succeseful 
                    
                    gta.delete_source(i)
                    gta.add_source(i,{ 'ra' : loc_t['ra'], 'dec' : loc_t['dec'],
                        'SpectrumType' : 'PowerLaw', 'Index' : roi[K]['dnde_index'],
                        'Scale' : 1000, 'Prefactor' : roi[K]['param_values'][0],
                        'SpatialModel' : 'PointSource' })
                        
                    # Fit the spectral parameters with the chosen optimizer
                    fit_results = do_fit(gta=gta, source=i, free_dist=free_dist, optimizer=opt, verb=verb)

                    ########### Check for fit problems or convergence issues
                    roi = QTable(gta.roi.create_table())
                    K = roi_i(roi, i)

                    if fit_results['fit_quality'] == 3 and abs(roi[K]['dnde_index']) < 4.8 and abs(roi[K]['dnde_index']) > 0.5:
                        print('Fit ok, TS =', np.round(roi[K]['ts'], 3))
                    else:
                        print('WARNING: \n   Fit quality ', fit_results['fit_quality'], 'Photon index = ', roi[K]['dnde_index'])
                    #############

                    target_coord = Coord(ra = loc_t['ra'], dec = loc_t['dec'], unit=(u.deg, u.deg))
                    sep = target_coord.separation(loc_coord).to(u.deg)

                    print('Dist. from', PS,'=', np.round(sep.value,3), 'deg')
                    
                    # save optimum ROI and plots
                    gta.write_roi('ROI_loc', clobber=True, make_plots=make_plts)

                ###### If original position lies outside the target's r99, then localize() has failed
                #         in this case we'll treat this TS peak as a new Point Source
                #         close to the target and proceed with the analysis
                elif loc_t['pos_offset'] > loc_t['pos_r99']:
                    print('WARNING: Original position is outside r99, localize failed \nAdding new point source')
                    loc_ta = False  # variable to remember if the localize() method was succeseful
                    new_ps = True   # variable to know if theres a new PS
                    
                    gta.add_source('PS_tooclose1',{ 'ra' : loc_t['ra'], 'dec' : loc_t['dec'],
                        'SpectrumType' : 'PowerLaw', 'Index' : roi[K]['dnde_index'],
                        'Scale' : 1000, 'Prefactor' : roi[K]['param_values'][0],
                        'SpatialModel' : 'PointSource' })
                        
                    # Fit the spectral parameters with the chosen optimizer
                    fit_results = do_fit(gta=gta, source=i, free_dist=free_dist, optimizer=opt, verb=verb)

                    ########### Check for fit problems or convergence issues
                    roi = QTable(gta.roi.create_table())
                    K = roi_i(roi, i)

                    if fit_results['fit_quality'] == 3 and abs(roi[K]['dnde_index']) < 4.8 and abs(roi[K]['dnde_index']) > 0.5:
                        print('Fit ok, TS =', np.round(roi[K]['ts'], 3))
                    else:
                        print('WARNING: \n   Fit quality ', fit_results['fit_quality'], 'Photon index = ', roi[K]['dnde_index'])
                    #############
                    
                    # save optimum ROI and plots
                    gta.write_roi('ROI_loc', clobber=True, make_plots=make_plts)
                    
                else:
                    print('No convergence: localize() placed', roi[K]['name'], '> ', np.round(loc_tol/2, 2), 'deg away from original position.')
                    loc_ta = False
        
        
        ######## re-print and save parameters if localize() succeeded
        if loc_ta or new_ps:
            if verb > 0:
                gta.print_roi()

            ##### get target params
            stdout_backup = sys.stdout
            sys.stdout = string_buffer = StringIO()

            pars(roi, K)

            sys.stdout = stdout_backup  # restore old sys.stdout
            string_buffer.seek(0)
            p = string_buffer.read()
            #####
            print(i,': (after localizing) TS =', int(roi[K]['ts']))


            if make_plts:
                print('----Making TS Map')
            # load previous ROI dictionary
            c = np.load('./'+main_dir+'/'+i+'/ROI.npy', allow_pickle=True).flat[0]

            # generate TS maps
            model={'SpatialModel' : 'PointSource'}
            maps = []
            for index in [c['sources'][i]['dnde_index']]:
                model['Index'] = index
                maps += [gta.tsmap('ts_map2', exclude=[i], model=model, make_plots=True)]
                #maps += [gta.tsmap('fit1', model=model, make_plots=True)]

            gta.plotter.make_tsmap_plots(maps, roi=gta.roi)

            # delete main source
            gta.delete_source(i)
            fit_nresults = gta.fit(optimizer=opt)
        
        elif loc_tol != 0:        
            # delete main source
            gta.delete_source(i)
            fit_nresults = gta.fit(optimizer=opt)
            
        
        ##### save likelihood values to txt
        # calculates likel. for main source and for the null hypothesis
        if loc_ta:
            print('----Saving Results')
            f = open('./'+main_dir+'/'+i+'/Results_'+i+'_loc.txt','w')
            f.write(p)
            f.write('\nEnergy flux upper limit (MeV cm-2 s-1): '+str(gta.roi.sources[K]['eflux_ul95']))
            f.write('\nPhoton flux upper limit (cm-2 s-1): '+str(gta.roi.sources[K]['flux_ul95']))
            f.close()
        
        elif new_ps:
            print('----Saving Results')
            f = open('./'+main_dir+'/'+i+'/Results_'+i+'.txt','w')
            f.write(p)
            f.write('\nEnergy flux upper limit (MeV cm-2 s-1): '+str(gta.roi.sources[K]['eflux_ul95']))
            f.write('\nPhoton flux upper limit (cm-2 s-1): '+str(gta.roi.sources[K]['flux_ul95']))
            f.close()
        
        
        ###########################################################################
        # Here we setup the lists of relevant values and then trace the TS profile by varying the index
        
        if trace:
            f = open('./'+main_dir+'/'+i+'/TSprofile_'+i+'.txt','w')
            
            a = fit_results['loglike']
            f.write('#Overall_loglike_testSource: '+str(a))

            b = fit_nresults['loglike']
            f.write('\n#Overall_loglike_nullHypothesis: '+str(b))
            TSlikelihood = 2*(a-b)       # calculating the TS from the loglikel.
            f.write('\n#TS_from_likelihood: '+str(TSlikelihood))
            f.write('\n#Lines below are: Index, TStarget') #, LoglikeTarget, LoglikNullTarget, TSrandom, Loglikerandom, LoglikeNullrandom and repeats')
            
            TSrandom = np.zeros([n_mocks, n_indexs]) # list for TS of mock sources
            LogLikerandom = np.zeros([n_mocks, n_indexs])
            LogLikerandomNull = np.zeros([n_mocks, n_indexs])
            TS = []
            LogLikeTarget = []
            LogLikeTargetNull = []
            contador = 0

            print('----Tracing TS profile')
            # for each spectral index:
            for j in Index_range:
                #print('Index: -', j)
                
                if loc_ta:
                    # re-add target on the new position
                    gta.add_source(i,{ 'ra' : loc_t['ra'], 'dec' : loc_t['dec'],
                            'SpectrumType' : 'PowerLaw', 'Index' : j,
                            'Scale' : 1000, 'Prefactor' : 1e-13,
                            'SpatialModel' : 'PointSource' })
                
                elif loc_ps:
                    # re-add the target on the position of nearby PS
                    gta.add_source(i,{ 'ra' : loc['ra'], 'dec' : loc['dec'],
                            'SpectrumType' : 'PowerLaw', 'Index' : j,
                            'Scale' : 1000, 'Prefactor' : 1e-13,
                            'SpatialModel' : 'PointSource' })
                else:
                    # re-add the target source and vary its index
                    gta.add_source(i,{ 'ra' : roi[K]['ra'], 'dec' : roi[K]['dec'],
                            'SpectrumType' : 'PowerLaw', 'Index' : j,
                            'Scale' : 1000, 'Prefactor' : 1e-13,
                            'SpatialModel' : 'PointSource' })
                            
                            
                # Fit the spectral parameters with the chosen optimizer
                fit_results = do_fit(gta=gta, source=i, source_pars=['Prefactor'], free_dist=free_dist, do_opt=False, optimizer=opt, verb=verb)

                ########### Check for fit problems or convergence issues
                roi = QTable(gta.roi.create_table())
                K = roi_i(roi, i)

                if fit_results['fit_quality'] == 3 and abs(roi[K]['dnde_index']) < 4.8 and abs(roi[K]['dnde_index']) > 0.5:
                    print('Fit ok, TS =', np.round(roi[K]['ts'], 3))
                else:
                    print('WARNING: \n   Fit quality ', fit_results['fit_quality'])
                #############
                
                if verb > 0:
                    print('Index =', np.round(j,3))
                    
                elif verb > 2:
                    print('Index =', np.round(j,3))
                    gta.print_roi()

                # search for index of target again (because it can change)
                roi = QTable(gta.roi.create_table())
                K = roi_i(roi, i)

                TS.append(gta.roi.sources[K]['ts'])
                LogLikeTarget.append(fit_results['loglike'])

                gta.delete_source(i)
                fit_results = gta.fit(optimizer=opt)
                LogLikeTargetNull.append(fit_results['loglike'])
                
                # below we trace the profiles of mock targets as well
                if test_mocks:
                    # Generate the positions for the mock sources
                    if contador == 0:
                        print('----Generating mock sources')
                        ra, dec = gen_rd_pos(roi[K]['ra'], roi[K]['dec'], n_mocks, thresh=1.5)

                    # save coordinates of mock sources
                    g = open('./'+main_dir+'/'+i+'/Mock_coords.txt','w')
                    g.write('Mock source coordinates: \n'+str(Names[n]))
                    g.write('\n'+str(ra))
                    g.write('\n'+str(dec))
                    g.close()

                    # add mock targets with same varying index as targer
                    for c in np.arange(n_mocks):
                        gta.add_source('random_'+str(c),{ 'ra' : ra[c], 'dec' : dec[c],
                                    'SpectrumType' : 'PowerLaw', 'Index' : j,
                                    'Scale' : 1000, 'Prefactor' : 1e-13,
                                    'SpatialModel' : 'PointSource' })
                                    
                        gta.free_source('random_'+str(c),free=False)
                        gta.free_source('random_'+str(c),pars='norm') #libera soh a normalizacao da fonte falsa
                        fit_results = gta.fit(optimizer=opt)

                        TSrandom[c][contador] = gta.roi['random_'+str(c)]['ts']
                        LogLikerandom[c][contador] = fit_results['loglike'] #loglike do modelo com a fonte falsa

                        gta.delete_source('random_'+str(c)) #deleta a fonte falsa do modelo
                        fit_results = gta.fit(optimizer=opt)
                        LogLikerandomNull[c][contador] = fit_results['loglike'] #loglike do modelo sem a fonte falsa

                    contador = contador + 1
                
            
            # Save results for the main target:    
            f.write('\n'+str(Index_range.tolist())[1:-1])
            f.write('\n'+str(TS)[1:-1])
            #f.write('\n'+str(LogLikeTarget)[1:-1]) 
            #f.write('\n'+str(LogLikeTargetNull)[1:-1])
            f.close()

            if test_mocks:
                # Save results for mock targets:
                g = open('./'+main_dir+'/'+i+'/TSrd_'+i+'.txt','w')
                for c in np.arange(n_mocks):
                    g.write(str(TSrandom[c].tolist())[1:-1]+'\n')
                    #g.write('\n'+str(LogLikerandom[c].tolist())[1:-1])
                    #g.write('\n'+str(LogLikerandomNull[c].tolist())[1:-1])
                
                g.close()
            
        print('Done!\n')


