import os, glob, sys
from platform import node
import numpy as np
import numpy.ma as ma
import pandas as pd
import geopandas as gpd
import cartopy.io.shapereader as shpreader
import regionmask
import warnings
warnings.filterwarnings("ignore")
from scipy import signal
sys.path.append(os.path.abspath("/dfs2/Efi/phongl3/Data/CMIP6/Monthly-Ensembles/PR/python"))
from funcs import *


if __name__ == "__main__":    
    hist_exp = 'historical'    
    experiments = ['ssp370', 'ssp245']
    pptdir = '/dfs2/Efi/phongl3/Data/CMIP6/Monthly-Ensembles/PR/FULL1/'
    sstdir = '/dfs2/Efi/phongl3/Data/CMIP6/Monthly-Ensembles/TOS/FULL1/'
    nev_sst = 5
    
    year_s = [1964,2049]
    year_e = [2014,2099]
    
    nodeid = int(sys.argv[1])  
    print(nodeid)  
    linear = False
    if linear:
        resdir = '/dfs2/Efi/phongl3/Data/CMIP6/Monthly-Ensembles/PR/python/RESULT_Linear/'
    else:
        resdir = '/dfs2/Efi/phongl3/Data/CMIP6/Monthly-Ensembles/PR/python/RESULT_MovAve/'

    fp = '/dfs2/Efi/phongl3/Data/GIS/Shapefiles/Continents.shp'
    shp_cont = gpd.read_file(fp)
    shape_cont = list(shpreader.Reader(fp).geometries())
    os.makedirs(resdir+'/historical', exist_ok=True)

    for exp in experiments[0:1]:
        os.makedirs(resdir+'/TimeSeries/'+exp,exist_ok=True)    
        print(exp)
        df = pd.read_csv('CSV/'+exp+'_variants.csv')
        modelid = df.index.values
        modelnames = df['variant'].values
        ind = np.where(modelid%8==nodeid)[0]
        os.makedirs(resdir+'/'+exp,exist_ok=True)
        input_models = []
        fn_all = modelnames[ind]
        for fn in fn_all:
            model = fn.split('/')[-2]
            variant = fn.split('/')[-1]
            inp = pptdir+'/'+exp+'/'+model+'/'+variant
            out = resdir+'/'+exp+'/'+model+'/'+variant
            inp_ncfiles = glob.glob(inp+'/*.nc')
            out_ncfiles = glob.glob(out+'/*.nc')
            if len(out_ncfiles)==0:
                input_models.append(inp)
            elif len(out_ncfiles)>1:
                for l,fnc in enumerate(out_ncfiles):
                    os.remove(fnc)
                input_models.append(inp)
            else:
                file_size = os.path.getsize(out_ncfiles[0])
                if file_size==0:
                    input_models.append(inp)
                    break

        mod_lst = SortVariant(input_models)
        
        for i, modelname in enumerate(mod_lst):
            print(modelname, flush=True)
            model = modelname.split('/')[-2]
            variant = modelname.split('/')[-1]
            try:
                pptfile = glob.glob(pptdir+exp+'/'+model+'/'+variant+'/pr'+'*.nc')[0]
                sstfile = glob.glob(sstdir+exp+'/'+model+'/'+variant+'/tos'+'*.nc')[0]
                # Load PPT data
                ppt, latp, lonp, day, month, year = LoadNetCDF(pptfile, 'pr', 1950, 2100, sf=86400*30)

                # Extract seasonal anomaly and detrend
                ppt_mam,ppt_jja,ppt_son,ppt_djf,ppt_anom_mam,ppt_anom_jja,ppt_anom_son,ppt_anom_djf\
                    = ExtractSeasonal(ppt,day,month,year,opt='sum',masked=False,linear=linear)
                num_yr,M,N = ppt_mam.shape
                print('Checkpoint')
                # Load SST data
                sst, lats, lons, day, month, year = LoadNetCDF(sstfile, 'tos', 1950, 2100, sf=1)

                # Extract seasonal anomaly and detrend
                sst_mam,sst_jja,sst_son,sst_djf,sst_anom_mam,sst_anom_jja,sst_anom_son,sst_anom_djf =\
                    ExtractSeasonal(sst,day,month,year,opt='mean',masked=True,linear=linear)
                
                ind = [ [0,1],[0,2],[0,3],[1,2],[1,3],[2,3] ]
                # ind = [ [0,1] ]
                num_sce = len(ind)
                            
                dirout = resdir+'/TimeSeries/'+exp+'/'+model+'/'+variant
                os.makedirs(dirout,exist_ok=True)                
                fout = dirout+'/'+model+'_'+exp+'_'+variant+'.nc'
                
                data_ocean = [('sst_anom_mam', sst_anom_mam),('sst_anom_jja', sst_anom_jja),('sst_anom_son', sst_anom_son),('sst_anom_djf', sst_anom_djf),
                                ('sst_mam', sst_mam),('sst_jja', sst_jja),('sst_son', sst_son),('sst_djf', sst_djf)]
                data_land = [('ppt_anom_mam', ppt_anom_mam),('ppt_anom_jja', ppt_anom_jja),('ppt_anom_son', ppt_anom_son),('ppt_anom_djf', ppt_anom_djf),
                                ('ppt_mam', ppt_mam),('ppt_jja', ppt_jja),('ppt_son', ppt_son),('ppt_djf', ppt_djf)]
                WriteNetCDF_Maps(fout, model, lats, lons, latp, lonp, nev_sst, num_yr, num_sce,None, None, data_ocean, data_land)
                            
                # Extract historical
                for yr in range(2):
                    print('yr=',yr,flush=True)
                    inds = year_s[yr]-1950
                    inde = year_e[yr]-1950
                    ppt_anom_mam2 = ppt_anom_mam[inds:inde,:,:]
                    ppt_anom_jja2 = ppt_anom_jja[inds:inde,:,:]
                    ppt_anom_son2 = ppt_anom_son[inds:inde,:,:]
                    ppt_anom_djf2 = ppt_anom_djf[inds:inde,:,:]

                    sst_anom_mam2 = sst_anom_mam[inds:inde,:,:]
                    sst_anom_jja2 = sst_anom_jja[inds:inde,:,:]
                    sst_anom_son2 = sst_anom_son[inds:inde,:,:]
                    sst_anom_djf2 = sst_anom_djf[inds:inde,:,:]

                    ppt_mam2 = ppt_mam[inds:inde,:,:]
                    ppt_jja2 = ppt_jja[inds:inde,:,:]
                    ppt_son2 = ppt_son[inds:inde,:,:]
                    ppt_djf2 = ppt_djf[inds:inde,:,:]
                    
                    num_yr2, M, N = ppt_anom_mam2.shape
                    ppt_anom_all = [ppt_anom_mam2,ppt_anom_jja2,ppt_anom_son2,ppt_anom_djf2]
                    
                    # Mask the land cells
                    mask_cont = regionmask.mask_geopandas(shp_cont, lonp, latp, wrap_lon=True)
                    maskppt_land = mask_cont.copy().data
                    maskppt_land[maskppt_land == 6] = np.nan
                    maskppt_land[maskppt_land >= 0] = 1

                    ppt_mean_annual = np.mean(ppt_mam2,axis=0) + np.mean(ppt_jja2,axis=0) + np.mean(ppt_son2,axis=0) + np.mean(ppt_djf2,axis=0)
                    ind_cv = (ppt_mean_annual>=250).astype(float)
                    ind_cv2 = 1-ind_cv
                    ind_cv[ind_cv==0] = np.nan
                    
                    # Principal Component Analysis
                    evec_sst_mam,eval_sst_mam,PCs_sst_mam,trace_sst_mam,masksst = PCA_kernel(sst_anom_mam2,nev_sst,masked=True)
                    evec_sst_jja,eval_sst_jja,PCs_sst_jja,trace_sst_jja,masksst = PCA_kernel(sst_anom_jja2,nev_sst,masked=True)
                    evec_sst_son,eval_sst_son,PCs_sst_son,trace_sst_son,masksst = PCA_kernel(sst_anom_son2,nev_sst,masked=True)
                    evec_sst_djf,eval_sst_djf,PCs_sst_djf,trace_sst_djf,masksst = PCA_kernel(sst_anom_djf2,nev_sst,masked=True)

                    evec_sst_all = [evec_sst_mam,evec_sst_jja,evec_sst_son,evec_sst_djf]
                    eval_sst_all = [eval_sst_mam,eval_sst_jja,eval_sst_son,eval_sst_djf]
                    trace_sst_all = [trace_sst_mam,trace_sst_jja,trace_sst_son,trace_sst_djf]
                    PCs_sst_all = [PCs_sst_mam,PCs_sst_jja,PCs_sst_son,PCs_sst_djf]

                    var_sst_mam = eval_sst_mam/trace_sst_mam*100
                    var_sst_jja = eval_sst_jja/trace_sst_jja*100
                    var_sst_son = eval_sst_son/trace_sst_son*100
                    var_sst_djf = eval_sst_djf/trace_sst_djf*100

                    evec_sst2d_mam = np.nan*np.zeros((nev_sst,lats.size,lons.size))
                    evec_sst2d_jja = np.nan*np.zeros((nev_sst,lats.size,lons.size))
                    evec_sst2d_son = np.nan*np.zeros((nev_sst,lats.size,lons.size))
                    evec_sst2d_djf = np.nan*np.zeros((nev_sst,lats.size,lons.size))

                    for i in range(len(masksst[0])):
                        for j in range(nev_sst):
                            evec_sst2d_mam[j,masksst[0][i],masksst[1][i]] = evec_sst_mam[i,j]
                            evec_sst2d_jja[j,masksst[0][i],masksst[1][i]] = evec_sst_jja[i,j]
                            evec_sst2d_son[j,masksst[0][i],masksst[1][i]] = evec_sst_son[i,j]
                            evec_sst2d_djf[j,masksst[0][i],masksst[1][i]] = evec_sst_djf[i,j]

                    var_sst_all = [var_sst_mam,var_sst_jja,var_sst_son,var_sst_djf]        
                    evec_sst2d_all = [evec_sst2d_mam,evec_sst2d_jja,evec_sst2d_son,evec_sst2d_djf]
                    R2_sst = np.zeros((4,num_sce,latp.size,lonp.size))

                    for i in range(num_sce):
                        for l,pr_ind in enumerate(range(0,4)):
                            print(model,i,l,flush=True)
                            sst_ind = pr_ind - 1
                            
                            if sst_ind == -1:   # Winter SST predicts spring PPT
                                dt = 1
                            else:
                                dt = 0
                            ppt_anom_sea = ppt_anom_all[pr_ind][dt:num_yr2,:,:]
                            PCs_sst_sea = PCs_sst_all[sst_ind][0:num_yr2-dt,ind[i]]

                            # Regression for SST & CI
                            R2_sst_sea,preds_sst_sea = MLR_CV(ppt_anom_sea,PCs_sst_sea,latp,lonp,maskppt_land)
                            R2_sst[l,i,:,:] = R2_sst_sea
                    
                    if yr==0:
                        dirout = resdir+'/historical/'+model+'/'+variant
                    else:
                        dirout = resdir+'/'+exp+'/'+model+'/'+variant
                    os.makedirs(dirout, exist_ok=True)
                    fout = dirout+'/'+model+'_'+exp+'_'+variant+'.nc'
                                        
                    data_PC = [('PCs_sst_mam', PCs_sst_mam),('PCs_sst_jja', PCs_sst_jja),('PCs_sst_son', PCs_sst_son),('PCs_sst_djf', PCs_sst_djf)]
                    data_evalue = [('eval_sst_mam',eval_sst_mam),('eval_sst_jja',eval_sst_jja),('eval_sst_son',eval_sst_son),('eval_sst_djf',eval_sst_djf),
                                ('var_sst_mam',var_sst_mam),('var_sst_jja',var_sst_jja),('var_sst_son',var_sst_son),('var_sst_djf',var_sst_djf)]
                    data_ocean = [('evec_sst2d_mam', evec_sst2d_mam),('evec_sst2d_jja', evec_sst2d_jja),('evec_sst2d_son', evec_sst2d_son),('evec_sst2d_djf', evec_sst2d_djf)]
                    data_land = [('R2_sst', R2_sst),('maskppt_land',maskppt_land),('ppt_anom_mam', ppt_anom_mam2),('ppt_anom_jja', ppt_anom_jja2),('ppt_anom_son', ppt_anom_mam2),('ppt_anom_djf', ppt_anom_djf2)]
                    WriteNetCDF_Maps(fout, model, lats, lons, latp, lonp, nev_sst, num_yr2, num_sce,
                                        data_PC, data_evalue, data_ocean, data_land)
            except:
                print(model + ' Error!!!', flush=True)    