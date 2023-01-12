import os, glob, sys
import numpy as np
import numpy.ma as ma
import pandas as pd
import geopandas as gpd
from scipy.sparse.linalg import eigs
from scipy import stats
from netCDF4 import Dataset,num2date
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cft
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pycpt.load import gmtColormap
import regionmask
import warnings
warnings.filterwarnings("ignore")
from scipy import signal
from sklearn.preprocessing import scale 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.model_selection import KFold


def DomainDecompose(comm,rank,size,srclist):   
    if rank == 0:
        print('Total number of models: %d' % len(srclist))
        numpairs = np.shape(srclist)[0]
        counts = np.arange(size,dtype=np.int32)
        displs = np.arange(size,dtype=np.int32)
        ave = int(numpairs / size)
        extra = numpairs % size
        offset = 0

        for i in range(0,size):
            col = ave if i<size-extra else ave+1
            counts[i] = col

            if i==0:
                col0 = col
                offset += col
                displs[i] = 0
            else:
                comm.send(offset, dest=i)
                comm.send(col, dest=i)
                offset += col
                displs[i] = displs[i-1] + counts[i-1]

            for j in range(offset-col,offset):
                print('Rank: %d - %s' % (i, srclist[j]))
                sys.stdout.flush()
                
        offset = 0
        col = col0

    comm.Barrier()

    if rank != 0: # workers
        offset = comm.recv(source=0)
        col = comm.recv(source=0)

    comm.Barrier()
    model_files = srclist[offset:offset+col]
    return model_files, col


def SortVariant(input_models):
    variant_all = []
    for inp in input_models:
        variant = inp.split('/')[-1]
        variant = variant[:-6]
        variant = variant[1:]
        variant_all.append(variant)
    df = pd.DataFrame(input_models, columns=["inputs"])
    df['variant'] = variant_all
    df.sort_values(by=['variant'], inplace=True)
    mod_lst = df['inputs'].values
    return mod_lst


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def LoadNetCDF(file_in,varname,year_s,year_e,sf):
    nc_fid = Dataset(file_in, 'r') 
    data = nc_fid.variables[varname][:]*sf
    lat = nc_fid.variables['lat'][:]
    lon = nc_fid.variables['lon'][:]   
    time = nc_fid.variables['time']
    units = time.units
    try:
        calendar = time.calendar        
        time_convert = num2date(time[:], units, calendar=calendar)
    except:
        time_convert = num2date(time[:], units)
        
    nptimes = time_convert.astype('datetime64[ns]')
    datetime = pd.to_datetime(nptimes)
    day = np.array(datetime.day)
    month = np.array(datetime.month)
    year = np.array(datetime.year)
    ind = np.where( (year>=year_s) & (year<=year_e) )[0]
    
    day = day[ind]
    month = month[ind]
    year = year[ind]
    data = data[ind,:,:]
    return data,lat,lon,day,month,year


def Load_ClimateIndices(cifile,year_s,year_e):
    for i,fname in enumerate(cifile):
        fn = os.path.splitext(os.path.basename(fname))[0]
        df = pd.read_csv(fname)
        df = df[(df.Years>=year_s) & (df.Years<=year_e)]
        df.reset_index(inplace=True)        
        if i==0:
            df_CIs = df
        else:
            df_CIs[fn] = df[fn]
    df_CIs.drop(columns=['index','Years', 'Months'],inplace=True)   
    return df_CIs


def GetAnomaly(data, num_yr, linear=True):
    T,M,N = data.shape
    data_anom2 = np.nan * np.zeros((T,M,N))
    data_mean = np.mean(data,axis=0) 
    data_std = np.std(data,axis=0)
    mask = np.isnan(data_mean).astype(int)
    data_anom = data - np.tile(data_mean,(num_yr-1,1,1))
    for j in range(M):
        for i in range(N):
            if mask[j,i]==0:
                x = data_anom[:,j,i].copy()
                if linear:
                    data_anom2[:,j,i] = signal.detrend(x)
                else:
                    df = pd.DataFrame(x,columns=['sst'])
                    dfa = df['sst'].rolling(window=10,center=True,min_periods=1).mean()
                    data_anom2[:,j,i] = x  - dfa.values
    return data_anom2


def ExtractSeasonal(data,day,month,year,opt='mean',masked=True,linear=False):
    T,M,N = data.shape
    num_yr = int(T/12)
    data_mam = np.zeros((num_yr-1,M,N))
    data_jja = np.zeros((num_yr-1,M,N))
    data_son = np.zeros((num_yr-1,M,N))
    data_djf = np.zeros((num_yr-1,M,N))
    for yr in range(num_yr-1):
        if opt=='mean':
            data_mam[yr,:,:] = np.mean(data[yr*12+2:yr*12+5,:,:],axis=0)
            data_jja[yr,:,:] = np.mean(data[yr*12+5:yr*12+8,:,:],axis=0)
            data_son[yr,:,:] = np.mean(data[yr*12+8:yr*12+11,:,:],axis=0)
            data_djf[yr,:,:] = np.mean(data[yr*12+11:yr*12+14,:,:],axis=0)
        else:
            data_mam[yr,:,:] = np.sum(data[yr*12+2:yr*12+5,:,:],axis=0)
            data_jja[yr,:,:] = np.sum(data[yr*12+5:yr*12+8,:,:],axis=0)
            data_son[yr,:,:] = np.sum(data[yr*12+8:yr*12+11,:,:],axis=0)
            data_djf[yr,:,:] = np.sum(data[yr*12+11:yr*12+14,:,:],axis=0)

    #Get Anomaly and Detrend
    data_anom_mam = GetAnomaly(data_mam, num_yr, linear=linear)
    data_anom_jja = GetAnomaly(data_jja, num_yr, linear=linear)
    data_anom_son = GetAnomaly(data_son, num_yr, linear=linear)
    data_anom_djf = GetAnomaly(data_djf, num_yr, linear=linear)

    if masked==True:
        data_mean = np.mean(data,axis=0)
        mask = data_mean.mask

        data_mam = ma.masked_array(data_mam, mask=np.tile(mask,(num_yr-1,1,1)))
        data_jja = ma.masked_array(data_jja, mask=np.tile(mask,(num_yr-1,1,1)))
        data_son = ma.masked_array(data_son, mask=np.tile(mask,(num_yr-1,1,1)))
        data_djf = ma.masked_array(data_djf, mask=np.tile(mask,(num_yr-1,1,1)))

        data_anom_mam = ma.masked_array(data_anom_mam, mask=np.tile(mask,(num_yr-1,1,1)))
        data_anom_jja = ma.masked_array(data_anom_jja, mask=np.tile(mask,(num_yr-1,1,1)))
        data_anom_son = ma.masked_array(data_anom_son, mask=np.tile(mask,(num_yr-1,1,1)))
        data_anom_djf = ma.masked_array(data_anom_djf, mask=np.tile(mask,(num_yr-1,1,1)))
        
    return data_mam, data_jja, data_son, data_djf, data_anom_mam, data_anom_jja, data_anom_son, data_anom_djf


def ExtractPredictor(data,day,month,year,mon_s,mon_e,opt='mean',masked=True,std=False):
    T,M,N = data.shape
    num_yr = int(T/12)
    data_out = np.zeros((num_yr-1,M,N))
    for yr in range(num_yr-1):
        if opt=='mean':
            data_out[yr,:,:] = np.mean(data[yr*12+mon_s:yr*12+mon_e,:,:],axis=0)
        else:
            data_out[yr,:,:] = np.sum(data[yr*12+mon_s:yr*12+mon_e,:,:],axis=0)

    #Get Anomaly and Detrend
    data_anom_out = GetAnomaly(data_out, num_yr, std=std)
            
    if masked==True:
        data_mean = np.mean(data,axis=0)
        mask = data_mean.mask        
        data_out = ma.masked_array(data_out, mask=np.tile(mask,(num_yr-1,1,1)))
        data_anom_out = ma.masked_array(data_anom_out, mask=np.tile(mask,(num_yr-1,1,1)))
    
    return data_out, data_anom_out

           
def PCA_kernel(data,nev,masked):
    T,M,N = data.shape
    if masked:
        data_mean = np.mean(data,axis=0)
        maskind=np.where(~data_mean.mask)
        data_nomask = data[:,maskind[0],maskind[1]]
    else:
        data_nomask = np.reshape(data,(T,M*N))
        maskind = None

    num_pts = data_nomask.shape[1]
    C = np.cov(data_nomask.T)
    evalue, evector = eigs(C,nev)
    evalue = evalue.astype('double'); evector = evector.astype('double')        
    for i in range(nev):
        maxind = np.argmax(np.abs(evector[:,i]))
        sign = np.sign(evector[maxind,i])
        evector[:,i] = evector[:,i]*sign
    
    PCs = np.dot(data_nomask,evector)#*np.sqrt(evalue)/num_pts
    trace = np.trace(C)
    return evector,evalue,PCs,trace,maskind           
         

def MLR_CV(ppt_in,PCs_in,lat,lon,mask):
    M = lat.size
    N = lon.size
    cv = 5
    T,_,_ = ppt_in.shape    
    r2 = np.nan * np.zeros((M,N))
    y_pred = np.nan * np.zeros((T,M,N))
    X = PCs_in
    ndims = PCs_in.ndim
    if ndims==1:
        X = X.reshape(-1, 1)    
        
    for j in range(M):
        for i in range(N):
            if mask[j,i]==1:       
                model = linear_model.LinearRegression()
                y = ppt_in[:,j,i]
                preds = cross_val_predict(model, X, y, cv=cv)                
                r2[j,i] = r2_score(y, preds)
                y_pred[:,j,i] = preds
    return r2,y_pred


def WriteNetCDF_Maps(fname, description, latsst, lonsst, latpr, lonpr,
                     nevs, num_yrs, num_sces,
                     data_PC, data_evalue, data_ocean, data_land):
    """
    This function saves numpy data into NetCDF format.
    """
    f = Dataset(fname, 'w', format='NETCDF4')
    f.description = description
        
    """ Lat & Lon info """
    # Latitude
    f.createDimension('lats',latsst.size)
    lats = f.createVariable('lats', np.float32, ('lats',))
    lats.units = 'degrees_north'
    lats.long_name = 'latitude'
    lats.axis = 'Y'
    lats[:] = latsst

    f.createDimension('latp',latpr.size)
    latp = f.createVariable('latp', np.float32, ('latp',))
    latp.units = 'degrees_north'
    latp.long_name = 'latitude'
    latp.axis = 'Y'
    latp[:] = latpr
    
    # Longitude
    f.createDimension('lons',lonsst.size)
    lons = f.createVariable('lons', np.float32, ('lons',))
    lons.units = 'degrees_east'
    lons.long_name = 'longitude'
    lons.axis = 'X'
    lons[:] = lonsst

    f.createDimension('lonp',lonpr.size)
    lonp = f.createVariable('lonp', np.float32, ('lonp',))
    lonp.units = 'degrees_east'
    lonp.long_name = 'longitude'
    lonp.axis = 'X'
    lonp[:] = lonpr

    # Number of PC
    f.createDimension('nev',nevs)
    nev = f.createVariable('nev', np.float32, ('nev',))
    nev.units = '-'
    nev.long_name = 'numPC'
    nev.axis = '-'
    nev[:] = np.linspace(1,nevs,nevs)

    # Number of year
    f.createDimension('num_yr',num_yrs)
    num_yr = f.createVariable('num_yr', np.float32, ('num_yr'))
    num_yr.units = '-'
    num_yr.long_name = 'numYear'
    num_yr.axis = '-'
    num_yr[:] = np.linspace(1,num_yrs,num_yrs)

    # Number of seasons
    f.createDimension('num_sea',4)
    num_sea = f.createVariable('num_sea', np.float32, ('num_sea'))
    num_sea.units = '-'
    num_sea.long_name = 'numSeasons'
    num_sea.axis = '-'
    num_sea[:] = np.linspace(1,4,4)
    
    # Number of scenarios
    f.createDimension('num_sce',num_sces)
    num_sce = f.createVariable('num_sce', np.float32, ('num_sce'))
    num_sce.units = '-'
    num_sce.long_name = 'numSce'
    num_sce.axis = '-'
    num_sce[:] = np.linspace(1,num_sces,num_sces)
     
    try:
        for i,(name,data) in enumerate(data_evalue):
            var = f.createVariable(name, np.float64, ('nev'))
            var[:] = data
    except:
        print("No data_evalue")
            
    try:
        for i,(name,data) in enumerate(data_PC):
            var = f.createVariable(name, np.float64, ('num_yr','nev'))
            var[:] = data
    except:
        print("No data_PC")
        
    for i,(name,data) in enumerate(data_ocean):
        if data.ndim==2:
            var = f.createVariable(name, np.float64, ('lats','lons'))
        else:
            T,_,_ = data.shape
            if T==nevs:
                var = f.createVariable(name, np.float64, ('nev','lats','lons'))
            elif T==num_yrs:
                var = f.createVariable(name, np.float64, ('num_yr','lats','lons'))
        var[:] = data

    for i,(name,data) in enumerate(data_land):    
        if data.ndim==2:
            var = f.createVariable(name, np.float64, ('latp','lonp'))
        elif data.ndim==3:
            T,_,_ = data.shape
            if T==num_yrs:                
                var = f.createVariable(name, np.float64, ('num_yr','latp','lonp'))
        elif data.ndim==4:
            var = f.createVariable(name, np.float64, ('num_sea','num_sce','latp','lonp'))
            
        var[:] = data
    f.close()