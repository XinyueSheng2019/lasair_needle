
import numpy as np
import requests
from astropy import coordinates as coords
from astropy import units as u
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore") 

'''
sherlock/tde host: 0.0014
guess host: 0.0083

'''
def PS1catalog_host(_id, _ra, _dec, radius = 0.00139, save_path = None):

  # if os.path.exists(save_path+'/'+str(_id)+'.csv'):
  #   return 0
  if _ra is None or _dec is None:
    return 0

  try:
    queryurl = 'https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/stack.json?'
    queryurl += 'ra='+str(_ra)
    queryurl += '&dec='+str(_dec)
    queryurl += '&radius='+str(radius) #0.003, 10 arcsec
    queryurl += '&columns=[raStack,decStack,gPSFMag,gPSFMagErr,rPSFMag,rPSFMagErr,iPSFMag,iPSFMagErr,zPSFMag,zPSFMagErr,yPSFMag,yPSFMagErr, gApMag,gApMagErr,rApMag,rApMagErr,iApMag,iApMagErr,zApMag,zApMagErr,yApMag,yApMagErr,yKronMag]'
    queryurl += '&nDetections.gte=6&pagesize=10000' # APMag
    print(queryurl)
    # print('\nQuerying PS1 for reference stars via MAST...\n')
    query = requests.get(queryurl, timeout=5)
    results = query.json()
  except:
    return 1

  if len(results['data']) >= 1:
    data = np.array(results['data'])

    # Get rid of very faint stars AND stars likely to saturate
    # for i in np.arange(10):
    #   data = data[data[:,2+i*2] < magmin]
    #   data = data[data[:,2+i*2] > magmax]

    # Star-galaxy separation: star if PSFmag - KronMag < 0.1
    # data = data[:,:-1][data[:,10]-data[:,-1] >= 0.1]
    data = data[:,:-1]
    
    # remove unvalid coordinates
    data = data[(data[:,0]> -999) & (data[:,1]>-999)]
    # Below is a bit of a hack to remove duplicates

    if len(data) < 1:
      print('no data after filtering out -999 and failing star-galaxy separation.')
      return 1

    # if there is no valid mag in a row, remove it
    # data = data[((data[:,2]>-999) | (data[:,3]>-999) | (data[:,4]>-999) | (data[:,5]>-999) 
    #   | (data[:,6]>-999) | (data[:,7]>-999) | (data[:,8]>-999) | (data[:,9]>-999) 
    #   | (data[:,10]>-999) | (data[:,11]>-999))]

    # data = data[~(data[:,2:11]<=-999)]

    data[data == -999] = np.nan
    
    catalog = coords.SkyCoord(ra=data[:,0]*u.degree, dec=data[:,1]*u.degree)
    data2 = []
    indices = np.arange(len(data))
    used = []
    for i in data:
      source = coords.SkyCoord(ra=i[0]*u.degree, dec=i[1]*u.deg)
      d2d = source.separation(catalog)
      catalogmsk = d2d < 2.5*u.arcsec
      indexmatch = indices[catalogmsk]
      for j in indexmatch:
        if j not in used:
          data2.append(data[j])
          for k in indexmatch:
            used.append(k)

    # print(data2)
    if len(data2)>=1:
      # add g-r and r-i columns
      data2 = np.array(data2)
      wdata = pd.DataFrame(data2, columns = ['ra', 'dec', 'gPSF', 'gPSFerr', 'rPSF', 'rPSFerr', 'iPSF', 'iPSFerr', 'zPSF', 'zPSFerr', 'yPSF', 'yPSFerr', 'gAp', 'gAperr', 'rAp', 'rAperr', 'iAp', 'iAperr', 'zAp', 'zAperr', 'yAp', 'yAperr'])

      wdata['g-r_PSF'] = wdata['gPSF'] - wdata['rPSF']
      wdata['r-i_PSF'] = wdata['rPSF'] - wdata['iPSF']
      wdata['g-r_PSFerr'] = np.sqrt(wdata['gPSFerr']**2 + wdata['rPSFerr']**2)
      wdata['r-i_PSFerr'] = np.sqrt(wdata['rPSFerr']**2 + wdata['iPSFerr']**2)
      wdata['g-r_Ap'] = wdata['gAp'] - wdata['rAp']
      wdata['r-i_Ap'] = wdata['rAp'] - wdata['iAp']
      wdata['g-r_Aperr'] = np.sqrt(wdata['gAperr']**2 + wdata['rAperr']**2)
      wdata['r-i_Aperr'] = np.sqrt(wdata['rAperr']**2 + wdata['iAperr']**2)

      # wdata['separationArcsec'] = np.repeat(offset, len(data2))
      # wdata['distance'] = np.repeat(distance, data2.shape[0])
      wdata.to_csv(save_path+'/'+str(_id)+'.csv')
      # np.savetxt(save_path+'/'+str(_id)+'_PS1'+'.csv',data2,fmt='%.8f\t%.8f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f', header='ra\tdec\tg\tgerr\tr\trerr\ti\tierr\tz\tzerr\ty\tyerr',comments='')
      print('Success! File created: ' + str(_id)+'.csv')
      return 1
    else:
      print('Field not good results')
      return 1 
  else:
    # sys.exit('Field not in PS1! Exiting') 
    print('Field not in PS1! Exiting',_id, ' ',  _ra, _dec )
    return 1



def get_host_from_magfile(ZTF_path, mag_path):
    obj_mag_path = mag_path + '/' + ZTF_path
    f = open(obj_mag_path)
    obj_mag = json.load(f)
    f.close()
    if obj_mag is not None:
      return obj_mag['sherlock']['raDeg'], obj_mag['sherlock']['decDeg']
    else:
      return None, None



