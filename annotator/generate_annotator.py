import json, sys, settings
import lasair, os
from astropy.io import fits
import numpy as np
sys.path.append("../source") 
import build_dataset
import ztf_image_pipeline 
from host_meta_pipeline import PS1catalog_host
from preprocessing import single_transient_preprocessing
from tensorflow.keras import models

NEEDLE_PATH = '../lasair_20240105/'
LABEL_PATH = NEEDLE_PATH + '/label_dict_equal_test.json'
BCLASSIFIER_PATH = '../bogus_model_without_zscale'
NEEDLE_OBJ_PATH = 'needle_objects'
LABEL_LIST = ['SN', 'SLSN-I', 'TDE']

BClassifier = models.load_model(BCLASSIFIER_PATH)


def get_obj_meta(candidates, candi_idx, disc_mjd, disc_mag, host_mag):
    
    candi_mag = candidates[candi_idx]['magpsf']
    delta_t_discovery = round(candidates[candi_idx]['jd'] - disc_mjd - 2400000.5, 5)
    delta_mag_discovery = round(candi_mag - disc_mag, 5)
    if candi_idx < len(candidates) - 1: 
        delta_t_recent = round(candidates[candi_idx]['jd'] - candidates[candi_idx + 1]['jd'], 5)
        delta_mag_recent = round(candi_mag - candidates[candi_idx + 1]['magpsf'], 5)
    else:
        delta_t_recent = delta_t_discovery
        delta_mag_recent = delta_mag_discovery
    delta_host_mag = round(candi_mag - host_mag, 5)
    ratio_recent, ratio_disc = build_dataset.get_ratio(delta_mag_recent, delta_t_recent, delta_mag_discovery, delta_t_discovery)

    return [candi_mag, disc_mag, delta_mag_discovery, delta_t_discovery, ratio_recent, ratio_disc, delta_host_mag] 
   

def get_obj_image(sci_filename, ref_filename, diff_filename, BClassifier):
    '''
    get images from one observation.
    '''
    sci_data = build_dataset.get_shaped_image_simple(fits.getdata(sci_filename))
    
    ref_data = build_dataset.get_shaped_image_simple(fits.getdata(ref_filename))
    
    if diff_filename is None:
        if build_dataset.check_shape(sci_data) and build_dataset.check_shape(ref_data):
            sci_data = build_dataset.img_reshape(build_dataset.image_normal(build_dataset.zscale(sci_data))) 
            ref_data = build_dataset.img_reshape(build_dataset.image_normal(build_dataset.zscale(ref_data)))
            if build_dataset.check_bogus(BClassifier, sci_data) and build_dataset.check_bogus(BClassifier, ref_data):  
                comb_data = np.concatenate((sci_data, ref_data), axis = -1)
                return comb_data
            else:
                return None
        else:
            return None
    else:
        diff_data = build_dataset.get_shaped_image_simple(fits.getdata(diff_filename))
        if build_dataset.check_shape(sci_data) and build_dataset.check_shape(diff_data) and build_dataset.check_shape(ref_data):
            sci_data = build_dataset.img_reshape(build_dataset.image_normal(build_dataset.zscale(sci_data))) 
            ref_data = build_dataset.img_reshape(build_dataset.image_normal(build_dataset.zscale(ref_data)))
            diff_data = build_dataset.img_reshape(build_dataset.image_normal(build_dataset.zscale(diff_data)))
            if build_dataset.check_bogus(BClassifier, sci_data) and build_dataset.check_bogus(BClassifier, ref_data) and build_dataset.check_bogus(BClassifier, diff_data):
                comb_data = np.concatenate((sci_data, ref_data), axis = -1)
                comb_data = np.concatenate((comb_data, diff_data), axis = -1)
                return comb_data
            else:
                return None
        else:
            return None


def remove_AGN(candidates):
    # is the first detection longer than 60 days?
    if candidates[0]['mjd'] - candidates[-1]['mjd'] >= 60:
        return True
    else:
        return False

def scaling_meta(meta_data, scaling_file_path):
    '''
    assume normaliztion method 1 in this case.
    '''
    f = open(scaling_file_path+'/scaling_data.json')
    scaling = json.load(f)
    f.close()
    mt_mean = np.array(scaling['mean'])
    mt_std = np.array(scaling['std'])
    meta_data = (meta_data - mt_mean)/mt_std
    return meta_data

def collect_data_from_lasair(objectId, objectInfo, band = 'r'):
    if band == 'g':
        fid = 1
    elif band == 'r':
        fid = 2

    candidates = objectInfo['candidates']
    # print(candidates)
    
    candidates = [x for x in candidates if 'image_urls' in x.keys() and x['fid'] == fid]
    
    
    disdate = objectInfo['objectData']['discMjd']
    discFilter = objectInfo['objectData']['discFilter']

    # print(objectInfo)


    if len(candidates) > 1:
        mags = np.array([m['magpsf']for m in candidates])
        idx = np.argmin(mags)

        if discFilter == band:
            discMag = objectInfo['objectData']['discMag']
            discMag = float(discMag.strip(r'\u')[0])
        else:
            discMag = candidates[-1]['magpsf']

        flag = False
        
        while flag == False and idx < len(candidates)-1 and idx >= 0:
            peak_urls = candidates[idx]['image_urls']
            science_url = peak_urls['Science']
            template_url = peak_urls['Template']
            # difference_url = peak_urls['Difference']

            obsjd_path = ztf_image_pipeline.create_path(NEEDLE_OBJ_PATH, objectId)
            sci_fname = "sci_ztf_peak.fits"
            sci_filename = obsjd_path + '/' + sci_fname
            ref_fname = 'ref_ztf_peak.fits'
            ref_filename = obsjd_path + '/' + ref_fname

            if not os.path.exists(sci_filename):
                os.system("curl -o %s %s" % (sci_filename, science_url))
                os.system("curl -o %s %s" % (ref_filename, template_url))
            
            if os.path.getsize(sci_filename) < 800 or os.path.getsize(ref_filename) < 800:
                flag = False
                idx += 1
            else:
                flag = True
                
        if flag:

            img_data = get_obj_image(sci_filename, ref_filename, None, BClassifier)

            if img_data is not None:    
                host_ra, host_dec = objectInfo['sherlock']['raDeg'], objectInfo['sherlock']['decDeg']
                PS1catalog_host(_id = objectId, _ra = host_ra, _dec = host_dec, save_path=NEEDLE_OBJ_PATH + '/hosts')
                host_meta = build_dataset.add_host_meta(objectId, host_path = NEEDLE_OBJ_PATH + '/hosts', only_complete = True)
                sherlock_meta = [objectInfo['sherlock']['separationArcsec']]
                if host_meta is None:
                    print('Host meta not found.')
                    return None, None
                else:

                    meta_data = get_obj_meta(candidates, idx, disdate, discMag, host_meta[1]) + host_meta + sherlock_meta

                    meta_data = scaling_meta(meta_data, NEEDLE_PATH)

                    return img_data, meta_data
        else:
            print('object %s images no found' % objectId)
            return None, None

    else:
        print('candidates for %s not found.\n' % objectId)
        return None, None
    
    # img_data, meta_data = single_transient_preprocessing(img_data, meta_data)
    


def needle_prediction(img_data, meta_data):
    
    img_data, meta_data = single_transient_preprocessing(img_data, meta_data)
    # print(img_data.shape, meta_data.shape)

    # average 10 models
    emsemble_results = []
    for i in np.arange(10):
        TSClassifier = models.load_model(NEEDLE_PATH + 'seed_456_model_128_3_64_3_nm1_lasair_' + str(i))
        results = TSClassifier.predict({'image_input': img_data, 'meta_input': meta_data})
        emsemble_results.append(results)
    emsemble_results = np.array(emsemble_results)
    return np.mean(emsemble_results, axis = 0)


# This function deals with an object once it is received from Lasair
def handle_object(objectId, L, topic_out, threhold = 0.75):
    # from the objectId, we can get all the info that Lasair has
    objectInfo = L.objects([objectId])[0]
    if not objectInfo:
        return 0
    
    if remove_AGN(objectInfo['candidates']):
        return 0

    img_data, meta_data = collect_data_from_lasair(objectId, objectInfo, band = 'r')

    if img_data is None or meta_data is None:
        print('object %s failed to be annocated.' % objectId)
        return 0
    else:
        results = needle_prediction(img_data, meta_data)
    
        # print(results)
        
        classdict      = {'SN': str(results[0][0]), 'SLSN-I': str(results[0][1]), 'TDE': str(results[0][2])} 
        if np.max(results[0]) >= threhold:
            classification = LABEL_LIST[np.argmax(results[0])]
        else:
            classification = 'unclear'
        explanation    = 'lasiar-NEEDLE-TH prediction.'

        # now we annotate the Lasair data with the classification
        L.annotate(
            topic_out, 
            objectId, 
            classification,
            version='20240110', 
            explanation=explanation, 
            classdict=classdict, 
            url='')
        print(objectId, '-- annotated!')
        return 1
    # get all images

    # print(objectInfo.keys())
    # print(objectInfo['objectId'])
    
    # objectInfo.keys():
    #  -- objectData: about the object and its features
    #  -- candidates: the lightcurve of detections and nondetections
    #  -- sherlock: the sherlock information
    #  -- TNS: any crossmatch with the TNS database

    # analyse object here. The following is a toy annotation
    # use NEEDLE to predict the class

    # STEP 1: add raw image and metadata
    # STEP 2: quality test - preprocessing
    # STEP 3: feed into NEEDLE and get results
    

        
    return 1

#####################################
# first we set up pulling the stream from Lasair
# a fresh group_id gets all, an old group_id starts where it left off
group_id = settings.GROUP_ID

# a filter from Lasair, example 'lasair_2SN-likecandidates'
topic_in = settings.TOPIC_IN

# kafka consumer that we can suck from
consumer = lasair.lasair_consumer('kafka.lsst.ac.uk:9092', group_id, topic_in)

# the lasair client will be used for pulling all the info about the object
# and for annotating it
L = lasair.lasair_client(settings.API_TOKEN)

# TOPIC_OUT is an annotator owned by a user. API_TOKEN must be that users token.
topic_out = settings.TOPIC_OUT

# just get a few to start
max_alert = 500

n_alert = n_annotate = 0
while n_alert < max_alert:
    msg = consumer.poll(timeout=20)
    if msg is None:
        break
    if msg.error():
        print(str(msg.error()))
        break
    jsonmsg = json.loads(msg.value())
    objectId       = jsonmsg['objectId']

    n_alert += 1
    n_annotate += handle_object(objectId, L, topic_out)

print('Annotated %d of %d objects' % (n_annotate, n_alert))

