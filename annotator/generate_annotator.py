import json, sys, settings
import lasair, os
from astropy.io import fits
from astropy.time import Time
import numpy as np
sys.path.append("../source") 
import build_dataset
import ztf_image_pipeline 
from host_meta_pipeline import PS1catalog_host
from preprocessing import single_transient_preprocessing, feature_reduction_for_mixed_band, feature_reduction_for_mixed_band_no_host, apply_data_scaling
from tensorflow.keras import models
from datetime import datetime
import json
import shutil


NEEDLE_PATH_TH_R   = '../lasair_th_r/'
NEEDLE_PATH_TH_MIX = '../lasair_th_mixed/'
NEEDLE_PATH_T_R    = '../lasair_t_r/'
NEEDLE_PATH_T_MIX  = '../lasair_t_mixed/'
RECORD_PATH = 'records'

LABEL_PATH = 'label_dict_equal_test.json'
BCLASSIFIER_PATH = '../quality_model'
NEEDLE_OBJ_PATH = 'needle_objects'
LABEL_LIST = ['SN', 'SLSN-I', 'TDE']

# add directories 
if os.path.exists(NEEDLE_OBJ_PATH) is False:
    os.makedirs(NEEDLE_OBJ_PATH)

if os.path.exists(NEEDLE_OBJ_PATH + '/hosts') is False:
    os.makedirs(NEEDLE_OBJ_PATH+ '/hosts')

if os.path.exists('logs') is False:
    os.makedirs('logs')

if os.path.exists(RECORD_PATH) is False:
    os.makedirs(RECORD_PATH)

# set a log file for recording the objects
log = open('logs/log_' + datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.txt', 'w')

BClassifier = models.load_model(BCLASSIFIER_PATH)

def remove_AGN(candidates):
    # is the first detection longer than 60 days?
    if candidates[0]['mjd'] - candidates[-1]['mjd'] >= 60:
        return True
    else:
        return False

def find_earliest_discovery_mjd(objectInfo):
    # some new objects have forced photometry, which the discovery dates could be earlier.
    # this functions check the earliest date that the event is rising from the forced information.
    # for g and r band, find the lastest detection date that the diff > 0, and choose the earliest among two dates as the final pre-discovery date
    def find_earliest_disc_each_band(forced_info, fid, unforced_disdate):
        b_info = np.array([(x['mjd'], x['forcediffimflux']) for x in forced_info if x['fid'] == fid and x['ranr'] > -99999.0 and x['mjd'] <= unforced_disdate], 
                        dtype=[('mjd', 'f8'), ('forcediffimflux', 'f8')])
        
        if b_info.shape[0] != 0:
            b_info_sort = np.sort(b_info, order='mjd')
            pre_disc_mjd = b_info_sort[0][0]
            n = 0
            while n < b_info_sort.shape[0] - 1:
                if b_info_sort[n][1] < 0 and b_info_sort[n+1][1] > 0:
                    pre_disc_mjd =  b_info_sort[n+1][0]
                n += 1
            return pre_disc_mjd
        else:
            return unforced_disdate

    unforced_disdate = objectInfo['objectData']['discMjd']
  
    if 'forcedphot' in objectInfo.keys():
        forced_info = objectInfo['forcedphot']
        max_mjd_1 = find_earliest_disc_each_band(forced_info, 1, unforced_disdate)
        max_mjd_2 = find_earliest_disc_each_band(forced_info, 2, unforced_disdate)
        print(f'g band pre_disc is {max_mjd_1}, r band pre_disc is {max_mjd_2}.')
        return max_mjd_1 if max_mjd_1 <= max_mjd_2 else max_mjd_2 
    else:
        return unforced_disdate


def get_obj_meta(candidates, candi_idx, disc_mjd, host_mag, for_mixed = False):

    candi_mag = candidates[candi_idx]['magpsf']
    disc_band_mag = candidates[-1]['magpsf']
    delta_t_discovery_band = round(candidates[candi_idx]['mjd'] - candidates[-1]['mjd'], 5)
    delta_t_discovery = round(candidates[candi_idx]['mjd'] - disc_mjd, 5)
    delta_mag_discovery = round(candi_mag - disc_band_mag, 5)

    if candi_idx < len(candidates) - 1: 
        delta_t_recent = round(candidates[candi_idx]['mjd'] - candidates[candi_idx + 1]['mjd'], 5)
        delta_mag_recent = round(candi_mag - candidates[candi_idx + 1]['magpsf'], 5)
    else:
        delta_t_recent = delta_t_discovery
        delta_mag_recent = delta_mag_discovery

    ratio_recent, ratio_disc = build_dataset.get_ratio(delta_mag_recent, delta_t_recent, delta_mag_discovery, delta_t_discovery_band)
    if for_mixed:
        row = [candi_mag, disc_band_mag, delta_mag_discovery, delta_t_discovery_band, delta_t_discovery, ratio_recent, ratio_disc]
    else: # for meta_r
        row = [candi_mag, disc_band_mag, delta_mag_discovery, delta_t_discovery, ratio_recent, ratio_disc]

    if host_mag is not None:
        delta_host_mag = round(candi_mag - host_mag, 5)
        row += [delta_host_mag]

    return row

   


def get_obj_image(image_urls, obsjd_path, BClassifier):
    """
    Download and process astronomical images from a list of URLs, handling cases where some images might be missing or corrupted.
    
    Parameters:
        image_urls (list of dict): List of dictionaries with 'Science' and 'Template' image URLs.
        obsjd_path (str): Path to the directory where images will be saved.
        BClassifier (object): Classifier object used for quality check of images.
        
    Returns:
        comb_data (np.ndarray or None): Concatenated and processed science and template images, or None if no valid images found.
    """
    
    sci_filename = os.path.join(obsjd_path, 'sci_peak.fits')
    ref_filename = os.path.join(obsjd_path, 'ref_peak.fits')
    comb_data = None
    i = 0
    
    while i < len(image_urls):
        print(image_urls[i])
        # Download science and reference images
        os.system(f'curl -o {sci_filename} {image_urls[i]["Science"]}')
        os.system(f'curl -o {ref_filename} {image_urls[i]["Template"]}')
        
        # Check if the downloaded files are valid
        if os.path.getsize(sci_filename) < 800 or os.path.getsize(ref_filename) < 800:
            i += 1
            continue
        
        # Load FITS data
        sci_data = build_dataset.get_shaped_image_simple(fits.getdata(sci_filename))
        ref_data = build_dataset.get_shaped_image_simple(fits.getdata(ref_filename))
        
        # Check shapes of the images
        if build_dataset.check_shape(sci_data) and build_dataset.check_shape(ref_data):
            # Normalize and reshape the images
            sci_data = build_dataset.img_reshape(build_dataset.image_normal(build_dataset.zscale(sci_data))) 
            ref_data = build_dataset.img_reshape(build_dataset.image_normal(build_dataset.zscale(ref_data)))
            
            # Check the quality of the images
            if build_dataset.check_quality(BClassifier, sci_data) and build_dataset.check_quality(BClassifier, ref_data):  
                comb_data = np.concatenate((sci_data, ref_data), axis=-1)
                break 
            else:
                i += 1
                continue
        else:
            i += 1
            continue
    
    return comb_data



def collect_data_from_lasair(objectId, objectInfo):

    def find_peak_mag_images(candids):
        if len(candids) > 0:
            info = np.array([[m["magpsf"], m["mjd"], m['image_urls']] for m in candids])
            idx = np.argmin(info[:,0])
            image_urls = []
            i = idx
            while i < info.shape[0]:
                image_urls.append(info[i][2])
                i += 1
            return float(info[idx][1]), float(info[idx][0]), image_urls, idx
        else:
            return None, None, None, None
        

    candidates = objectInfo['candidates']
    candids_g = [x for x in candidates if 'image_urls' in x.keys() and x['fid'] == 1]
    candids_r = [x for x in candidates if 'image_urls' in x.keys() and x['fid'] == 2]
    disdate = find_earliest_discovery_mjd(objectInfo)

    flag_r = False

    img_data, meta_r, meta_mixed, find_host = None, None, None, False
    
    # get the images 
    peak_mjd_r, peak_mag_r, image_urls_r, idx_r = find_peak_mag_images(candids_r)
    peak_mjd_g, peak_mag_g, image_urls_g, idx_g = find_peak_mag_images(candids_g)

    obsjd_path = os.path.join(NEEDLE_OBJ_PATH, objectId)
    if os.path.isdir(obsjd_path) == False:
        os.mkdir(obsjd_path)

    comb_data = get_obj_image(image_urls_r, obsjd_path, BClassifier)
    if comb_data is None:
        comb_data = get_obj_image(image_urls_g, obsjd_path, BClassifier)
    if comb_data is None:
        log.write('object %s images in g and r bands do not pass criteria or not found.\n' % objectId)
    else:
        img_data = comb_data

        # get host meta 
        host_ra, host_dec = objectInfo['sherlock']['raDeg'], objectInfo['sherlock']['decDeg']
        get_host = PS1catalog_host(_id = objectId, _ra = host_ra, _dec = host_dec, save_path = NEEDLE_OBJ_PATH + '/hosts') 
        if get_host:
            host_meta = build_dataset.add_host_meta(objectId, host_path = NEEDLE_OBJ_PATH + '/hosts', only_complete = True)
            if host_meta is not None:
                host_g, host_r = host_meta[0], host_meta[1]
                sherlock_meta = [objectInfo['sherlock']['separationArcsec']]
                find_host = True
            else:
                host_g, host_r = None, None
        else:
            host_g, host_r = None, None

        # get meta_mixed
        if peak_mag_r is not None and peak_mag_g is not None:
            peak_mag_g_minus_r = peak_mag_g - peak_mag_r
            peak_t_g_minus_r = peak_mjd_g - peak_mjd_r
            meta_data_g = get_obj_meta(candids_g, idx_g, disdate, host_g, True)
            meta_data_r = get_obj_meta(candids_r, idx_r, disdate, host_r, True)
            flag_r = True
            
        elif peak_mag_r is not None and peak_mag_g is None:
            peak_mag_g_minus_r, peak_t_g_minus_r = 0., 0.
            meta_data_r = get_obj_meta(candids_r, idx_r, disdate, host_r, True)
            meta_data_g = [0.] * len(meta_data_r)
            flag_r = True

        elif peak_mag_g is not None and peak_mag_r is None:
            peak_mag_g_minus_r, peak_t_g_minus_r = 0., 0.
            meta_data_g = get_obj_meta(candids_g, idx_g, disdate, host_g, True)
            meta_data_r = [0.] * len(meta_data_g)

        if find_host:
            meta_mixed = meta_data_r + meta_data_g + [peak_mag_g_minus_r, peak_t_g_minus_r] + host_meta + sherlock_meta
        else:
            meta_mixed = meta_data_r + meta_data_g + [peak_mag_g_minus_r, peak_t_g_minus_r]

        # get meta_r
        if flag_r:
            meta_data_r = get_obj_meta(candids_r, idx_r, disdate, host_r, False)
            if find_host:
                meta_r = meta_data_r + host_meta + sherlock_meta
            else:
                meta_r = meta_data_r
        
    # delete image files
    shutil.rmtree(obsjd_path)
 
    return img_data, meta_r, meta_mixed, find_host
    

def needle_th_prediction(img_data, meta_r, meta_mixed):

    # Object with a host predicted by Sherlock. 
    if meta_r is not None:
        result_r = []
        _img_data, meta_r = single_transient_preprocessing(img_data, meta_r)
        meta_r = np.nan_to_num(meta_r)
        for i in np.arange(5):
            model_r_path = NEEDLE_PATH_TH_R + 'seed_456_model_128_3_64_3_nm1_lasair_' + str(i)
            _meta_r = apply_data_scaling(meta_r, model_r_path + '/scaling_data.json')
            r_classifier = models.load_model(model_r_path)
            result_r.append(r_classifier.predict({'image_input': _img_data, 'meta_input': _meta_r}))
        result_r = np.array(result_r)
        result_r = np.mean(result_r, axis = 0)
    else:
        result_r = None

    if meta_mixed is not None:
        result_mixed = [] 
        _img_data, meta_mixed = single_transient_preprocessing(img_data, meta_mixed)
        meta_mixed =  np.nan_to_num(meta_mixed)
        meta_mixed, _ = feature_reduction_for_mixed_band(meta_mixed)

        
        for i in np.arange(5):
            model_mixed_path = NEEDLE_PATH_TH_MIX + 'seed_456_model_nor1_neurons_64_128_128_ranking_updated_lasair' + str(i)
            mixed_classifier = models.load_model(model_mixed_path) 
            _meta_mixed = apply_data_scaling(meta_mixed, model_mixed_path + '/scaling_data.json')
            result_mixed.append(mixed_classifier.predict({'image_input': _img_data, 'meta_input': _meta_mixed}))
        result_mixed = np.array(result_mixed)
        result_mixed =  np.mean(result_mixed, axis = 0)
    else:
        result_mixed = None

    return result_r, result_mixed


def needle_t_prediction(img_data, meta_r, meta_mixed):

    # Object with no archived host.
    if meta_r is not None:
        result_r = []
        _img_data, meta_r = single_transient_preprocessing(img_data, meta_r)
        meta_r = np.nan_to_num(meta_r)

        for i in np.arange(5):
            model_r_path = NEEDLE_PATH_T_R + 'seed_456_model_64_3_128_3_128_3_nm1_lasair' + str(i)
            _meta_r = apply_data_scaling(meta_r, model_r_path + '/scaling_data.json')
            r_classifier = models.load_model(model_r_path)
            result_r.append(r_classifier.predict({'image_input': _img_data, 'meta_input': _meta_r}))
        result_r = np.array(result_r)
        result_r = np.mean(result_r, axis = 0)
    else:
        result_r = None

    if meta_mixed is not None:
        result_mixed = [] 
        _img_data, meta_mixed = single_transient_preprocessing(img_data, meta_mixed)
        meta_mixed =  np.nan_to_num(meta_mixed)
        meta_mixed, _ = feature_reduction_for_mixed_band_no_host(meta_mixed)
        for i in np.arange(5):
            model_mixed_path = NEEDLE_PATH_T_MIX + 'seed_456_model_nor1_neurons_64_128_128_lasair' + str(i)
            mixed_classifier = models.load_model(model_mixed_path) 
            _meta_mixed = apply_data_scaling(meta_mixed, model_mixed_path + '/scaling_data.json')
            result_mixed.append(mixed_classifier.predict({'image_input': _img_data, 'meta_input': _meta_mixed}))
        result_mixed = np.array(result_mixed)
        result_mixed =  np.mean(result_mixed, axis = 0)
    else:
        result_mixed = None
  
    return result_r, result_mixed



def update_records(objectId, record_path, classdict, classification):
    # store records to a folder, where contains obj prediction file with JSON format
    file_path = os.path.join(record_path, f'{objectId}.json')
    current_mjd = str(round(Time.now().mjd, 3))
    flag = 0
    record_msg = ''
    
    # Check if the file exists
    if not os.path.exists(file_path):
        # Create new record if file does not exist
        if classification == 'SLSN-I' or classification == 'TDE':
            flag = 1
            record_msg = f'At MJD {current_mjd}, this object is firstly predicted as {classification}.'
        record_dict = {}
        record_dict[current_mjd] = {'classdict': classdict, 'prediction': classification, 'flag': flag}
        record_json = json.dumps(record_dict, indent=4)
        
        with open(file_path, 'w') as f:
            f.write(record_json)
    else:
        # Update existing record if file exists
        with open(file_path, 'r+') as f:
            try:
                # Load existing records
                f.seek(0)
                record_dict = json.load(f)
            except json.JSONDecodeError:
                # Handle the case where the file is empty or contains invalid JSON
                record_dict = {}
            
            first_mjd = None
            for m in record_dict.keys():
                if record_dict[m]['flag'] == 1:
                    if classification != record_dict[m]['prediction']: # always decided by the up-to-date classification
                        record_dict[m]['flag'] = 0
                    else:
                        first_mjd = m
                        record_msg = f'At MJD {first_mjd}, this object is firstly predicted as {classification}.'
            
            if classification == 'SLSN-I' or classification == 'TDE':
                if first_mjd is None:
                    record_msg = f'At MJD {current_mjd}, this object is firstly predicted as {classification}.'
                    flag = 1
            
            record_dict[current_mjd] = {'classdict': classdict, 'prediction': classification, 'flag': flag}
            record_json = json.dumps(record_dict, indent=4)
            
            # Move cursor to the beginning of the file and truncate it
            f.seek(0)
            f.write(record_json)
            f.truncate()
    
    return record_msg



def handle_object(objectId, L, topic_out, threshold = 0.70, test = False):

    # from the objectId, we can get all the info that Lasair has
    objectInfo = L.objects([objectId])[0]

    if objectInfo is None:
        log.write('object %s is removed as there is no information.\n' % objectId)
        return 0
    
    if remove_AGN(objectInfo['candidates']):
        log.write('object %s is removed as it is older than 60 days.\n' % objectId)
        return 0

    # print(objectId, objectInfo)
    img_data, meta_r, meta_mixed, findhost = collect_data_from_lasair(objectId, objectInfo)

    # img_data, meta_r, meta_mixed, findhost, findhost change type to boolean
  
    if img_data is None:
        return 0
    
    if findhost:
        result_r, result_mixed = needle_th_prediction(img_data, meta_r, meta_mixed)
    else:
        log.write('object %s host meta not found, use NEEDLE-T\n' % objectId)
        result_r, result_mixed = needle_t_prediction(img_data, meta_r, meta_mixed)

    SN_mix = float(result_mixed[0][0]) if result_mixed is not None else None
    SN_r = float(result_r[0][0]) if result_r is not None else None
    SLSN_mix = float(result_mixed[0][1]) if result_mixed is not None else None
    SLSN_r = float(result_r[0][1]) if result_r is not None else None
    TDE_mix = float(result_mixed[0][2]) if result_mixed is not None else None
    TDE_r = float(result_r[0][2]) if result_r is not None else None

    detailed_classdict      = {'SN_mix': SN_mix, 'SN_r': SN_r, 'SLSN-I_mix': SLSN_mix, 'SLSN-I_r': SLSN_r, 'TDE_mix': TDE_mix, 'TDE_r': TDE_r}
    # 6/4 weighted average
    weight_score = 0.60
    if result_mixed is not None and result_r is not None:
        weighted_results = [SN_mix*weight_score + SN_r*(1-weight_score), SLSN_mix*weight_score + SLSN_r*(1-weight_score), TDE_mix*weight_score + TDE_r*(1-weight_score)] 
    elif result_mixed is None and result_r is not None:
        weighted_results = [SN_r, SLSN_r, TDE_r]
    else:
        weighted_results = [SN_mix, SLSN_mix, TDE_mix]

    classdict = {'SN': float(weighted_results[0]), 'SLSN-I': float(weighted_results[1]), 'TDE': float(weighted_results[2])}
    
    if np.max(weighted_results) >= threshold: 
        classification = LABEL_LIST[np.argmax(weighted_results)]
    else:
        classification = 'unclear'
    
    explanation    = update_records(objectId, RECORD_PATH, detailed_classdict, classification) # record/track the first time stamp
    
    if not test:
        # now we annotate the Lasair data with the classification
        L.annotate(
            topic_out, 
            objectId, 
            classification,
            version='test_20240729', 
            explanation=explanation, 
            classdict=classdict, 
            url='')
        print(objectId, '-- annotated!')
        log.write('object %s -- annotated!\n' % objectId)
        return 1
    else:
        print('TEST: \n', objectId, '\n', classification,'\n', explanation, '\n',classdict, '\n')
        return 1
        



def test_annotator(topic_in, group_id):
    # run this test function after each upgrade, without updating to Lasair database.

    consumer = lasair.lasair_consumer('kafka.lsst.ac.uk:9092', group_id, topic_in)

    L = lasair.lasair_client(settings.API_TOKEN)

    topic_out = settings.TOPIC_OUT

    max_alert = 25
 
    n_alert = n_annotate = 0

    print('\n----------- START OF TEST -----------\n')
    while n_alert < max_alert:
        msg = consumer.poll(timeout=20)
        if msg is None:
            break
        if msg.error():
            print(str(msg.error()))
            break

        jsonmsg = json.loads(msg.value())
        objectId       = jsonmsg['objectId'] 
        print('PROCESS OBJECT %s \n' % objectId)
        # annotating_objs.append(objectId) # predict them together
        n_alert += 1
        n_annotate += handle_object(objectId, L, topic_out, 0.70, True)

    print('\n----------- END OF TEST -----------\n')

def run_annotator(topic_in, group_id):
        # kafka consumer that we can suck from
    consumer = lasair.lasair_consumer('kafka.lsst.ac.uk:9092', group_id, topic_in)

    # the lasair client will be used for pulling all the info about the object
    # and for annotating it
    L = lasair.lasair_client(settings.API_TOKEN)

    # TOPIC_OUT is an annotator owned by a user. API_TOKEN must be that users token.
    topic_out = settings.TOPIC_OUT


    # just get a few to start
    max_alert = 10

    # annotating_objs = []
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
        # annotating_objs.append(objectId) # predict them together

        n_alert += 1
        n_annotate += handle_object(objectId, L, topic_out, 0.70)


    log.close()
    print('Annotated %d of %d objects' % (n_annotate, n_alert))

#####################################


if __name__ == '__main__':

    # first we set up pulling the stream from Lasair
    # a fresh group_id gets all, an old group_id starts where it left off
    group_id = settings.GROUP_ID

    # a filter from Lasair, example 'lasair_2SN-likecandidates'
    topic_in = settings.TOPIC_IN

    run_annotator(topic_in, group_id)