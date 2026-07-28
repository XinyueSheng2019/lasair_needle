import json, sys, os
from settings import *

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "source"))
sys.path.insert(0, PROJECT_ROOT)
import lasair

from astropy.time import Time
import numpy as np
import logs
from needle_stream.get_input import *
from needle_stream.get_predict import *
from needle_stream.preprocessing import apply_offset_tde_gate, get_host_offset_arcsec
from datetime import datetime
from settings import *



# add directories 
if os.path.exists(NEEDLE_OBJ_PATH) is False:
    os.makedirs(NEEDLE_OBJ_PATH)


if os.path.exists('logs') is False:
    os.makedirs('logs')

if os.path.exists(RECORD_PATH) is False:
    os.makedirs(RECORD_PATH)

# set a shared log (usable across all modules; execution order preserved)
logs.set_log(open('logs/log_' + datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.txt', 'w'))
log = logs.log




def update_records(objectId=list, classdict=dict, classification=str):
    # store records to a folder, where contains obj prediction file with JSON format
    file_path = os.path.join(RECORD_PATH, f'{objectId}.json')
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



def handle_object(objectId=str, L=lasair.lasair_client, threshold = 0.70, test = False): 

    def check_tns_update():
        if test: # test object so ignore TNS update.
            return False
        else:
            # check if the object is updated in TNS
            if 'TNS' in objectInfo.keys() and 'type' in objectInfo['TNS'].keys() and objectInfo['TNS']['type'] is not None:
                log.write('object %s is updated in TNS, type %s, skipped.\n' % (objectId, objectInfo['TNS']['type']))
                return True
            else:  
                return False
        
    def remove_unvalid_object():
        # check good quality photometry
        if test: # test object so ignore unvalid object.
            return False
        else:
            if objectInfo['objectData']['ncand'] < 2:
                log.write('object %s is removed as there is no valid or enough detection.\n' % objectId)
                return True
            else:
                return False

    def remove_fading(window_size = 10):
        # if it has been 10 days after the latest peak detection, regard it as fading.
        if test: # test object so ignore fading.
            return False
        else:
            if objectInfo['objectData']['latestMjd'] - objectInfo['objectData']['peakMjd'] > window_size:
                return True
            else:
                return False

    def remove_long(max_len = 60):
        # is the first detection longer than 60 days?
        if test: # test object so ignore long.
            return False
        else:
            if objectInfo['objectData']['jdmin'] - objectInfo['objectData']['jdmax'] >= max_len:
                log.write('object %s is removed as it is longer than %d days.\n' % (objectId, max_len))
                return True
            else:
                return False



    def remove_agn_like(candidates): # function in test
        #  this function is for objects labelled as NT, orphan or SN, but might be recurrent AGN. We remove them at this stage, cautiously.

        def is_changing(n=5, threshold = 0.2):
            # check if the candidate is rising or fading.
            # 0.2 mag is the threshold of the change.
            if len(objectInfo['candidates']) < n:
                n = len(objectInfo['candidates'])
            check_mags = np.array([c['magpsf'] for c in sorted_candidates[-n:]])
            diff_mags = np.diff(check_mags)
            return np.any(np.abs(diff_mags) >= threshold) 

        def check_std(window_size = 20, threshold = 0.2): # double check. 
            # check the std of the candidates
            if len(candidates) < window_size:
                return False
            else:
                std_list = []
                for i in range(len(candidates) - window_size + 1):
                    window = sorted_candidates[i:i+window_size]
                    std_list.append(np.std([c['magpsf'] for c in window]))
                if np.max(std_list) >= threshold:
                    return False
                else:
                    return True

        # only remain SN-like candidates, if for all window_size, the std of the candidates is less than threshold, it is assumed to be AGN.
        sorted_candidates = sorted(objectInfo['candidates'], key=lambda x: x['mjd'])

        # if is_rising() or is_declining():
        #     return False
        # elif check_std():
        #     return True
        # else:
        #     return False


    # from the objectId, we can get all the info that Lasair has
    print('objectId: ', objectId)

    objectInfo = L.object(objectId, lasair_added=True, lite=True)
    if test:
        if not os.path.exists(os.path.join(NEEDLE_OBJ_PATH, objectId)):
            os.makedirs(os.path.join(NEEDLE_OBJ_PATH, objectId))
        
        with open(os.path.join(NEEDLE_OBJ_PATH, objectId, 'objectInfo.json'), 'w') as f:
            json.dump(objectInfo, f, indent=4)
            f.close()

    if check_tns_update() or remove_unvalid_object() or remove_fading() or remove_long():
        return 0
    
    img_data, _, meta_mixed, findhost = collect_data_from_lasair(objectId = objectId, objectInfo = objectInfo)


    if img_data is None:
        print('object %s images in g and r bands do not pass criteria or not found.\n' % objectId)
        return 0


    if findhost:
        result_mixed = needle_th_prediction(img_data, meta_mixed)
        calibrated_thresholds = CALIBRATED_THRESHOLDS['hosted']
        SN_mix = float(result_mixed[0][0]) if result_mixed is not None else None
        SLSN_mix = float(result_mixed[0][1]) if result_mixed is not None else None
        TDE_mix = float(result_mixed[0][2]) if result_mixed is not None else None
    else:
        log.write('object %s host meta not found, use binary NEEDLE-T\n' % objectId)
        result_mixed = needle_t_prediction(img_data, meta_mixed)
        calibrated_thresholds = CALIBRATED_THRESHOLDS['hostless']
        SN_mix = float(result_mixed[0][0]) if result_mixed is not None else None
        SLSN_mix = float(result_mixed[0][1]) if result_mixed is not None else None
        TDE_mix = 0.0 if result_mixed is not None else None

    if TDE_mix is not None:
        TDE_mix = apply_offset_tde_gate(
            TDE_mix,
            get_host_offset_arcsec(objectInfo),
            object_id=objectId,
            log=log,
            max_offset_arcsec=OFFSET_TDE_MAX_ARCSEC,
        )
    
    classdict = {'SN': 1 - SLSN_mix - TDE_mix, 'SLSN-I': SLSN_mix, 'TDE': TDE_mix}
    print('classdict: ', classdict)

    # Decide final classification using per-class calibrated thresholds
    class_values = list(classdict.values())
    best_idx = int(np.argmax(class_values))
    best_class = LABEL_LIST[best_idx]
    class_threshold = calibrated_thresholds.get(best_class, threshold)
    if class_values[best_idx] >= class_threshold:
        classification = best_class
    else:
        classification = 'unclear'
    
    # Record/track the first time stamp this object crosses the threshold
    explanation = update_records(objectId, classdict, classification)

    # Wrap scalars as single-element lists for update_to_lasair
    update_to_lasair(
        L,
        [objectId],
        [classification],
        [explanation],
        [classdict],
        [None],
        test,
    )
    return 1


    

def update_to_lasair(L, objectId=list, classification=list, explanation=list, classdict=dict, url=None, test=False):
    # push the annotation to the Lasair database
    lasair_web_base = LASAIR_ENDPOINT.rstrip('/').removesuffix('/api')

    if not test:
        for i in range(len(objectId)):
            annotation_url = url[i] if url and url[i] else f"{lasair_web_base}/object/{objectId[i]}/"
            L.annotate(
                TOPIC_OUT,
                objectId[i],
                classification[i],
                version='needle-v2',
                explanation=explanation[i],
                classdict=classdict[i],
                url=annotation_url,
            )
    else:
        for i in range(len(objectId)):
            annotation_url = url[i] if url and url[i] else f"{lasair_web_base}/object/{objectId[i]}/"
            msg = (
                f"TEST: \n"
                f"{objectId[i]}\n"
                f"{classification[i]}\n"
                f"{explanation[i]}\n"
                f"{classdict[i]}\n"
                f"{annotation_url}\n"
            )
            print(msg)
            log.write(msg)
    return 1


def test_annotator(test_objectId = None):
    # run this test function after each upgrade, without updating to Lasair database.

    consumer = lasair.lasair_consumer('kafka.lsst.ac.uk:9092', GROUP_ID, TOPIC_IN)

    # Use Lasair-specific configuration (API token, output topic)
    L = lasair.lasair_client(API_TOKEN, endpoint=LASAIR_ENDPOINT)

    if test_objectId is None:
        
        max_alert = 25
    
        n_alert = n_annotate = 0
        print('\n----------- START OF TEST -----------\n')
        while n_alert < max_alert:
            msg = consumer.poll(timeout=20)
            print(msg)
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
            n_annotate += handle_object(objectId, L, 0.70, True)
    else:
        handle_object(test_objectId, L, 0.70, True)
    

    print('\n----------- END OF TEST -----------\n')


def run_annotator(parallel = False):
    # kafka consumer that we can suck from
    consumer = lasair.lasair_consumer('kafka.lsst.ac.uk:9092', GROUP_ID, TOPIC_IN)

    # the lasair client will be used for pulling all the info about the object
    # and for annotating it
    L = lasair.lasair_client(API_TOKEN, endpoint=LASAIR_ENDPOINT)


    # just get a few to start
    max_alert = 200
    n_annotate = 0
    if parallel:
        import concurrent.futures
        for batch in range(max_alert // 10):
            objectIds = [json.loads(consumer.poll(timeout=20).value())['objectId'] for _ in range(10) if consumer.poll(timeout=20) is not None]
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(handle_object, objectId, L, 0.70, False) for objectId in objectIds]
                results = [future.result() for future in futures]
                n_annotate += sum(results)
    else:
        n_alert = 0
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
            n_annotate += handle_object(objectId, L, 0.70)


    logs.close_log()
    print('Annotated %d of %d objects' % (n_annotate, n_alert))

#####################################


if __name__ == '__main__':

    # first we set up pulling the stream from Lasair
    # a fresh group_id gets all, an old group_id starts where it left off


    # a filter from Lasair, example 'lasair_2SN-likecandidates'
  

    run_annotator(parallel = False)
    # test_annotator(test_objectId = 'ZTF26abjqqyj')