'''
This script is for dividing the training, validation, and test set by custom.
'''

import json
import os
import numpy as np
import random
from training import train 
from tensorflow.keras import models
from preprocessing import open_with_h5py
from sklearn.metrics import classification_report

FORDER_PATH = '/Users/xinyuesheng/Documents/astro_projects/scripts/classifier_v2/model_with_data/r_band/v13_v2_optimized_training_v4/'

LABEL_PATH = '/Users/xinyuesheng/Documents/astro_projects/scripts/classifier_v2/model_labels/label_dict_equal_test.json'
h = open(LABEL_PATH, 'r')
label_dict = json.loads(h.read())
h.close()

HASH_PATH = '/Users/xinyuesheng/Documents/astro_projects/scripts/classifier_v2/model_with_data/r_band/v13_v2_3c_20231126/hash_table.json'
h = open(HASH_PATH, 'r')
hash_table = json.loads(h.read())
h.close()
    # reverse hash table
reversed_hash = {}

SN_objs = []
for i in hash_table.keys():
    reversed_hash[hash_table[i]['ztf_id']] = int(i)
    if hash_table[i]['label'] == 0:
        SN_objs.append(hash_table[i]['ztf_id'])


DATA_PATH = '/Users/xinyuesheng/Documents/astro_projects/scripts/classifier_v2/model_with_data/r_band/v13_v2_3c_20231126/data.hdf5'

imageset, labelset, metaset, idx_set = open_with_h5py(DATA_PATH)


# remove other classes here
for k in label_dict['classify'].keys():
    if label_dict['classify'][k] not in label_dict['label'].values():
        ab_idx = np.where(labelset == label_dict["classify"][k])
        imageset, metaset, labelset = np.delete(imageset, ab_idx, 0), np.delete(metaset, ab_idx, 0), np.delete(labelset, ab_idx, 0)
        idx_set = np.delete(idx_set, ab_idx, 0)



def store_summary(exp_id = str, p = float, train_valid = str, test = str, result_dict = dict, metric_dict = dict, save_path = str):
    summary_dict = {'exp_id': exp_id, 'threshold': '%.3f'%float(p), 'train_valid': train_valid, 'test': test}
    metric_dict = {**summary_dict, **metric_dict}
    results_dict = {**summary_dict, **result_dict}

    np.save(save_path + '/test_metrics.npy', metric_dict)
    np.save(save_path + '/result_analysis.npy', results_dict)
 

def get_customized_idxs(reversed_hash, selected_objs):
    '''
    get training, validation, test indexes for three classes.
    '''
    selected_objs_idx = []
    for x in selected_objs:
        selected_objs_idx.append(reversed_hash[x])

    return selected_objs_idx

def get_two_level_objs(_type, p):
    level_path = '/Users/xinyuesheng/Documents/astro_projects/scripts/classifier_v2/model_with_data/r_band/v13_v2_3c_20231126/levels_of_objects/' + _type + '_two_levels_dict_t' + str(p) + '0000.json'
    h = open(level_path, 'r')
    level_objs = json.loads(h.read())
    h.close()
    return level_objs["easy"], level_objs["hard"]

def split_train_valid(train_valid_obj, smallest_valid = 5):
    # take 1/3 to be the validation set, but if the number of validation sample is smaller than two times of 'smallest_valid', divide it up.

    if len(train_valid_obj) < 2 * smallest_valid:
        random.shuffle(train_valid_obj)
        valid_obj = train_valid_obj[:int(1/2*len(train_valid_obj))]
        train_obj = train_valid_obj[int(1/2*len(train_valid_obj)):]
    else:
        random.shuffle(train_valid_obj)
        valid_obj = train_valid_obj[:smallest_valid]
        train_obj = train_valid_obj[smallest_valid:]
    return train_obj, valid_obj

def preprocessing(metaset, train_valid_idx, train_idx, valid_idx, test_idx, output_path):
    def scaling(x, mean, std):
        return (x - mean)/std

    # print(metaset.shape)

    mt_mean = np.nanmean(metaset[train_valid_idx], axis=0)
    mt_std = np.nanstd(metaset[train_valid_idx], axis=0)
    
    s_data = {'mean': mt_mean.astype('float64').tolist(), 'std': mt_std.astype('float64').tolist()}
    with open(output_path + '/scaling_data.json', 'w') as sd:
        json.dump(s_data, sd, indent = 4)

    return scaling(metaset[train_idx], mt_mean, mt_std), scaling(metaset[valid_idx], mt_mean, mt_std), scaling(metaset[test_idx], mt_mean, mt_std)

def model_predict(BClassifier_path, img_data, meta_data):
    BClassifier = models.load_model(BClassifier_path)
    results = BClassifier.predict({'image_input': img_data, 'meta_input': meta_data})
    # BClassifier.plot_CM(img_data, meta_data, labels, save_path = BClassifier_path, suffix = 'test')
    return results

def get_metrics(predictions, true_labels, label_names):
    # calculate the recall, precision, f1
    predicted_labels = np.argmax(predictions, axis=-1)
    true_labels = true_labels.flatten()
    return classification_report(y_true=true_labels, y_pred=predicted_labels, target_names=label_names, output_dict=True)
    
def find_dataset_idx(x):
    x = np.array(x)
    return np.nonzero(np.isin(idx_set, x))[0].tolist()

def gain_results(model_path, image_train, meta_train, label_train, obj_train, image_valid, meta_valid, label_valid, obj_valid, image_test, meta_test, label_test, obj_test, label_names):
    '''
    get prediction results for training, validation and test sets.
    '''
    predict_train = model_predict(model_path, image_train, meta_train)
    predict_valid = model_predict(model_path, image_valid, meta_valid)
    predict_test = model_predict(model_path, image_test, meta_test)
    metric_dict = get_metrics(predict_test, label_test, label_names)

    results_dict = {'training':{'SN':{}, 'SLSN':{}, 'TDE':{}}, 'valid':{'SN':{}, 'SLSN':{}, 'TDE':{}}, 'test':{'SN':{}, 'SLSN':{}, 'TDE':{}}}
    for x, y, z in zip(predict_train, label_train, obj_train):
        if y == 0:
            results_dict['training']['SN'][z] = x.tolist()
        elif y == 1:
            results_dict['training']['SLSN'][z] = x.tolist()
        elif y == 2:
            results_dict['training']['TDE'][z] = x.tolist()
    for x, y, z in zip(predict_valid, label_valid, obj_valid):
        if y == 0:
            results_dict['valid']['SN'][z] = x.tolist()
        elif y == 1:
            results_dict['valid']['SLSN'][z] = x.tolist()
        elif y == 2:
            results_dict['valid']['TDE'][z] = x.tolist()
    for x, y, z in zip(predict_test, label_test, obj_test):
        if y == 0:
            results_dict['test']['SN'][z] = x.tolist()
        elif y == 1:
            results_dict['test']['SLSN'][z] = x.tolist()
        elif y == 2:
            results_dict['test']['TDE'][z] = x.tolist()

    for x in ['training', 'valid', 'test']:
        for y,z in zip(['SN', 'SLSN', 'TDE'],[0,1,2]):
            results_dict[x][y]['averaged_tp_probs'] = np.mean([results_dict[x][y][i][z] for i in results_dict[x][y].keys()])
    # print(results_dict)
    return results_dict, metric_dict


def optimized_training(reversed_hash):

    focus_level = ['easy', 'hard', 'random']
    threshold_level = ['0.65', '0.75', '0.85']

    for p in threshold_level:
        # find the paths
        SLSN_easy, SLSN_hard = get_two_level_objs('slsn', p)
        TDE_easy, TDE_hard = get_two_level_objs('tde', p)

        emsemble_SLSN = [SLSN_easy, SLSN_hard]
        emsemble_TDE = [TDE_easy, TDE_hard]
        SLSN_all = SLSN_easy + SLSN_hard
        TDE_all = TDE_easy + TDE_hard

        m = 0 
        while m < 20: # each condition run 20 models
            random.seed(m)
            x = 0
            while x < 3: # slsn three levels 

                #SLSN
                if x == 2:
                    # random shuffle, 5 in the test set
                    random.shuffle(SLSN_all)
                    SLSN_test_objs = SLSN_all[:15]
                    SLSN_train_valid_objs = SLSN_all[15:]
                else:
                    SLSN_train_valid_objs = emsemble_SLSN[x]
                    SLSN_test_objs = [x for x in SLSN_all if x not in SLSN_train_valid_objs]  

                slsn_train_objs, slsn_valid_objs = split_train_valid(SLSN_train_valid_objs, smallest_valid = 8) 

                y = 0
                while y < 3: # tde three levels 

                    model_path = FORDER_PATH + 'exp_p_' + p + '_SLSN_' + focus_level[x] + '_TDE_' + focus_level[y] + '_id_' + str(m)
                    
                    if os.path.isdir(model_path) == False:
                        os.mkdir(model_path)
                    else:
                        print('model %s exists!\n' %(model_path))
                        y += 1
                        continue
                    
                    exp_id = 'exp_p_' + p + '_SLSN_' + focus_level[x] + '_TDE_' + focus_level[y] + '_id_' + str(m)

                    # TDE
                    if y == 2:
                        random.shuffle(TDE_all)
                        TDE_test_objs = TDE_all[:15]
                        TDE_train_valid_objs = TDE_all[15:]
                    else:
                        TDE_train_valid_objs = emsemble_TDE[y]
                        TDE_test_objs = [x for x in TDE_all if x not in TDE_train_valid_objs] 
                    tde_train_objs, tde_valid_objs = split_train_valid(TDE_train_valid_objs, smallest_valid = 8) 

                    # SN random shuffle  
                    random.shuffle(SN_objs)
                    SN_test_objs = SN_objs[:300]
                    SN_train_valid_objs = SN_objs[300:]
                    sn_train_objs = SN_train_valid_objs[50:]
                    sn_valid_objs = SN_train_valid_objs[:50]


                    
                    if slsn_train_objs is not None and tde_train_objs is not None:
                        train_valid_objs = SN_train_valid_objs + SLSN_train_valid_objs + TDE_train_valid_objs

                        test_objs = SN_test_objs + SLSN_test_objs + TDE_test_objs
                        train_objs = slsn_train_objs + tde_train_objs + sn_train_objs
                        valid_objs = slsn_valid_objs + tde_valid_objs + sn_valid_objs

                        # train_valid_idx, test_idx = get_customized_idxs(reversed_hash, train_valid_objs, test_objs)
                        train_valid_idx = get_customized_idxs(reversed_hash, train_valid_objs)
                        train_idx = get_customized_idxs(reversed_hash, train_objs)
                        valid_idx = get_customized_idxs(reversed_hash, valid_objs)
                        test_idx = get_customized_idxs(reversed_hash, test_objs)
                        
                        # scaling metadata of training and validation
                        train_idx = find_dataset_idx(train_idx)
                        valid_idx = find_dataset_idx(valid_idx)
                        test_idx = find_dataset_idx(test_idx)
                        train_valid_idx = find_dataset_idx(train_valid_idx)


                        meta_train, meta_valid, meta_test = preprocessing(metaset, train_valid_idx, train_idx, valid_idx, test_idx, model_path)

                        # train and validation 
                        train(imageset[train_idx], meta_train, labelset[train_idx], imageset[valid_idx], meta_valid, labelset[valid_idx], label_dict["label"], neurons = [[64,3],[128,3]], res_cnn_group = None, meta_only = False, 
          batch_size = 128, epoch = 300, learning_rate = 8e-5, model_name = model_path)
                        
                        # get results
                        results_dict, metric_dict = gain_results(model_path, imageset[train_idx], meta_train, labelset[train_idx], train_objs, imageset[valid_idx], meta_valid, labelset[valid_idx], valid_objs, imageset[test_idx], meta_test, labelset[test_idx], test_objs, list(label_dict['label'].keys()))
                         
                     
                    else:
                        # fill in none to json
                        results_dict = {}
                        metric_dict = {}

                   
                    if x == 0:
                        SLSN_test = 1
                    elif x == 1:
                        SLSN_test = 0
                    elif x == 2:
                        SLSN_test = 2
                    if y == 0:
                        TDE_test = 1
                    elif y == 1:
                        TDE_test = 0
                    elif y == 2:
                        TDE_test = 2

                    store_summary(exp_id, p, train_valid = 'SLSN_%s, TDE_%s'% (focus_level[x], focus_level[y]), test ='SLSN_%s, TDE_%s'%(focus_level[SLSN_test], focus_level[TDE_test]), result_dict=results_dict, metric_dict = metric_dict, save_path = model_path)
                            

                    y += 1
                x += 1
            m += 1
            



if __name__ == '__main__':

    
    optimized_training(reversed_hash)