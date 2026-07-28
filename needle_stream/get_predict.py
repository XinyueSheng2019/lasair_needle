import os
import numpy as np
from astropy.io import fits
from needle_stream.preprocessing import single_transient_preprocessing
from needle_stream.preprocessing import feature_reduction_for_mixed_band
from needle_stream.preprocessing import apply_data_scaling
from needle_stream.preprocessing import feature_reduction_for_mixed_band_no_host
from needle_stream.preprocessing import impute_meta_missing_values
from needle_train.transient_model import *
# from tensorflow.keras import models 

from settings import *

custom_objects = {
    'F1PerClassMetrics': F1PerClassMetrics,
    'CustomLearningRateSchedule': CustomLearningRateSchedule,
    'PrecisionPerClassMetrics': PrecisionPerClassMetrics,
    'RecallPerClassMetrics': RecallPerClassMetrics,
    'focal_loss_fixed_modified': focal_loss_modified()
}



def load_needle_classifier(model_path, label_dict, model_epoch=None):
    """Rebuild a TransientClassifier and load saved weights."""
    config_path = os.path.join(model_path, 'model_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Missing {config_path}. Retrain with the current code or provide model_config.json."
        )

    with open(config_path, 'r') as config_file:
        model_config = json.load(config_file)

    # if model_config.get('num_classes') == 2:
    #     label_dict = label_dict_full['label-hostless']
    # else:
    #     label_dict = label_dict_full['label-hosted']

    feature_importance = model_config.get('feature_importance')
    if feature_importance is not None:
        feature_importance = np.array(feature_importance)

    classifier = TransientClassifier(
        label_dict,
        N_image=model_config['N_image'],
        image_dimension=model_config['image_dimension'],
        meta_dimension=model_config['meta_dimension'],
        neurons=model_config['neurons'],
        Resnet_op=model_config.get('resnet_op', False),
        meta_only=model_config.get('meta_only', False),
        feature_importance=feature_importance,
    )

    dummy_image = tf.zeros((1, model_config['N_image'], model_config['N_image'], model_config['image_dimension']))
    dummy_meta = tf.zeros((1, model_config['meta_dimension']))
    classifier({'image_input': dummy_image, 'meta_input': dummy_meta})

    def _resolve_epoch_weights_path(epoch):
        candidates = [
            os.path.join(model_path, f'model_epoch_{epoch}.weights.h5'),
            os.path.join(model_path, f'model_epoch_{int(epoch):02d}.weights.h5'),
        ]
        for weights_path in candidates:
            if os.path.exists(weights_path):
                return weights_path
        return None

    if model_epoch is not None:
        weights_path = _resolve_epoch_weights_path(model_epoch)
        if weights_path is None:
            raise FileNotFoundError(
                f"No checkpoint weights found for epoch {model_epoch} under {model_path}. "
                f"Expected model_epoch_{model_epoch}.weights.h5 or "
                f"model_epoch_{int(model_epoch):02d}.weights.h5."
            )
        classifier.load_weights(weights_path)
        return classifier

    for weights_name in ('best_weights.weights.h5', 'final_weights.weights.h5'):
        weights_path = os.path.join(model_path, weights_name)
        if os.path.exists(weights_path):
            classifier.load_weights(weights_path)
            return classifier

    raise FileNotFoundError(
        f"No weights found under {model_path}. "
        "Expected best_weights.weights.h5 or final_weights.weights.h5."
    )


def needle_th_prediction(img_data, meta_mixed):
    # for lasair-needle2.0, we only use the mixed meta. If one object only has r-band data, then g-band features will be padded with zeros.

    if meta_mixed is not None:
        result_mixed = [] 
        _img_data, meta_mixed = single_transient_preprocessing(img_data, meta_mixed)
        meta_mixed = impute_meta_missing_values(
            meta_mixed,
            has_host=True,
            use_offset_sentinel=USE_EXTENDED_OFFSET_FEATURES,
            missing_offset_sentinel=MISSING_OFFSET_SENTINEL_ARCSEC,
        )
        meta_mixed, _ = feature_reduction_for_mixed_band(
            meta_mixed,
            extended_offset_features=USE_EXTENDED_OFFSET_FEATURES,
            offset_tde_max_arcsec=OFFSET_TDE_MAX_ARCSEC,
        )

        mixed_classifier = load_needle_classifier(MODEL_PATH_TH, LABEL_DICT_HOSTED, CALIBRATED_EPOCHS['hosted'])
        _meta_mixed = apply_data_scaling(meta_mixed, SCALING_PATH['hosted'])
        result_mixed = mixed_classifier.predict({'image_input': _img_data, 'meta_input': _meta_mixed})
    else:
        result_mixed = None

    return result_mixed


def needle_t_prediction(img_data, meta_mixed):
    # binary classifier to selec t SLSN-I or SN, as TDE should be filtered out by previous steps.

    if meta_mixed is not None:
        result_mixed = [] 
        _img_data, meta_mixed = single_transient_preprocessing(img_data, meta_mixed)
        meta_mixed =  np.nan_to_num(meta_mixed)
        meta_mixed, _ = feature_reduction_for_mixed_band_no_host(meta_mixed)

        mixed_classifier = load_needle_classifier(MODEL_PATH_T, LABEL_DICT_HOSTLESS, CALIBRATED_EPOCHS['hostless'])
        _meta_mixed = apply_data_scaling(meta_mixed, SCALING_PATH['hostless'])
        result_mixed = mixed_classifier.predict({'image_input': _img_data, 'meta_input': _meta_mixed})
    else:
        result_mixed = None
  
    return result_mixed

