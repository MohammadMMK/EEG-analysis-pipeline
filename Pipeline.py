
import os 
import mne
import numpy as np
import pandas as pd
import pickle

from config import data_path
from functions import preprocess,get_noisyICs

def pre_ica_denoise(id, lowPassFilter = None):

    with open(os.path.join( data_path, 'bridged_channels_analysis.pkl'), "rb") as f:
        all_bridged_channels = pickle.load(f)
    with open(os.path.join( data_path, 'bad_channels_detected.pkl'), "rb") as f:
        all_bads = pickle.load(f)

    bads_channel= all_bads[id]['channel_names']
    bad_trials= all_bads[id]['trial_numbers']
    bridged_channels= all_bridged_channels[5][id] 
    sub = preprocess(id)
    raw = sub.load_data()

    # 1. remove noisy channels
    raw.info['bads'] = bads_channel

    # 2. Filter the data
    raw.notch_filter([50,100], fir_design='firwin', skip_by_annotation='edge')
    raw.filter(l_freq=1, h_freq= lowPassFilter)

    # 3. segment the data from stim to response (remove noisy trials and trials with wrong answers)
    events = mne.find_events(raw)
    all_events = sub.get_all_events_times(id, events).dropna()
    pre_ica_data = sub.segment_stimRt(raw, all_events, bad_trials)

    # interpolate bridged channels
    pre_ica_data = mne.preprocessing.interpolate_bridged_electrodes(pre_ica_data, bridged_channels['bridged_idx'], bad_limit=3) 
    return pre_ica_data

def ICA_denoise(id, lowPassFilter = None, n_components=None, decim=2, ica_name = 'ica_infomax', overwrite = False):
    ICA_path = os.path.join(data_path, f'S{id}_{ica_name}.fif')
    if os.path.exists(ICA_path) and overwrite == False:
        print(f'ICA already exists for subject {id}, skipping ICA computation.')
        return 
    pre_ica_data = pre_ica_denoise(id, lowPassFilter = lowPassFilter)
    ica = mne.preprocessing.ICA(n_components = n_components, method= 'infomax', fit_params=dict(extended=True))
    ica.fit(pre_ica_data, decim=decim)
    ica.save(ICA_path, overwrite=True)
    return


import os
import pickle
import gc
import mne
import numpy as np

def pre_gICA(ids,
            ica_name='ica_infomax',
            lowPassFilter_pregICA=30,
            noise_type='blinks',
            file_name='groupData'):

    # Load once
    with open(os.path.join(data_path, 'bridged_channels_analysis.pkl'), "rb") as f:
        all_bridged_channels = pickle.load(f)
    with open(os.path.join(data_path, 'bad_channels_detected.pkl'), "rb") as f:
        all_bads = pickle.load(f)

    pre_concatenated_data = []

    for subject_id in ids:
        # Per‐subject params
        bads_channel = all_bads[subject_id]['channel_names']
        bad_trials   = all_bads[subject_id]['trial_numbers']
        bridged      = all_bridged_channels[5][subject_id]

        # 1. load & preprocess
        sub = preprocess(subject_id)
        raw = sub.load_data()
        raw.info['bads'] = bads_channel

        # 2. filter
        raw.notch_filter([50, 100], fir_design='firwin', skip_by_annotation='edge')
        raw.filter(l_freq=1, h_freq=lowPassFilter_pregICA)

        # 3. epoch & drop bad trials
        events    = mne.find_events(raw)
        all_events = sub.get_all_events_times(subject_id, events).dropna()
        new_raw    = sub.segment_stimRt(raw, all_events, bad_trials)

        # 4. ICA cleaning
        ica_path = os.path.join(data_path, f'S{subject_id}_{ica_name}.fif')
        if not os.path.exists(ica_path):
            print(f'ICA missing for subject {subject_id}; skipping.')
            return

        ica = mne.preprocessing.read_ica(ica_path)
        labels_path = os.path.join(data_path, f'S{subject_id}_{ica_name}_labels.pkl')
        if os.path.exists(labels_path):
            with open(labels_path, "rb") as f:
                ic_labels = pickle.load(f)
        else:
            from mne_icalabel import label_components
            ic_labels = label_components(new_raw, ica, method="iclabel")
            with open(labels_path, "wb") as f:
                pickle.dump(ic_labels, f)

        noisy = get_noisyICs(ic_labels, threshold=0.7, noise_type=noise_type)
        ica.exclude = noisy
        ica.apply(new_raw)

        # 5–7. interpolate & re‐ref
        new_raw = mne.preprocessing.interpolate_bridged_electrodes(
            new_raw, bridged['bridged_idx'], bad_limit=3
        )
        new_raw.interpolate_bads()
        new_raw.set_eeg_reference(ref_channels='average')

        # 8. z‐score
        data = new_raw.get_data()
        means = data.mean(axis=1, keepdims=True)
        stds  = data.std(axis=1, keepdims=True)
        new_raw._data = (data - means) / stds

        # store and then clear
        pre_concatenated_data.append(new_raw)

        # --- clear out all the big locals ---
        del (sub, raw, events, all_events, ica, ic_labels,
             noisy, data, means, stds, bridged,
             bads_channel, bad_trials)
        gc.collect()

    # concatenate and save
    concat = mne.concatenate_raws(pre_concatenated_data)
    concat.save(os.path.join(data_path, f'{file_name}.fif'), overwrite=True)

    return
