
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

def ICA_denoise(id, pre_ica_data, n_components=None, decim=2, overwrite = False):
    ICA_path = os.path.join(data_path, f'S{id}_ica_infomax.fif')
    if os.path.exists(ICA_path) and overwrite == False:
        print(f'ICA already exists for subject {id}, skipping ICA computation.')
        return 
    ica = mne.preprocessing.ICA(n_components = n_components, method= 'infomax', fit_params=dict(extended=True))
    ica.fit(pre_ica_data, decim=decim)
    ica.save(ICA_path, overwrite=True)
    return


def pre_gICA(ids, lowPassFilter = 30, noise_type = 'blinks', file_name = 'groupData'):

    with open(os.path.join( data_path, 'bridged_channels_analysis.pkl'), "rb") as f:
        all_bridged_channels = pickle.load(f)
    with open(os.path.join( data_path, 'bad_channels_detected.pkl'), "rb") as f:
        all_bads = pickle.load(f)
    pre_concatenated_data = []
    for id in ids:

        bads_channel= all_bads[id]['channel_names']
        bad_trials= all_bads[id]['trial_numbers']
        blink_components = all_bads[id]['ica_blinks']
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
        new_raw = sub.segment_stimRt(raw, all_events, bad_trials)

        # 4. remove blink components 
        ica_path = os.path.join(data_path, f'S{id}_ica_infomax.fif')
        ica = mne.preprocessing.read_ica(ica_path)
        path_labels = os.path.join(data_path, f'S{id}_ica_infomax_labels.pkl')
        if os.path.exists(path_labels):
            with open(path_labels, "rb") as f:
                ica_labels = pickle.load(f)
        else:
            from mne_icalabel import label_components
            ic_labels = label_components(new_raw, ica, method="iclabel")
            # save the labels
            with open(path_labels, "wb") as f:
                pickle.dump(ic_labels, f)
        noisy_components = get_noisyICs(ic_labels, threshold= 0.7, noise_type=noise_type)
        ica.exclude = noisy_components
        ica.apply(new_raw)

        # 5. interpolate bridged channels
        new_raw = mne.preprocessing.interpolate_bridged_electrodes(new_raw, bridged_channels['bridged_idx'], bad_limit=3) 
        # 6. interpolate bad channels
        new_raw.interpolate_bads()
        # 7. rereference the data
        new_raw.set_eeg_reference(ref_channels='average')
        # 8. zscore the data
        data = new_raw.get_data()
        chan_means = np.mean(data, axis=1, keepdims=True)
        chan_stds  = np.std(data,  axis=1, keepdims=True)
        zscored_data = (data - chan_means) / chan_stds
        new_raw._data = zscored_data
        pre_concatenated_data.append(new_raw)
    # concatenate the data
    concat_data = mne.concatenate_raws(pre_concatenated_data)
    # save the data
    concat_data.save(os.path.join(data_path, f'{file_name}.fif'), overwrite=True)
    return concat_data