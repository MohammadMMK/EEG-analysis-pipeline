
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
    with open(os.path.join( data_path, 'BadTrialsChannel_manualDetected.pkl'), "rb") as f:
        all_bads = pickle.load(f)

    bads_channel= all_bads[id]['channel_names']
    bad_trials= all_bads[id]['trial_numbers']
    bridged_channels= all_bridged_channels[id] 
    sub = preprocess(id)
    raw = sub.load_data()

    # 1. remove noisy channels
    raw.info['bads'] = bads_channel

    # 2. Filter the data
    raw.notch_filter([50,100], fir_design='firwin', skip_by_annotation='edge')
    raw.filter(l_freq=1, h_freq= lowPassFilter)

    # 3. segment the data from stim to response (remove noisy trials and trials with wrong answers)
    events = mne.find_events(raw)
    all_events = sub.get_all_events_times( events).dropna()
    pre_ica_data = sub.segment_stimRt(raw, all_events, bad_trials)

    # interpolate bridged channels
    pre_ica_data = mne.preprocessing.interpolate_bridged_electrodes(pre_ica_data, bridged_channels['bridged_idx'], bad_limit=4) 
    return pre_ica_data

def pre_HA_denoise(id, lowPassFilter = None):

    with open(os.path.join( data_path, 'bridged_channels_analysis.pkl'), "rb") as f:
        all_bridged_channels = pickle.load(f)
    with open(os.path.join( data_path, 'BadTrialsChannel_manualDetected.pkl'), "rb") as f:
        all_bads = pickle.load(f)
    path_ic = os.path.join(data_path, f'S{id}_ica_infomax_LpF.fif')
    bads_channel= all_bads[id]['channel_names']
    bad_trials= all_bads[id]['trial_numbers']
    noisy_components = all_bads[id]['noisy_components']
    bridged_channels= all_bridged_channels[id] 
    sub = preprocess(id)
    raw = sub.load_data()

    # 1. remove noisy channels
    raw.info['bads'] = bads_channel

    # 2. Filter the data
    raw.notch_filter([50,100], fir_design='firwin', skip_by_annotation='edge')
    raw.filter(l_freq=1, h_freq= lowPassFilter)

    # 3. segment the data from stim to response (remove noisy trials and trials with wrong answers)
    events = mne.find_events(raw)
    all_events = sub.get_all_events_times( events).dropna()
    pre_ica_data = sub.segment_stimRt(raw, all_events, bad_trials)
    
    # 4. ICA cleaning
    ica = mne.preprocessing.read_ica(path_ic)
    ica.exclude = noisy_components
    ica.apply(pre_ica_data)
    # interpolate bridged channels
    pre_ica_data = mne.preprocessing.interpolate_bridged_electrodes(pre_ica_data, bridged_channels['bridged_idx'], bad_limit=4) 
    # 5. interpolate bad channels
    pre_ica_data.interpolate_bads()
    # 6. re-reference to average
    pre_ica_data.set_eeg_reference(ref_channels='average')
    # 7. z-score
    data = pre_ica_data.get_data()
    means = data.mean(axis=1, keepdims=True)
    stds  = data.std(axis=1, keepdims=True)
    pre_ica_data._data = (data - means) / stds

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
    del ica, pre_ica_data
    gc.collect()
    return

def detect_HA_outliers(subject, threshold=6):

    datanorm = pre_HA_denoise(subject , 30)
    annotation = datanorm._annotations
    df = annotation.to_data_frame()
    df = df[df['duration'] != 0]
    df= df.reset_index(drop=True)
    # ----- Extract epochs and data -----
    all_raws = []
    all_data = []
    all_description = []
    for i, duration in enumerate(df['duration']):
        start = np.sum(df['duration'][:i]) if i > 0 else 0
        end = start + duration
        # Crop raw for this trial
        raw_epoch = datanorm.copy().crop(tmin=start, tmax=end)
        all_raws.append(raw_epoch)
        data = raw_epoch.get_data(picks='eeg')
        all_data.append(data)
        all_description.append(df['description'][i])

    # ----- Compute thresholds -----
    concat = datanorm.copy().get_data(picks='eeg')
    mean = np.mean(concat, axis=1)
    std = np.std(concat, axis=1)

    channel_thresholds = mean + threshold * std  # shape: (n_channels,)

    # ----- Detect threshold crossings -----
    n_trials = len(all_data)
    n_channels = concat.shape[0]
    detection = np.zeros((n_channels, n_trials ), dtype=bool)
    for i, trial in enumerate(all_data):
        for ch in range(n_channels):
            if np.any(np.abs(trial[ch, :]) > channel_thresholds[ch]):
                detection[ch, i] = True

    # ----- Interpolate and flag bad trials -----
    trials_to_remove = []
    for i, raw_epoch in enumerate(all_raws):
        bad_idx = np.where(detection[i])[0]
        # If >20% channels bad, mark for removal
        if len(bad_idx) > 0.2 * n_channels:
            trials_to_remove.append((i, all_description[i]))
            continue
    return detection, trials_to_remove

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
    with open(os.path.join(data_path, 'BadTrialsChannel_manualDetected.pkl'), "rb") as f:
        all_bads = pickle.load(f)

    pre_concatenated_data = []

    for subject_id in ids:
        # Per‐subject params
        bads_channel = all_bads[subject_id]['channel_names']
        bad_trials   = all_bads[subject_id]['trial_numbers']
        bridged      = all_bridged_channels[subject_id]

        # 1. load & preprocess
        sub = preprocess(subject_id)
        raw = sub.load_data()
        raw.info['bads'] = bads_channel

        # 2. filter
        raw.notch_filter([50, 100], fir_design='firwin', skip_by_annotation='edge')
        raw.filter(l_freq=1, h_freq=lowPassFilter_pregICA)

        # 3. epoch & drop bad trials
        events    = mne.find_events(raw)
        all_events = sub.get_all_events_times( events).dropna()
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
            new_raw, bridged['bridged_idx'], bad_limit=4
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

def pre_gICA4(ids,
            ica_name='ica_infomax',
            lowPassFilter_pregICA=30,
            file_name='groupData'):

    # Load once
    with open(os.path.join(data_path, 'bridged_channels_analysis.pkl'), "rb") as f:
        all_bridged_channels = pickle.load(f)
    with open(os.path.join(data_path, 'BadTrialsChannel_manualDetected.pkl'), "rb") as f:
        all_bads = pickle.load(f)

    pre_concatenated_data = []

    for subject_id in ids:
        # Per‐subject params
        bads_channel = all_bads[subject_id]['channel_names']
        bad_trials   = all_bads[subject_id]['trial_numbers']
        bridged      = all_bridged_channels[subject_id]
        bad_components = all_bads[subject_id]['noisy_components']
        detected_noise = all_bads[subject_id]['detection']

        # 1. load & preprocess
        sub = preprocess(subject_id)
        raw = sub.load_data()
        raw.info['bads'] = bads_channel

        # 2. filter
        raw.notch_filter([50, 100], fir_design='firwin', skip_by_annotation='edge')
        raw.filter(l_freq=1, h_freq=lowPassFilter_pregICA)

        # 3. epoch & drop bad trials
        events    = mne.find_events(raw)
        all_events = sub.get_all_events_times( events).dropna()
        new_raw    = sub.segment_stimRt(raw, all_events, bad_trials)

        # 4. ICA cleaning
        ica_path = os.path.join(data_path, f'S{subject_id}_{ica_name}.fif')
        if not os.path.exists(ica_path):
            print(f'ICA missing for subject {subject_id}; skipping.')

        ica = mne.preprocessing.read_ica(ica_path)

        noisy = bad_components
        ica.exclude = noisy
        ica.apply(new_raw )


        annotation = new_raw._annotations
        df = annotation.to_data_frame()
        df = df[df['duration'] != 0]
        df= df.reset_index(drop=True)
        # ----- Extract epochs and data -----
        all_raws = []
        removed_trial_HA = []
        for i, duration in enumerate(df['duration']):
            start = np.sum(df['duration'][:i]) if i > 0 else 0
            end = start + duration
            # Crop raw for this trial
            raw_epoch = new_raw.copy().crop(tmin=start, tmax=end)
            detected_noiseT = detected_noise.copy().transpose()
            bad_idx = np.where(detected_noiseT[i])[0]
            n_total_bads = len(bad_idx) + len(raw_epoch.info['bads'])
            if n_total_bads > 30:
                print(f"Skipping trial {i} for subject {subject_id} due to excessive bad channels.")
                removed_trial_HA.append(df['description'][i])
            elif n_total_bads > 0:
                print(f"Interpolating {n_total_bads} bad channels in trial {i} for subject {subject_id}.")
                raw_epoch.info['bads'] += [raw_epoch.ch_names[idx] for idx in bad_idx]
                print(f"Bad channels in trial {i}: {raw_epoch.info['bads']}")
                raw_epoch.interpolate_bads()
                raw_epoch.info['bads'] = []  # Clear bads after interpolation
                all_raws.append(raw_epoch)
            elif n_total_bads == 0:
                print(f"No bad channels in trial {i} for subject {subject_id}.")
                all_raws.append(raw_epoch)
                
        new_raw = mne.concatenate_raws(all_raws)
        # 5–7. interpolate & re‐ref
        new_raw = mne.preprocessing.interpolate_bridged_electrodes(
            new_raw, bridged['bridged_idx'], bad_limit=4
        )
        # new_raw.interpolate_bads()
        new_raw.set_eeg_reference(ref_channels='average')

        # 8. z‐score
        data = new_raw.get_data()
        means = data.mean(axis=1, keepdims=True)
        stds  = data.std(axis=1, keepdims=True)
        new_raw._data = (data - means) / stds
        # store and then clear
        pre_concatenated_data.append(new_raw)

        # --- clear out all the big locals ---
        del (sub, raw, events, all_events, ica,
                data, means, stds, bridged,
                bads_channel, bad_trials)
        gc.collect()

    # concatenate and save
    concat = mne.concatenate_raws(pre_concatenated_data)
    concat.save(os.path.join(data_path, f'{file_name}.fif'), overwrite=True)

    return