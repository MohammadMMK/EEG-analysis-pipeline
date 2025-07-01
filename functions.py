import os 
from config import data_path
import mne
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
from mne.preprocessing import compute_bridged_electrodes
from unidecode import unidecode

class preprocess:
    def __init__(self, id):
        self.id = str(id)
        self.eeg_path = os.path.join(data_path, [f for f in os.listdir(data_path) if self.id in f and f.endswith('.bdf')][0])
        self.raw = self.load_data()


    def load_data(self):
        """Load EEG data from a file."""
        raw = mne.io.read_raw(self.eeg_path, preload=True)

        """Add montage to EEG data."""
        ch_location_file = 'Head128elec.xyz'
        ch_location_path = os.path.join(data_path, ch_location_file)
        # Load channel locations and create montage
        xyz_data = pd.read_csv(ch_location_path, delim_whitespace=True, skiprows=1, header=None)
        rotated_montage = mne.channels.make_dig_montage(
            ch_pos={name: R.from_euler('z', 90, degrees=True).apply(coord)  # Rotate 90 degrees
                    for name, coord in zip(xyz_data[3].values, xyz_data[[0, 1, 2]].values)},
            coord_frame="head"
        )
        # Set the rotated montage to the Raw object
        raw.set_montage(rotated_montage)
        return raw
    
    def epoching(self, raw, stim = "unicity", tmin=-0.2, tmax=0.8, baseline=(None, 0) ):
        events = mne.find_events(raw)
        if stim == "unicity": 
            epochs = mne.Epochs(raw, events, event_id={'High':1, 'Low':2}, tmin=tmin, tmax=tmax, baseline=baseline,   preload=True)
        return epochs

    def bridged_channels(self,instant,   lm_cutoff = 5, epoch_threshold=0.5):
    
        bridged_idx, ed_matrix  = compute_bridged_electrodes( instant, lm_cutoff = lm_cutoff, epoch_threshold= epoch_threshold)

        bridged_channels = list(set([channel for pair in bridged_idx for channel in pair]))
        bridged_ch_names = [self.raw.ch_names[i] for i  in bridged_channels]
        self.raw.info['bads'] += bridged_ch_names

        return bridged_idx, ed_matrix, bridged_ch_names
    
    def Bad_segments(self, raw, diff_stim_threshold=11):
        import pandas as pd
        df = self.df
        events, sfreq = mne.find_events(raw), raw.info['sfreq']
        # Create response annotations
        response_annotations = [
            ((events[i, 0] / sfreq) + (row['RT'] / 1000), 0.0, 'response')
            for i, row in df.iterrows() if not pd.isna(row['RT'])
        ]

        # Create break annotations
        threshold = diff_stim_threshold
        stims = events[:, 0] / sfreq
        diff_stim = stims[1:] - stims[:-1]
        prior_stim = np.where(diff_stim > threshold)[0]
        next_stim = prior_stim + 1
        break_annotations = [
            (stims[prior_stim[i]] + 1, (stims[next_stim[i]] - stims[prior_stim[i]] - 1.5), 'BAD_breaks')
            for i in range(len(prior_stim))
        ]
        print(f'numnber of breaks found with threshold {threshold}: {len(break_annotations)}')
        # Add beginning and end annotations
        beginning_end_annotations = [
            (0.0, stims[0] - 1, 'BAD_beginning'),
            (stims[-1] + 2, raw.times[-1] - (stims[-1] + 2), 'BAD_end')
        ]
        # Combine all annotations
        all_annotations = response_annotations + break_annotations + beginning_end_annotations

        # Convert to MNE Annotations and set them on raw
        onsets, durations, descriptions = zip(*all_annotations)
        raw = raw.set_annotations(mne.Annotations(onset=onsets, duration=durations, description=descriptions))

        return raw
    
    
    def get_all_events_times(self, events, path_beh_task=None, path_beh_subject=None):


        # load data from exel file 
        if path_beh_task is None or path_beh_subject is None:
            path_beh_task = os.path.join(data_path, 'BehavioralResultsDefinition24ss_11_11_2017_RF_2023.xlsx')
            path_beh_subject =  os.path.join(data_path, 'ClasseurCompDef_Lifespan_V8_Avril2019_Outliers.xlsx')
        
        # Load Excel sheets
        behavior_tasks = pd.read_excel(path_beh_task, sheet_name='ItemsBalancés')
        behavior_subjets = pd.read_excel(path_beh_subject, sheet_name='Data')
        subject_id = self.id
        # Filter data for current subject
        subject_key = f'S{subject_id}'
        subject_data = behavior_subjets[behavior_subjets['Sujet'] == subject_key]
        subject_data = subject_data.dropna(subset=['Cible'])
        subject_data = subject_data[subject_data['Ordre'] <= 108]

        # Clean and normalize stimulus duration
        behavior_tasks = behavior_tasks.dropna(subset=['ortho'])
        behavior_tasks['ortho'] = behavior_tasks['ortho'].apply(unidecode)

        sfreq = self.raw.info['sfreq']
        results = {
            'Trial': [],
            'defOnset': [],
            'SecWordOnset': [],
            'LWOnset': [],
            'Respons': [],
            'freqs': [],
            'word': []
        }

        for trial in range(1, 109):
            # Get stimulus word
            word = subject_data[subject_data['Ordre'] == trial]['Cible'].values[0]

            # Lookup in duration data
            stim_info = behavior_tasks[behavior_tasks['ortho'] == word]
            
            total_duration_ms = stim_info['DureeTot_second'].values[0] * 1000
            pu_ms = stim_info['PU_second'].values[0] * 1000
            def2_onset_ms = stim_info['Def2_Audio_Onset'].values[0] * 1000
            lw_onset_ms = stim_info['LW_Onset'].values[0] * 1000
            # Get frequency
            freq = stim_info['Freq_Manulex'].values[0]
    

            # Get response time
            rt_corrPU_ms = subject_data[subject_data['Ordre'] == trial]['RT_Correct_CorrPU'].values[0]

            # Calculate onsets
            event_time_ms = (events[trial - 1][0] / sfreq) * 1000
            onset_def = event_time_ms - pu_ms
            onset_sec_word = onset_def + def2_onset_ms
            onset_lw = onset_def + lw_onset_ms
            response_time = event_time_ms + rt_corrPU_ms
            # return onset_def, onset_sec_word, onset_lw, response_time, event_time_ms, 
            # Append to results
            results['Trial'].append(trial)
            results['defOnset'].append(onset_def / 1000.0)
            results['SecWordOnset'].append(onset_sec_word / 1000.0)
            results['LWOnset'].append(onset_lw / 1000.0)
            results['Respons'].append(response_time / 1000.0)
            results['freqs'].append(freq)
            results['word'].append(word)

        return pd.DataFrame(results)
    
    def segment_stimRt(self, raw, all_events, bad_trials, prestim_duration = 0):

        all_trials = []
        for idx, row in all_events.iterrows():

            Tnum = row['Trial']
            if Tnum in bad_trials:
                continue
            start = row['defOnset'] - prestim_duration
            end = row['Respons'] - 0.1
            duration = end - start

            # Copy and crop raw data
            data = raw.copy().crop(start, end)
            
            # Create annotation
            onset_in_cropped = 0  # onset relative to start of cropped data
            annotation = mne.Annotations(onset=[onset_in_cropped],
                                        duration=[duration],
                                        description=[f'S{self.id}_Trial_{Tnum}'])
            
            # Set annotation to this segment
            data.set_annotations(annotation)

            all_trials.append(data)
        # all_trials_returen = all_trials
        # new_raw = mne.concatenate_raws(all_trials)

        return all_trials
    
   

def interpolate_HANoise(all_trials, detected_noise):
    all_trials_interpolated = []
    if len(all_trials) != detected_noise.shape[1]:
        raise ValueError("Length of all_trials and detected_noise must match.")
    detected_noiseT = detected_noise.copy().transpose()
    for i in range(len(all_trials)):
        raw_epoch = all_trials[i]
        
        bad_idx = np.where(detected_noiseT[i])[0]
        n_total_bads = len(bad_idx) + len(raw_epoch.info['bads'])
        if n_total_bads > 30:
            continue
        elif n_total_bads > 0:
            raw_epoch.info['bads'] += [raw_epoch.ch_names[idx] for idx in bad_idx]
            raw_epoch.interpolate_bads()
            raw_epoch.info['bads'] = []  # Clear bads after interpolation
            all_trials_interpolated.append(raw_epoch)
        elif n_total_bads == 0:
            all_trials_interpolated.append(raw_epoch)
    return all_trials_interpolated





class behaviorAnalysis:
    def __init__(self, id):
        self.id = str(id)
    
    def load_subject_data(self, path_beh_subject):

        behavior_subjets = pd.read_excel(path_beh_subject, sheet_name='Data')
        # Filter data for current subject
        subject_key = f'S{self.id}'
        subject_data = behavior_subjets[behavior_subjets['Sujet'] == subject_key]
        # remove the rows that Ordre is more than 108
        subject_data = subject_data[subject_data['Ordre'] <= 108]

        return subject_data
    



from mne_icalabel import label_components

def get_noisyICs(ic_labels, threshold=0.7, noise_type= 'all'):
    if noise_type == 'all':
        noisy_components = []
        for i, label in enumerate(ic_labels['labels']):
            prob = ic_labels['y_pred_proba'][i]
            if not label == 'brain':
                if not label == 'other':
                    if prob > threshold:
                        noisy_components.append(i)
    elif noise_type == 'blinks':
        noisy_components = []
        for i, label in enumerate(ic_labels['labels']):
            prob = ic_labels['y_pred_proba'][i]
            if label == 'eye blink':
                if prob > threshold:
                    noisy_components.append(i)
    return noisy_components

import pickle
def compute_bridged(ids, overwrite = False):

    lm_cutoff = 5 
    bridged_channels_dict = {}
    path_bridge = os.path.join(data_path,'bridged_channels_analysis.pkl')
    if os.path.exists(path_bridge) and not overwrite:
        print("Bridged channels analysis already exists. Use overwrite=True to recompute.")

    for i, id in enumerate(ids):

        # Preprocess the data for each subject
        print(f"Processing subject {id} with LM cutoff {lm_cutoff}...")
        subject = preprocess(id)
        epochs = subject.epoching(tmin=-0.2, tmax=0.8, baseline=(None, 0))
        bridged_idx, ed_matrix, bridged_ch_names = subject.bridged_channels(
            epochs, 
            lm_cutoff=lm_cutoff, 
            epoch_threshold=0.5
        )
        
        # Save bridged channels information in the dictionary
        bridged_channels_dict[id] = {
            "bridged_idx": bridged_idx,
            "bridged_ch_names": bridged_ch_names
        }
        print(f"Found {len(bridged_ch_names)} bridged channels for subject {id}")

    # Save the bridged channels information to a file
    with open(path_bridge, 'wb') as f:
        pickle.dump(bridged_channels_dict, f)
    print(f"Bridged channels analysis saved to {path_bridge}")
    return


def plot_HA_autoreject(all_bads, sub, threshold = 0.2):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    detection = all_bads['detection']  # shape (n_channels, n_trials) with True/False
    n_bad_ch = len(all_bads['channel_names'])
    # === PARAMETERS ===
    threshold_channel_pct = 0.4
    threshold_trials = threshold * 128

    # === COMPUTE ===
    noisy_per_trial = detection.sum(axis=0) + n_bad_ch # trials
    noisy_per_channel = detection.sum(axis=1)  # channels

    n_trials = detection.shape[1]
    n_channels = detection.shape[0]

    threshold_channels = int(n_trials * threshold_channel_pct)

    bad_trials = np.where(noisy_per_trial > threshold_trials)[0]
    bad_channels = np.where(noisy_per_channel > threshold_channels)[0]

    print(f"Trials with >{threshold_trials} noisy channels: {bad_trials.tolist()}")
    print(f"Channels noisy in >{threshold_channel_pct*100:.0f}% of trials: {bad_channels.tolist()}")

    # === PLOT ===
    plt.figure(figsize=(16, 8))
    ax = sns.heatmap(detection, cmap="Blues", cbar=True,
                    xticklabels=1, yticklabels=1)

    # Show all x and y ticks
    ax.set_xticks(np.arange(n_trials) + 0.5)
    ax.set_xticklabels(np.arange(n_trials))
    ax.set_yticks(np.arange(n_channels) + 0.5)
    ax.set_yticklabels(np.arange(n_channels))
    ax.invert_yaxis()

    # Highlight whole columns for bad trials
    for t in bad_trials:
        ax.add_patch(
            plt.Rectangle(
                (t, 0),            # (x, y) lower left corner
                1, n_channels,     # width, height
                fill=True,
                color='red',
                alpha=0.3,
                lw=0
            )
        )

    # Highlight whole rows for bad channels
    for c in bad_channels:
        ax.add_patch(
            plt.Rectangle(
                (0, c),            # (x, y) lower left corner
                n_trials, 1,       # width, height
                fill=True,
                color='red',
                alpha=0.3,
                lw=0
            )
        )

    plt.xlabel('Trial')
    plt.ylabel('Channel')
    plt.title(f' Subject {sub} Noisy Channel Detection Matrix (True=Noisy)\nRed columns = bad trials, Red rows = bad channels')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__),'plots','HA_noise', f'HA_autoreject_{sub}.png'), dpi=300)

    plt.show()


