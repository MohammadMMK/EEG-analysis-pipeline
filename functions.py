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
        self.txt_path = os.path.join(data_path, [f for f in os.listdir(data_path) if self.id in f and f.endswith('.txt')][0])
        self.raw = self.load_data()
        self.df = pd.read_csv(self.txt_path, delimiter='\t', header=None)
        self.df.columns = ['outcome', 'RT', 'unicityDistance', 'earlyVSlate']

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
    
    def epoching(self,raw, stim = "unicity", tmin=-0.2, tmax=0.8, baseline=(None, 0) ):
        events = mne.find_events(raw)
        if stim == "unicity": 
            epochs = mne.Epochs(raw, events, event_id={'EasyWord':1, 'HardWord':2}, tmin=tmin, tmax=tmax, baseline=baseline,   preload=True)
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


    def get_all_events_times(self, subject_id, events, 
                            path_beh_task=None, path_beh_subject=None):


        # load data from exel file 
        if path_beh_task is None or path_beh_subject is None:
            path_beh_task = os.path.join(data_path, 'BehavioralResultsDefinition24ss_11_11_2017_RF_2023.xlsx')
            path_beh_subject =  os.path.join(data_path, 'ClasseurCompDef_Lifespan_V8_Avril2019_Outliers.xlsx')
        
        # Load Excel sheets
        behavior_tasks = pd.read_excel(path_beh_task, sheet_name='ItemsBalancés')
        behavior_subjets = pd.read_excel(path_beh_subject, sheet_name='Data')

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
            'Respons': []
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

            # Get response time
            rt_corrPU_ms = subject_data[subject_data['Ordre'] == trial]['RT_Correct_CorrPU'].values[0]

            # Calculate onsets
            event_time_ms = (events[trial - 1][0] / sfreq) * 1000
            onset_def = event_time_ms - pu_ms
            onset_sec_word = onset_def + def2_onset_ms
            onset_lw = onset_def + lw_onset_ms
            response_time = event_time_ms + rt_corrPU_ms

            # Append to results
            results['Trial'].append(trial)
            results['defOnset'].append(onset_def / 1000.0)
            results['SecWordOnset'].append(onset_sec_word / 1000.0)
            results['LWOnset'].append(onset_lw / 1000.0)
            results['Respons'].append(response_time / 1000.0)

        return pd.DataFrame(results)
    
    def segment_stimRt(self, raw, all_events, bad_trials):

        all_trials = []
        for idx, row in all_events.iterrows():

            Tnum = row['Trial']
            if Tnum in bad_trials:
                continue
            start = row['defOnset'] 
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
        new_raw = mne.concatenate_raws(all_trials)

        return new_raw




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

def get_noisyICs(prep_data, ica, threshold=0.7, noise_type= 'all'):
    if noise_type == 'all':
        ic_labels = label_components(prep_data, ica, method="iclabel")
        noisy_components = []
        for i, label in enumerate(ic_labels['labels']):
            prob = ic_labels['y_pred_proba'][i]
            if not label == 'brain' or label == 'other':
                if prob > threshold:
                    noisy_components.append(i)
    elif noise_type == 'blinks':
        ic_labels = label_components(prep_data, ica, method="iclabel")
        noisy_components = []
        for i, label in enumerate(ic_labels['labels']):
            prob = ic_labels['y_pred_proba'][i]
            if label == 'eye blink':
                if prob > threshold:
                    noisy_components.append(i)
    return noisy_components, ic_labels


# bothBad = [ch for ch in bads_channel if ch in bridged_channels['bridged_ch_names']]
# bothbad_indx = [new_raw.ch_names.index(ch) for ch in bads_channel]
# new_badIndx = []
# for ch1, ch2 in bridged_channels['bridged_idx']:
#     if ch1 in bothbad_indx or ch2 in bothbad_indx:
#         new_badIndx.append(ch1)
#         new_badIndx.append(ch2)
#         # remove this tuple from the list
#         bridged_channels['bridged_idx'].remove((ch1, ch2))
# new_badIndx = list(set(new_badIndx))
# new_bads = [new_raw.ch_names[ch] for ch in new_badIndx]