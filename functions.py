import os 
from config import data_path
import mne
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
from mne.preprocessing import compute_bridged_electrodes

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

        df = pd.read_csv(self.txt_path, delimiter='\t', header=None)
        df.columns = ['outcome', 'RT', 'unicityDistance', 'earlyVSlate']
        events = mne.find_events(raw)
        epochs = mne.Epochs(raw, events, event_id={'EasyWord':1, 'HardWord':2}, tmin=tmin, tmax=tmax, baseline=baseline, metadata=df,  preload=True)
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

        



