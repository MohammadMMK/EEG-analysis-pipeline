import mne
import numpy as np

def load_data(file_path):
    """Load EEG data from a file."""
    raw = mne.io.read_raw(file_path, preload=True)
    return raw

def clean_data(raw):
    """Clean EEG data by removing artifacts."""
    raw.filter(1., 40., fir_design='firwin')
    return raw

def analyze_data(raw):
    """Analyze EEG data."""
    print("Data info:", raw.info)
    print("Data shape:", raw.get_data().shape)

if __name__ == "__main__":
    # Example usage
    file_path = "data/sample_eeg.fif"  # Replace with your EEG file path
    raw_data = load_data(file_path)
    cleaned_data = clean_data(raw_data)
    analyze_data(cleaned_data)
