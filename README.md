This repository is under active developement and the documentation is not yet complete.

# EEG Preprocessing Pipeline

A comprehensive Python pipeline for preprocessing EEG data using MNE-Python and other neuroimaging libraries. This pipeline includes tools for loading, cleaning, filtering, and analyzing EEG data from BDF files.

## Features

- 🧠 **EEG Data Processing**: Load and preprocess BDF files using MNE-Python
- 🔧 **Automated Cleaning**: Remove artifacts, bad channels, and apply filters
- 📊 **Analysis Tools**: Statistical analysis and visualization capabilities
- 🎯 **ICA Processing**: Independent Component Analysis for artifact removal
- 📈 **Visualization**: Generate plots and reports for data quality assessment

## Prerequisites

- Python 3.10 or higher
- Git (for cloning the repository)
- Either Conda or pip package manager

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/mohammadmmk/preprocessing_pipline.git
cd preprocessing_pipline
```

### Step 2: Set Up Environment

You have two options for setting up the environment:

#### Option A: Using Conda (Recommended)

1. **Install Conda** if you haven't already:
   - Download and install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. **Create the environment from the environment.yml file**:
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment**:
   ```bash
   conda activate BCLenv
   ```

#### Option B: Using pip

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
preprocessing_pipline/
├── config.py              # Configuration settings and paths
├── functions.py            # Core preprocessing functions
├── Pipeline.py             # Main pipeline implementation
├── preprocess.py          # Main preprocessing script
├── environment.yml        # Conda environment file
├── requirements.txt       # pip requirements file
├── README.md             # This file
├── Data/                 # EEG data files (BDF format)
│   ├── *.bdf            # Raw EEG data files
│   ├── *.fif            # Processed MNE files
│   └── *.pkl            # Pickle files with metadata
├── plots/               # Generated plots and visualizations
├── notebooks/           # Jupyter notebooks for analysis
└── drafts/             # Draft scripts and experimental code
```

## Usage

### Basic Usage

1. **Ensure your environment is activated**:
   ```bash
   conda activate BCLenv  # if using conda
   # or
   source venv/bin/activate  # if using pip on macOS/Linux
   # or
   venv\Scripts\activate  # if using pip on Windows
   ```

2. **Run the preprocessing pipeline**:
   ```bash
   python preprocess.py
   ```

   Or use the VS Code task:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Tasks: Run Task"
   - Select "Run Preprocessing Script"

### Configuration

Edit `config.py` to modify:
- **Data paths**: Update `data_path` to point to your EEG data directory
- **Subject IDs**: Modify `old_adults_ids` and `young_adults_ids` arrays
- **Processing parameters**: Adjust filtering and preprocessing settings

### Working with Your Data

1. **Place your BDF files** in the `Data/` directory
2. **Update subject IDs** in `config.py` to match your data
3. **Modify preprocessing parameters** in `Pipeline.py` as needed
4. **Run the pipeline** using the commands above



## Data Requirements

The pipeline expects:
- **EEG data**: BDF format files in the `Data/` directory
- **Bad channels/trials**: Pickle files with manual artifact detection
- **Bridged channels**: Analysis results in pickle format


## Common Issues

1. **Import errors**: Make sure you've activated the correct environment
2. **Missing data files**: Check that your BDF files are in the `Data/` directory
3. **Permission errors**: Ensure you have read/write access to the project directory


## Dependencies

### Core Libraries
- **MNE-Python**: EEG/MEG data processing
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **SciPy**: Scientific computing
- **Matplotlib/Seaborn**: Visualization

### Specialized Tools
- **autoreject**: Automated artifact rejection
- **pyprep**: Preprocessing utilities
- **asrpy**: Artifact Subspace Reconstruction
- **mne-icalabel**: ICA component labeling

