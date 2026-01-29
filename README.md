# Quickstart User Guide 

## Dataset Generation

This repository contains tools for generating training datasets from the LLC4320 ocean model.

To run or understand the dataset generation workflow, **start with the Jupyter notebook**:

`notebooks/running_generate_front_training_data_script.ipynb`

The notebook documents:
- required configuration files
- how temporal and spatial sampling are defined
- how to launch a dataset generation run

## Dataset Access

To explore and inspect generated datasets, see the Jupyter notebook:

`notebooks/access_dbof_dataset.ipynb`

This notebook demonstrates:
- how to connect to the remote Zarr store containing image patches
- how to read individual samples and batches from the dataset
- how to load and inspect the associated Parquet metadata
- how to link metadata fields (location, time, gradients) back to image samples
- basic visualization examples for checking the data

The notebook is intended as a reference for integrating the dataset into downstream training workflows.

## Build the Project for Downstream Data Access

Users who want to **read or train on datasets produced by this pipeline** can install the package as a normal Python dependency.

Install directly from GitHub:
```bash
pip install "git+https://github.com/Sea-Meets-the-Stars/llc4320-native-grid-preprocessing.git"
```
Optional dependency groups can be installed as needed (for example, plotting utilities used in the notebooks):
```bash
pip install "dbof-in-native-grid[plotting] @ git+https://github.com/Sea-Meets-the-Stars/llc4320-native-grid-preprocessing.git"
```
A running list of optional dependencies are:
- plotting

User can also clone locally and install via 
```
git clone https://github.com/Sea-Meets-the-Stars/llc4320-native-grid-preprocessing.git
cd llc4320-native-grid-preprocessing
pip install -e .
```

After installation, dataset readers and utilities can be imported in downstream code, for example:
```
from dbof.dataset_creation.zarr_dataset import ZarrDatasetReader, ZarrTorchDataset
```

# An explanation of the dataset generation
This section is under construction. It will point to the relevant docs in docs/ that explain the code. 