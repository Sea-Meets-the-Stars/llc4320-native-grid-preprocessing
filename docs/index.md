# DBOF in Native Grid

Documentation for the LLC4320 native grid preprocessing pipeline for detecting fronts in the ocean.

## Overview

This project provides tools for:
- Accessing raw LLC4320 model output data
- Preprocessing LLC data for machine learning
- Building global maps and single tiles on the native grid
- Weighted sampling strategies
- Halo masking techniques

## Contents

```{toctree}
:maxdepth: 2
:caption: Documentation

Accessing_Raw_LLC4320_Data
Data_Organization
Global_Maps.md
Preprocess_LLC_Data
Global_Maps
Tiles
Weighted_Sampling
Sampling_With_GradB2
Halo_Masking
Tasks_TODO
```

```{toctree}
:maxdepth: 2
:caption: Infrastructure

nautilus/s3_DBOF
```

## Installation

```bash
pip install -e .
```

## Authors

- J. Xavier Prochaska (jxp@ucsc.edu)
- P. Cornillon
- J. Tallman (jttallman@ucdavis.edu)
- L. Hoffman (lhoffma2@ucsc.edu)

## License

BSD License
