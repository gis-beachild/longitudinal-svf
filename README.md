# Registration‐based analysis of early brain development using longitudinal MRI

This repository contains code and documentation for performing the experimentation and analysis described in the paper:
Registration‐based analysis of early brain development using longitudinal MRI

 # Installation
To install the necessary dependencies, please follow these steps:
1. Clone the repository:
2. Install the required packages using pip:
```pip install -r requirements.txt ```
3. Install local registration-svf package:
```pip install . ```
4. Configure the data configuration file in the `data/` directory to point to your dataset and its relative information.
5. Launch the experimentation (training/predict part) using :
```bash launch_experimentation.sh -d DATASET ...```
6. Compute the metrics from the results folder using:
```bash result.sh -d DATASET ```