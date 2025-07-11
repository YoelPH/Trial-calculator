This repository will hold the files and functions to compute the guidelines of the DeEsco trial
# DeEsco Trial Guidelines

The guidelines for the DeEsco trial have been produced using version **0.4.3** of the `lymph` repository. To ensure reproducibility of the results, it is essential to install the correct version of the `lymph` repository.

## Installation Instructions

You can install the required version of the `lymph` repository using one of the following methods:

1. Install directly from the GitHub repository:
    ```bash
    pip install git+https://github.com/rmnldwg/lymph.git@0.4.3
    ```

2. Install the specific version from PyPI:
    ```bash
    pip install lymph-model==0.4.3
    ```

3. Install all dependencies from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

Ensure that the correct version is installed before proceeding with the trial.

# Additional Information

## Notebooks Overview

This repository contains two Jupyter Notebooks that are integral to the DeEsco trial analysis:

1. **risk_evalutor_trial.ipynb**: This notebook focuses on trial-specific risk evaluation, including model setup, classical analysis, and combination analysis. It also generates risk tables and full treatment tables for the trial. Note that to get the exact original results, you need to take 203 samples which are spaced by 89.

2. **risk_evalutor_central.ipynb**: This notebook is centered on central risk evaluation. It includes similar analyses as the trial notebook but is tailored for central data processing and evaluation.

Both notebooks are structured with markdown and Python cells, ensuring clarity and reproducibility of the analysis.

## Usage Instructions

To run the notebooks, ensure the following:

1. Install the required dependencies as outlined in the Installation Instructions section.
2. Open the notebooks in a Jupyter environment.
3. Execute the cells sequentially to reproduce the results.

## Data Files

The `data` directory contains the necessary datasets for the analysis. Ensure that the paths to the data files are correctly set in the notebooks before execution.

## Treatment List Generation

The treatment lists are generated through a combination of risk evaluation and decision-making algorithms implemented in the notebooks. Here is a brief overview of the process:

1. **Model Setup**: The lymphatic model is initialized with predefined graphs and diagnostic modalities. This setup includes parameters for early and late diagnosis probabilities.

2. **Risk Sampling**: Using the `risk_sampled` function, risks are sampled for different diagnostic scenarios. This involves several sample calculations to estimate the distribution of risks.

3. **Threshold-Based Decision**: The `levels_to_spare` function is used to determine which lymph node levels (LNLs) to treat or spare. This decision is based on a predefined risk threshold (e.g., 10%).

4. **Combination Analysis**: All possible diagnostic combinations are analyzed to compute risks and generate treatment recommendations.

5. **Export**: The results, including treated and spared LNLs, are exported as tables for further analysis or reporting.