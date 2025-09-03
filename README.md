This repository will hold the files and functions to compute the guidelines of the DeEsco trial
# DeEsco Trial Guidelines

The guidelines for the DeEsco trial have been produced using version **1.0.0.clin-trial** of the `lymph` repository. To ensure reproducibility of the results, it is essential to install the correct version of the `lymph` repository.

## Installation Instructions

You can install the required version of the `lymph` repository using one of the following methods:

1. Install directly from the GitHub repository:
    ```bash
    pip install git+https://github.com/rmnldwg/lymph.git@1.0.0.clin-trial
    ```

2. Install additional dependencies:
    ```bash
    pip install -r requirements.txt
    ```

Ensure that the correct version is installed before proceeding with the trial.

## Repository Contents

This repository contains the following key files:

- **`midline_and_central_calculator.ipynb`**: Main Jupyter notebook with all analysis steps
- **`sparing_scripts.py`**: Core functions for risk calculation and treatment optimization
- **`requirements.txt`**: Python dependencies
- **`data/`**: Contains input datasets and MCMC samples
  - `cleanedUSZ.csv`: Clinical dataset
  - `samples_midline_trial.hdf5`: MCMC parameter samples (2160 samples total)
- **`tables/`**: Generated treatment decision tables
  - `lymph_1_midline_full_table_new_code.csv`: Treatment tables for non-central tumors
  - `lymph_1_midline_full_table_central_new_code.csv`: Treatment tables for central tumors

## Usage Instructions

To reproduce the DeEsco trial treatment guidelines:

1. Install the required dependencies as outlined in the Installation Instructions section.
2. Ensure you have the required data files in the `data/` directory:
   - `samples_midline_trial.hdf5` (MCMC samples)
3. Open `midline_and_central_calculator.ipynb` in a Jupyter environment.
4. Execute the cells sequentially to reproduce the results.
5. The final treatment tables will be generated and saved to the `tables/` directory.

## Data Files

The `data` directory contains the necessary datasets for the analysis. The main files are:

- **`cleanedUSZ.csv`**: Clinical dataset from USZ with patient diagnostic information
- **`samples_midline_trial.hdf5`**: MCMC parameter samples (2160 total samples) used for uncertainty quantification

## Treatment List Generation

The treatment lists are generated through a combination of risk evaluation and decision-making algorithms implemented in the notebooks. Here is a detailed overview of the process with specific parameters:

### Key Parameters Used

- **Sample Size**: 216 samples are used for uncertainty quantification, selected with evenly spaced sampling (step_size = 10) from the original MCMC chain
- **Risk Threshold**: 10% (0.10) maximum allowable risk for lymph node level sparing decisions
- **Confidence Interval**: 95% credibility intervals are calculated and the **upper bound** is used for threshold comparison
- **Diagnostic Sensitivity**: Treatment diagnosis modality uses 81% sensitivity (0.81) with 100% specificity

### Process Overview

1. **Model Setup**: The lymphatic model is initialized using lymph version **1.0.0.clin-trial** with predefined graphs and diagnostic modalities. This setup includes parameters for early and late diagnosis probabilities, with early diagnosis probability set to 30% (p=0.3).

2. **Sample Preparation**: From the original MCMC samples, exactly **216 samples** are selected using evenly spaced sampling with a step size of 10 to ensure proper coverage while maintaining computational efficiency.

3. **Risk Sampling**: Using the `risk_sampled` function, risks are sampled for different diagnostic scenarios (T-stage, midline extension, and LNL involvement patterns). Each of the 216 parameter samples generates a full risk matrix, creating a distribution of risk estimates.

4. **Threshold-Based Decision**: The `levels_to_spare` function determines which lymph node levels (LNLs) to treat or spare using the following algorithm:
   - LNLs are ranked by individual risk contribution (lowest to highest)
   - LNLs are iteratively excluded from treatment starting with the lowest risk
   - The algorithm stops when the **95% confidence interval upper bound** of the total remaining risk exceeds the **10% threshold**
   - This ensures that there is 95% confidence that the actual risk remains below 10%

5. **Combination Analysis**: All possible diagnostic combinations (2^14 = 16,384 combinations for non-central tumors, 2^13 = 8,192 for central tumors) are analyzed to compute risks and generate treatment recommendations. Each combination includes:
   - T-stage (early/late)
   - Midline extension (True/False, for non-central tumors only)
   - LNL involvement pattern (12 boolean values for 6 ipsilateral + 6 contralateral LNLs)

6. **Statistical Rigor**: For each combination, the algorithm:
   - Calculates the mean risk across all 216 samples
   - Computes the 95% credibility interval (2.5th to 97.5th percentiles)
   - Uses the upper bound of this interval for conservative decision-making
   - Ensures treatment decisions account for parameter uncertainty

7. **Export**: The results, including treated and spared LNLs with their associated risks and confidence intervals, are exported as comprehensive tables for clinical use and further analysis.

## Lymph Node Levels (LNLs)

The analysis considers 6 lymph node levels on each side:
- **I, II, III, IV, V, VII** for both ipsilateral and contralateral sides
- Total of 12 LNLs analyzed per patient case
- Central tumors are analyzed without midline extension parameter

## Output Tables

The generated treatment tables contain the following information for each diagnostic combination:
- **T-stage**: Early or late
- **Midline Extension**: True/False (non-central tumors only)
- **LNL Involvement**: Diagnosed positive LNLs for each side
- **Treated LNLs**: Recommended LNLs to include in treatment volume
- **Risk Estimates**: Mean risk and 95% confidence intervals
- **Spared LNLs**: Top 3 lowest-risk LNLs that can be safely omitted