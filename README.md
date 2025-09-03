This repository will hold the files and functions to compute the guidelines of the DeEsco trial
# DeEsco Trial Guidelines

The guidelines for the DeEsco trial have been produced using version **1.0.0.clin-trial** of the `lymph` repository. To ensure reproducibility of the results, it is essential to install the correct version of the `lymph` repository.

## Installation Instructions

You can install the required version of the `lymph` repository using one of the following methods:

1. Install directly from the GitHub repository:
    ```bash
    pip install git+https://github.com/rmnldwg/lymph.git@1.0.0.clin-trial
    ```
Ensure that the correct version is installed before proceeding with the trial.

# Additional Information

## Notebooks Overview

This repository contains two Jupyter Notebooks that are integral to the DeEsco trial analysis:

1. **risk_evalutor_trial.ipynb**: This notebook focuses on trial-specific risk evaluation, including model setup, classical analysis, and combination analysis. It also generates risk tables and full treatment tables for the trial. Note that to get the exact original results, you need to take 216.



## Usage Instructions

To run the notebooks, ensure the following:

1. Install the required dependencies as outlined in the Installation Instructions section.
2. Open the notebooks in a Jupyter environment.
3. Execute the cells sequentially to reproduce the results.

## Data Files

The `data` directory contains the necessary datasets for the analysis. Ensure that the paths to the data files are correctly set in the notebooks before execution.

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
   - LNLs are iteratively included in treatment starting with the highest risk
   - The algorithm stops when the **95% confidence interval upper bound** of the total risk is lower than the **10% threshold**
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