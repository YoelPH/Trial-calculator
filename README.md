# Trial Calculator

This repository contains analysis code and outputs for treatment table generation using `lymph` models.

## Important Version Note

This repository is **not** tied to `1.0.0.clin-trial` anymore.

Use the **newest available `lymph-model`** release.
At the time of writing, this workflow was tested with `lymph-model` **1.3.8**.

## Installation

From the repository root:

```bash
pip install --upgrade lymph-model
pip install -r requirements.txt
```

## Current Repository Contents

### Main analysis notebook

- `lymph_models_analysis_simple.ipynb`: Main notebook for model setup, posterior sampling analysis, and treatment table generation.

### Python helper functions

- `sparing_scripts.py`: Core helper functions for
  - posterior risk sampling,
  - confidence interval calculation,
  - unilateral and bilateral level-sparing decisions,
  - combination-level treatment analysis.

### Data currently present

In `data/`:

- `cleanedUSZ.csv`
- `oropharynx_evaluation_dataset_clin_trial_updated_2025.csv`

### Existing output artifacts currently present

In `tables/`:

- `lymph_1_midline_full_table_new_code.csv`
- `lymph_1_midline_full_table_central_new_code.csv`

In the repository root:

- `II_III.csv` (CSV output from notebook analysis)
- `II_II` and `II_II_midline` (binary `emcee` HDF backend files used by the notebook)

## How To Run

1. Install dependencies (see Installation).
2. Ensure the files listed above exist in `data/`.
3. Open `lymph_models_analysis_simple.ipynb`.
4. Run notebook cells in order.

## Notes

- Some older README references (for example `midline_and_central_calculator.ipynb` and `samples_midline_trial.hdf5`) are no longer valid for the current repository state.
- If you want clean reproducibility, keep outputs under `tables/` and/or version newly generated CSV files with clear names.