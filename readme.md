# Machine Learning Course 2025

This repository provides Jupyter notebooks and datasets for a basic machine learning curriculum.

## Installation

1. Install [conda](https://docs.conda.io/en/latest/miniconda.html) for Python 3.11 or newer.
2. Create the course environment with the provided `environment.yml`:
   ```bash
   conda env create -f environment.yml
   ```
3. Activate the new environment:
   ```bash
   conda activate machine-learning-course-2025
   ```
4. Launch JupyterLab or the classic Jupyter Notebook:
   ```bash
   jupyter lab
   ```

All required packages (numpy, pandas, scikit-learn, matplotlib, seaborn, jupyterlab) will be installed automatically as part of the conda environment.

## Contents

- `data/` – sample datasets used in the notebooks.
- `utils/` – helper plotting functions.
- `notebooks/` – step‑by‑step lessons executed with output saved.
- Key notebooks:
  - `00_Welcome.ipynb` – introduction and course overview.
  - `01_Data_Preparation.ipynb` – handling missing values and tidying data.
  - `02_Data_Preprocessing.ipynb` – normalization, cross validation, regularization and dimensional reduction.
  - `03_Regression_Models.ipynb` – theory and examples of linear regression.
  - `04_Classification_Models.ipynb` – logistic regression and evaluation metrics.
  - `05_Unsupervised_Learning.ipynb` – clustering and dimensionality reduction.
  - `06_Model_Evaluation.ipynb` – validation techniques and error metrics.
  - `07_Intro_Neural_Networks.ipynb` – basics of neural networks.
  - `08_Project_Template.ipynb` – template for course projects.
- `environment.yml` – list of dependencies for reproducible setup.

After activating the environment you can open any notebook from the `notebooks/` directory to begin exploring the examples.
