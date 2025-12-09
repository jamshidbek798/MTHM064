# MTHM064 Dissertation Project
## Supervised Autoencoders for Functional Classification of Single-Cell Genetic Perturbations

This repository contains the code and results for the **MTHM064 Masters' Dissertation Project**. The study investigates the use of Supervised Autoencoders to map high-dimensional single-cell gene expression data into biologically interpretable latent spaces, applied to both labelled scSNV-seq data and unlabelled CRISPR screens.

### 📂 Repository Contents

The project is organized into two main directories:

* **`sc-SNV sequential/`**: Validation study on the labelled JAK1 dataset. Contains the `main.ipynb` notebook for training and visualization.
* **`CRISPR/`**: Application to unlabelled CRISPRi and CRISPRn datasets. Contains the core Python modules (`optimisation/`), training notebooks, and result logs.

### 🛠️ Usage

1.  **Installation:**
    ```bash
    pip install -r CRISPR/requirements.txt
    ```

2.  **Running the Analysis:**
    * For the validation study, run `sc-SNV sequential/main.ipynb`.
    * For CRISPR datasets, run the specific notebooks located in `CRISPR/` (e.g., `CRISPRi_targeted.ipynb`).

