<div align="center">

# **QAIEDGE MetaboLocus**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A State-of-the-Art Machine Learning Framework for Predicting the Biofluid Localization of Human Metabolites from Chemical Structure.**

</div>

---

**MetaboLocus** is a pioneering computational framework that addresses a grand challenge in metabolomics: determining the biological location of metabolites. Inspired by the transformative impact of AlphaFold in proteomics, MetaboLocus leverages a sophisticated, multi-expert stacked ensemble model to predict the distribution of any metabolite across six major human biofluids (Blood, Urine, Saliva, CSF, Feces, Sweat) using only its chemical structure as input.

This tool is the culmination of research that definitively establishes the intrinsic link between molecular architecture and biological fate. By providing high-throughput, high-accuracy predictions, MetaboLocus obviates the need for prohibitively expensive and time-consuming experimental workflows. Our work has already annotated **171,456** previously uncharacterized human metabolites, a task that would have required an estimated **$399 Million** and over **5,600 years** of continuous laboratory work.

## 🧪 Key Features

-   **State-of-the-Art Accuracy:** Achieves over **93% subset accuracy** on a complex multi-label classification task, powered by a novel stacked ensemble architecture.
-   **Comprehensive Predictions:** Provides probabilistic predictions for metabolite presence in Blood, Urine, Saliva, Cerebrospinal Fluid (CSF), Feces, and Sweat.
-   **Large-Scale Annotation:** Includes the complete, pre-computed biofluid atlas for 171,456 metabolites from the Human Metabolome Database (HMDB).
-   **High-Throughput:** Capable of processing thousands of molecules in minutes, enabling metabolome-scale analysis.
-   **Transparent & Reproducible:** The repository contains all datasets, scripts, and trained models required to fully reproduce the original research and results.
-   **Multiple Model Architectures:** Includes implementations for Graph Neural Networks (GNN), LightGBM, standard XGBoost, and the final stacked ensemble model for comparative analysis.

## 📜 Scientific Context

The Human Metabolome Database (HMDB) catalogs hundreds of thousands of metabolites, yet the biological context for most remains unknown. Determining the biofluid in which a metabolite resides is a critical first step toward understanding its function, pathway, and potential as a biomarker. Traditional experimental methods (e.g., LC-MS/MS) are precise but suffer from extreme limitations in cost, time, and scale, creating a fundamental bottleneck in metabolomics research.

MetaboLocus was developed to break this bottleneck. Our central hypothesis—that chemical structure inherently encodes the information of a metabolite's biofluid fate—has been rigorously validated. By training on a large, curated dataset of experimentally confirmed metabolite-biofluid associations, our framework learns the complex, non-linear relationships between molecular properties and their physiological distribution, offering a powerful new engine for discovery.

## 📋 Table of Contents

-   [Installation](#-installation)
-   [Dataset](#-dataset)
-   [Usage](#-usage)
    -   [Quick Start: Predicting a Single SMILES String](#quick-start-predicting-a-single-smiles-string)
    -   [Batch Prediction from a File](#batch-prediction-from-a-file)
-   [Model Performance](#-model-performance)
-   [Repository Structure](#-repository-structure)
-   [How to Cite](#-how-to-cite)
-   [Contributing](#-contributing)
-   [License](#-license)
-   [Acknowledgements](#-acknowledgements)

## ⚙️ Installation

MetaboLocus requires Python 3.9 or higher. We strongly recommend using a `conda` environment to manage dependencies.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/QAIEDGE-MetaboLocus.git
    cd QAIEDGE-MetaboLocus
    ```

2.  **Create and activate a conda environment:**
    ```bash
    conda create -n metabolocus python=3.9 -y
    conda activate metabolocus
    ```

3.  **Install the required packages:**
    The dependencies are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## 🗂️ Dataset

The datasets used for training and prediction are located in the `/data` directory.

-   `data/processed/HMDB_labled_corpus_30k.csv`: The curated, labeled dataset of 29,956 metabolites used to train and evaluate the final stacked ensemble model.
-   `data/processed/HMDB_unannotated_corpus_171k.csv`: The corpus of 171,456 uncharacterized metabolites with valid SMILES strings.
-   `data/predictions/MetaboLocus_Annotations_171k.parquet`: The complete, high-confidence biofluid predictions for the unannotated corpus, provided in the efficient Parquet format.

## 🚀 Usage

The primary script for making predictions is `predict.py`. It leverages the final trained stacked ensemble model located in `/models`.

### Quick Start: Predicting a Single SMILES String

To get a quick prediction for a single molecule, use the `--smiles` argument.

```bash
python predict.py --smiles "CCO"

Example Output:

code
JSON
download
content_copy
expand_less
{
  "smiles": "CCO",
  "predictions": {
    "Blood": 0.987,
    "Urine": 0.954,
    "Saliva": 0.761,
    "CSF": 0.043,
    "Feces": 0.112,
    "Sweat": 0.001
  },
  "predicted_biofluids": [
    "Blood",
    "Urine",
    "Saliva"
  ]
}
Batch Prediction from a File

For high-throughput analysis, provide an input file containing one SMILES string per line.

Create an input file, e.g., my_metabolites.txt:

code
Text
download
content_copy
expand_less
CCO
C1=CC=C(C=C1)C(=O)O
CC(=O)OC1=CC=CC=C1C(=O)O

Run the prediction script:

code
Bash
download
content_copy
expand_less
python predict.py --input_file my_metabolites.txt --output_file predictions.csv

This will generate a predictions.csv file with the probability scores and final binary predictions for each biofluid.

📊 Model Performance

The final stacked ensemble model was rigorously evaluated on a held-out test set. The performance metrics underscore its state-of-the-art accuracy.

Metric	Score
Subset Accuracy	0.9352
Hamming Loss	0.0178
F1 Score (Micro)	0.9524
Precision (Micro)	0.9564
Recall (Micro)	0.9484
ROC AUC (Micro)	0.9967

For a detailed breakdown of class-wise performance and feature importance, please refer to the thesis/publication associated with this work.

📁 Repository Structure
code
Code
download
content_copy
expand_less
QAIEDGE-MetaboLocus/
│
├── data/
│   ├── raw/                # Raw HMDB XML data
│   ├── processed/          # Curated training and prediction datasets (.csv)
│   └── predictions/        # Pre-computed annotations for the 171k corpus (.parquet)
│
├── models/
│   └── stacked_ensemble/   # Final trained model artifacts (.pkl, .json)
│
├── notebooks/
│   ├── 01_Data_Curation.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training_and_Evaluation.ipynb
│   └── 04_Prediction_and_Analysis.ipynb
│
├── scripts/
│   ├── parse_hmdb.py       # Script for parsing raw HMDB XML
│   ├── train.py            # Master script for training models
│   └── feature_engineering/ # Modules for feature generation
│
├── predict.py              # Main script for running predictions
├── requirements.txt        # Project dependencies
├── .gitignore              # Git ignore file
├── LICENSE                 # MIT License
└── README.md               # This file
✒️ How to Cite

If you use MetaboLocus in your research, please cite the following work:

code
Bibtex
download
content_copy
expand_less
@phdthesis{YourName2025MetaboLocus,
  author       = {[Your Name]},
  title        = {A Machine Learning Framework for Predicting Metabolite Biofluid Distribution from Chemical Structure},
  school       = {[Your University, e.g., Massachusetts Institute of Technology]},
  year         = {2025},
  month        = {[Month]},
  url          = {https://github.com/YourUsername/QAIEDGE-MetaboLocus}
}
🤝 Contributing

We welcome contributions and suggestions to improve MetaboLocus. If you have an idea for a new feature or have found a bug, please open an issue on the GitHub repository.

📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

🙏 Acknowledgements

This work would not have been possible without the comprehensive data provided by the Human Metabolome Database (HMDB). We also extend our gratitude to the developers of the open-source scientific computing libraries that were essential to this project, including RDKit, Scikit-learn, XGBoost, and the PyTorch ecosystem.
