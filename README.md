# QAIEDGE-MetaboLocus

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)

**QAIEDGE-MetaboLocus** is a state-of-the-art machine learning framework for predicting the biofluid localization of human metabolites from chemical structure.

MetaboLocus addresses a fundamental challenge in metabolomics: identifying the biological location of metabolites. Inspired by the transformative impact of AlphaFold in proteomics, it employs a sophisticated multi-expert stacked ensemble model to predict the distribution of metabolites across six major human biofluids—Blood, Urine, Saliva, Cerebrospinal Fluid (CSF), Feces, and Sweat—using only molecular structure as input.

By providing high-throughput, high-accuracy predictions, MetaboLocus eliminates the need for costly and time-consuming experimental workflows. Using this framework, we have annotated 171,456 previously uncharacterized human metabolites—a task that would otherwise require an estimated $399 million and over 5,600 years of laboratory work.

---

## 🧪 Key Features

*   **State-of-the-Art Accuracy**: Achieves over 92% subset accuracy on a complex multi-label classification task.
*   **Comprehensive Predictions**: Probabilistic predictions for six human biofluids.
*   **Large-Scale Annotation**: Pre-computed biofluid atlas for 171,456 HMDB metabolites.
*   **High-Throughput**: Processes thousands of molecules within minutes.
*   **Reproducible**: All datasets, scripts, and trained models included for full reproducibility.
*   **Multiple Model Architectures**: Includes Graph Neural Networks (GNN), LightGBM (LGBM), XGBoost, and the final stacked ensemble.

---

## 📜 Scientific Context

Although the Human Metabolome Database catalogs hundreds of thousands of metabolites, the biological context of most remains unknown. Determining the biofluid in which a metabolite resides is crucial for understanding its function, pathway, and biomarker potential. Traditional experimental methods, such as LC-MS/MS, are precise but limited by cost, time, and scale.

MetaboLocus overcomes these limitations. By learning complex relationships between molecular structure and biofluid distribution from a curated dataset of experimentally validated metabolite-biofluid associations, the framework enables predictive metabolomics at an unprecedented scale.

---

## 📂 Repository Structure

```
QAIEDGE-MetaboLocus/
│
├── data/
│   ├── predictions/
│   │   └── hmdb_uncharacterized_metabolites_for_annotation_171456.csv
│   └── processed/
│       ├── hmdb_mother_dataset_validated_46432.csv
│       ├── hmdb_uncharacterized_metabolites_for_annotation_171456.csv
│       └── Supplementary Data Files.txt
│
├── models/
│   └── stacked_ensemble.pkl
│
├── scripts/
│   ├── xgb/
│   │   ├── hmdb_mother_dataset_validated_46432.csv
│   │   ├── hmdb_uncharacterized_metabolites_for_annotation_171456.txt
│   │   ├── xgb-1.875k-20m-v1.py
│   │   ├── xgb-3.75k-0.5h-v1.py
│   │   ├── xgb-7.5k-1.5h-v1.py
│   │   ├── xgb-15k-1.5h-v1.py
│   │   ├── xgb-25k-15m-v1.py
│   │   ├── xgb-30k-15m-v1.0.py
│   │   ├── xgb-30k-15m-v1.1.txt
│   │   ├── xgb-30k-15m-v1.2-pred.txt
│   │   ├── xgb-30k-15m-v1.2.txt
│   │   ├── xgb-35k-15m-v1.2.txt
│   │   ├── xgb-46k-15m-v1.2.txt
│   │   └── xgb-46k-30m-v1.3.txt
│   ├── 15m-gnn.py
│   └── 15m-lgbm.py
│
├── LICENSE
├── README.md
└── requirements.txt
```

### Folder Descriptions:

*   **`data/processed/`**: Curated training datasets and supplementary files.
*   **`data/predictions/`**: Pre-computed annotations for uncharacterized metabolites.
*   **`models/`**: Final trained stacked ensemble model.
*   **`scripts/xgb/`**: All XGBoost training and prediction scripts.
*   **`scripts/15m-gnn.py` & `15m-lgbm.py`**: Baseline GNN and LightGBM models.
*   **`requirements.txt`**: Python dependencies.
*   **`README.md` & `LICENSE`**: Documentation and licensing.

---

## ⚙️ Installation

Requires Python 3.9+. Conda is recommended:

```bash
git clone https://github.com/iamsayemsarkar/QAIEDGE-MetaboLocus.git
cd QAIEDGE-MetaboLocus
conda create -n metabolocus python=3.9 -y
conda activate metabolocus
pip install -r requirements.txt
```

---

## 🚀 Usage

### Predict a single SMILES string:

```bash
python scripts/xgb/xgb-30k-15m-v1.2-pred.txt --smiles "CCO"
```

### Batch prediction from a file:

```bash
python scripts/xgb/xgb-30k-15m-v1.2-pred.txt --input_file my_metabolites.txt --output_file predictions.csv
```

---

## 📊 Model Performance

### Global Correctness Metrics:

| Metric          | Score    |
| --------------- | -------- |
| Subset Accuracy | 0.925679 |
| Hamming Loss    | 0.019545 |

### Label-wise Performance Metrics (Micro):

| Metric    | Score    |
| --------- | -------- |
| F1 Score  | 0.952381 |
| Precision | 0.956443 |
| Recall    | 0.948353 |
| ROC AUC   | 0.996683 |

---

## ✒️ How to Cite

```bibtex
@phdthesis{Sarkar2025MetaboLocus,
  author       = {Sayem Sarkar},
  title        = {Machine Learning–Driven Prediction of Metabolite Localization in Human Biofluids from Molecular Structures Using the Human Metabolome Database},
  school       = {Jahangirnagar University},
  year         = {2025},
  month        = {October},
  url          = {https://github.com/iamsayemsarkar/QAIEDGE-MetaboLocus}
}
```

---

## 🤝 Contributing

Contributions and suggestions are welcome. Open an issue on GitHub for bugs or feature requests.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

This work was made possible through the guidance of mentors, the metabolomics research community, and collaborative efforts in computational biology.
