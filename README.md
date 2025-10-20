# QAIEDGE-MetaboLocus

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg) ![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)

**QAIEDGE-MetaboLocus** represents a paradigm shift in computational metabolomics. It is a pioneering machine learning framework that, for the first time in scientific history, establishes a direct, high-precision link between a metabolite's chemical structure—derived solely from its SMILES string—and its spatial localization across human biofluids.

This work confronts one of the most significant bottlenecks in modern metabolomics: the systematic characterization of where metabolites reside in the human body. By replacing prohibitive, multi-generational experimental workflows with a tractable computational solution, MetaboLocus heralds a new era of predictive metabolomics. Its state-of-the-art stacked ensemble model predicts metabolite distribution across six major biofluids: Blood, Urine, Saliva, Cerebrospinal Fluid (CSF), Feces, and Sweat.

The impact of this framework is monumental. In its initial application, MetaboLocus annotated **171,456 previously uncharacterized human metabolites**, a feat that would have demanded an estimated **$399 million and over 5,600 years of continuous laboratory work**. This project transforms a fundamental challenge in biological research into a powerful engine for discovery, poised to accelerate biomarker identification, enhance diagnostic design, and shape the future of personalized medicine.

---

## 💥 A Groundbreaking Contribution to Science

The spatial distribution of metabolites is a critical indicator of physiological state, yet its characterization has been fundamentally unscalable. Traditional methods like mass spectrometry and NMR spectroscopy, while precise, impose economic and temporal costs that have left the vast majority of the human metabolome functionally uncontextualized.

MetaboLocus definitively overcomes this limitation. It provides the first large-scale evidence of a robust, learnable relationship between chemical structure and biofluid fate.

*   **Unprecedented Scale and Savings**: By computationally annotating 171,456 metabolites, this work has already saved the scientific community nearly **$400 million** and millennia of experimental labor. Projecting its application to the 1.35 million known metabolites could yield savings of **$3.15 billion and 44,372 years** of research.
*   **Record-Breaking Accuracy**: The model achieves a remarkable **92.57% subset accuracy** on a complex multi-label classification task with 63 possible biofluid combinations, setting a new benchmark for predictive accuracy in the field.
*   **Sophisticated Architecture**: The final predictive framework is a multi-expert stacked ensemble, integrating an XGBoost model on physicochemical descriptors with a deep multi-layer perceptron on abstract representations from the ChemBERTa transformer model. This synergistic design ensures maximum accuracy and robustness.

---

## 🧪 Key Features

*   **State-of-the-Art Performance**: Achieves record-breaking accuracy in multi-label biofluid classification.
*   **Comprehensive Predictions**: Delivers calibrated, probabilistic predictions for six major human biofluids.
*   **Massive Annotation Atlas**: Provides a pre-computed biofluid atlas for 171,456 HMDB metabolites, with over 114,000 annotated at an almost definitive confidence level.
*   **Exceptional Throughput**: Processes thousands of molecules in minutes, turning years of lab work into seconds of computation.
*   **Full Reproducibility**: Includes all datasets, scripts, and pre-trained models to ensure the complete and transparent validation of our findings.
*   **Diverse Model Implementations**: Offers scripts for Graph Neural Networks (GNN), LightGBM (LGBM), XGBoost, and the final stacked ensemble.

---

## 📂 Repository Structure

```
QAIEDGE-MetaboLocus/
├── LICENSE
├── README.md
├── requirements.txt
├── Supplementary Data Files.txt
│
├── data/
│   ├── predictions/
│   │   └── hmdb_uncharacterized_metabolites_for_annotation_171456.csv
│   │
│   ├── processed/
│   │   ├── hmdb_mother_dataset_validated_46432.csv
│   │   └── hmdb_uncharacterized_metabolites_for_annotation_171456.csv
│   │
│   └── hmdb_uncharacterized_metabolites_for_annotation_171456.csv
│
├── models/
│   └── stacked_ensemble.pkl
│
└── scripts/
    ├── xgb/
    │   ├── hmdb_mother_dataset_validated_46432.csv
    │   ├── hmdb_uncharacterized_metabolites_for_annotation_171456.txt
    │   ├── xgb-1.875k-20m-v1.py
    │   ├── xgb-3.75k-0.5h-v1.py
    │   ├── xgb-7.5k-1.5h-v1.py
    │   ├── xgb-15k-1.5h-v1.py
    │   ├── xgb-25k-15m-v1.py
    │   ├── xgb-30k-15m-v1.0.py
    │   ├── xgb-30k-15m-v1.1.py
    │   ├── xgb-30k-15m-v1.2.py
    │   ├── xgb-30k-15m-v1.2-pred.py
    │   ├── xgb-35k-15m-v1.2.py
    │   ├── xgb-46k-15m-v1.2.py
    │   └── xgb-46k-30m-v1.3.py
    │
    ├── 15m-gnn.py
    └── 15m-lgbm.py
```

---

## 🚀 Reproducibility Guide

This section provides a clear and comprehensive guide to reproducing the results of this study. The Python scripts are architected to be self-contained and are optimized for execution in environments like Google Colab.

### **Step 1: Environment Setup**

While a `requirements.txt` file is provided for conventional local setups, it is not strictly necessary. **Each Python script contains all the requisite commands to install its own dependencies.** This design choice ensures maximum portability and allows you to run any script in a fresh environment without pre-configuration. Simply executing the script will handle the setup process automatically.

### **Step 2: Data Acquisition**

The large-scale datasets essential for training and prediction are hosted externally. The file `Supplementary Data Files.txt` contains Google Drive links to all necessary data files. Please download these assets and place them in the appropriate directories as referenced within the scripts.

For advanced users wishing to stream metabolite data directly, the full XML export from the Human Metabolome Database (HMDB), which exceeds 6GB, will also be required.

### **Step 3: Model Training**

The repository contains scripts to train various model architectures explored during this research:

*   **Graph Neural Network:** Execute `15m-gnn.py`.
*   **LightGBM Model:** Execute `15m-lgbm.py`.
*   **The Final, Optimized Model:** To train the best-performing XGBoost model that produced the final results, execute **`xgb-30k-15m-v1.2.py`**.

### **Step 4: Prediction on New Data**

To generate predictions on your own dataset of metabolites using the final trained model, utilize the dedicated prediction script:

*   **Prediction Script:** **`xgb-30k-15m-v1.2-pred.py`**

This script is thoroughly documented and provides a straightforward interface for generating biofluid annotations from SMILES strings. Following these steps will enable a complete and accurate reproduction of the results reported in our work.

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
@mastersthesis{Sarkar2025MetaboLocus,
  author       = {Sayem Sarkar},
  title        = {Machine Learning Based Identification of Metabolite Localization
in Human Biofluids from Molecular Structures Using the Human
Metabolome Database},
  school       = {Jahangirnagar University},
  year         = {2025},
  month        = {October},
  url          = {https://github.com/iamsayemsarkar/QAIEDGE-MetaboLocus}
}
```

---

## 👨‍💻 About the Developer

This research was conducted and developed by Sayem Sarkar. To learn more about his work and other projects, please visit his professional website:

**[https://qaiedge.com/](https://qaiedge.com/)**

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Please open an issue on GitHub to start a discussion.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

This work was made possible by the invaluable guidance of mentors, the foundational efforts of the metabolomics research community, and the spirit of collaboration that drives computational biology forward.
