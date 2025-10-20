# ==============================================================================
# Stacked Ensemble for Multi-Label Classification
# VERSION:  1.1
# PURPOSE:  A two-stage stacked ensemble for predicting metabolite biofluid
#           locations. This architecture trains an XGBoost model on molecular
#           fingerprints/descriptors and an MLP on ChemBERTa embeddings. A
#           logistic regression meta-model synthesizes base model predictions.
#
# ARCHITECTURE:
#   1. STACKED ENSEMBLE: A meta-learner trained on base model predictions.
#   2. DUAL-EXPERT SYSTEM:
#      - XGBoost Expert: Trained on RDKit-derived molecular features.
#        Includes per-class feature importance analysis.
#      - MLP Expert: PyTorch MLP for identifying non-linear patterns in
#        ChemBERTa embeddings.
#   3. META-LEARNER: Logistic regression model that weighs expert predictions
#      to generate a final classification.
# ==============================================================================

# ==============================================================================
# STEP 1: INSTALL DEPENDENCIES
# ==============================================================================

print("Installing dependencies...")
!pip install xgboost --quiet
!pip install lxml --quiet
!pip install rdkit --quiet
!pip install optuna --quiet
!pip install transformers torch --quiet
print("Dependencies installed.\n")

# ==============================================================================
# STEP 2: IMPORTS & CONFIGURATION
# ==============================================================================

print("Importing libraries and setting up configuration...")

import os
import pickle
import warnings
import zipfile
import random
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from lxml import etree
from tqdm.notebook import tqdm

import xgboost as xgb
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Fragments

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from transformers import AutoTokenizer, AutoModel

# --- Global Constants ---
DRIVE_XML_FILE_PATH: str = '/content/drive/My Drive/hmdb/hmdb_metabolites.xml'
MAX_METABOLITES_TO_PARSE: int = 30000
OUTPUT_ARCHIVE_NAME: str = 'StackedEnsemble_Run_30k.zip'

TARGET_BIOFLUIDS: List[str] = [
    'Blood', 'Urine', 'Saliva', 'Cerebrospinal Fluid (CSF)', 'Feces', 'Sweat'
]
RANDOM_SEED: int = 42

# --- ChemBERTa Configuration ---
CHEMBERTA_MODEL_NAME: str = 'seyonec/ChemBERTa-zinc-base-v1'

# --- XGBoost Optuna Configuration ---
XGB_TOTAL_OPTUNA_TIMEOUT_SECONDS: int = 900  # 15 minutes total
XGB_OPTUNA_EARLY_STOPPING_ROUNDS: int = 15

# --- MLP Expert Configuration ---
MLP_EPOCHS: int = 25
MLP_BATCH_SIZE: int = 256
MLP_LEARNING_RATE: float = 1e-4

# --- Feature Engineering Configuration ---
COMPUTED_NUMERICAL_FEATURES: List[str] = [
    'MolWt', 'ExactMolWt', 'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors',
    'NumValenceElectrons', 'NumAromaticRings', 'NumAliphaticRings', 'RingCount',
    'TPSA', 'MolLogP', 'MolMR', 'FractionCSP3', 'NumRotatableBonds', 'qed',
    'HallKierAlpha', 'MaxPartialCharge', 'MinPartialCharge', 'BalabanJ',
    'BertzCT', 'Chi0v', 'Kappa2', 'fr_NH2', 'fr_COO', 'fr_phenol',
    'fr_aldehyde', 'fr_ketone', 'fr_ether',
    'NumHDonors_vs_TPSA', 'MolLogP_vs_MolWt', 'RingCount_vs_HeavyAtomCount'
]
FINGERPRINT_BITS: int = 2048

# --- Environment Setup ---
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

try:
    from google.colab import drive, files
    drive.mount('/content/drive', force_remount=True)
    print("Google Drive mounted.")
except ImportError:
    print("Google Drive not detected; assuming non-Colab environment.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ==============================================================================
# STEP 3: DATA PARSING & FEATURE ENGINEERING FUNCTIONS
# ==============================================================================
print("Defining data processing and feature engineering functions...")

def parse_hmdb_for_smiles_and_biofluids(xml_path: str, max_records: int) -> List[Dict[str, Any]]:
    """Parses HMDB XML for metabolite records with SMILES and biofluid locations."""
    print(f"Parsing XML for up to {max_records} valid metabolites...")
    NAMESPACE = 'http://www.hmdb.ca'
    TAGS = {
        'metabolite': f'{{{NAMESPACE}}}metabolite', 'smiles': f'{{{NAMESPACE}}}smiles',
        'locations': f'{{{NAMESPACE}}}biospecimen_locations', 'specimen': f'{{{NAMESPACE}}}biospecimen'
    }
    parsed_data = []
    context = etree.iterparse(xml_path, events=('end',), tag=TAGS['metabolite'])

    for _, elem in tqdm(context, desc="Scanning Metabolite Records"):
        smiles_elem = elem.find(TAGS['smiles'])
        if smiles_elem is None or not smiles_elem.text:
            elem.clear()
            continue
        locs_elem = elem.find('.//' + TAGS['locations'])
        if locs_elem is not None:
            biofluids = [spec.text for spec in locs_elem.findall(TAGS['specimen'])]
            if biofluids:
                parsed_data.append({'smiles': smiles_elem.text, 'biofluids': list(set(biofluids))})
        # Memory cleanup for iterative parsing
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        if len(parsed_data) >= max_records:
            break
    print(f"Found {len(parsed_data)} metabolites with required data.")
    return parsed_data

def compute_all_features(data: List[Dict[str, Any]], target_biofluids: List[str], chemberta_tokenizer, chemberta_model, device) -> Tuple:
    """Computes RDKit features and ChemBERTa embeddings from SMILES strings."""
    print("\nPreprocessing data and engineering features...")
    df = pd.DataFrame(data)
    df['biofluids'] = df['biofluids'].apply(lambda lst: [b for b in lst if b in target_biofluids])
    df = df[df['biofluids'].apply(len) > 0].reset_index(drop=True)
    mlb = MultiLabelBinarizer(classes=target_biofluids)
    y = mlb.fit_transform(df['biofluids'])

    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)  # Suppress RDKit warnings
    rdkit_features, chemberta_embeddings = [], []

    for smiles in tqdm(df['smiles'], desc="Computing All Features"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rdkit_features.append(None)
            chemberta_embeddings.append(None)
            continue
        # RDKit Feature Computation
        try:
            Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
            fp = np.array(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FINGERPRINT_BITS).ToBitString()), dtype=np.uint8)
            desc = {
                'MolWt': Descriptors.MolWt(mol), 'ExactMolWt': Descriptors.ExactMolWt(mol),
                'HeavyAtomCount': Descriptors.HeavyAtomCount(mol), 'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumHDonors': Descriptors.NumHDonors(mol), 'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol), 'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
                'RingCount': Descriptors.RingCount(mol), 'TPSA': Descriptors.TPSA(mol),
                'MolLogP': Descriptors.MolLogP(mol), 'MolMR': Descriptors.MolMR(mol),
                'FractionCSP3': Descriptors.FractionCSP3(mol), 'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'qed': Descriptors.qed(mol), 'HallKierAlpha': Descriptors.HallKierAlpha(mol),
                'MaxPartialCharge': Descriptors.MaxPartialCharge(mol), 'MinPartialCharge': Descriptors.MinPartialCharge(mol),
                'BalabanJ': Descriptors.BalabanJ(mol), 'BertzCT': Descriptors.BertzCT(mol),
                'Chi0v': Descriptors.Chi0v(mol), 'Kappa2': Descriptors.Kappa2(mol),
                'fr_NH2': Fragments.fr_NH2(mol), 'fr_COO': Fragments.fr_COO(mol),
                'fr_phenol': Fragments.fr_phenol(mol), 'fr_aldehyde': Descriptors.fr_aldehyde(mol),
                'fr_ketone': Descriptors.fr_ketone(mol), 'fr_ether': Descriptors.fr_ether(mol)
            }
            epsilon = 1e-6
            desc['NumHDonors_vs_TPSA'] = desc['NumHDonors'] / (desc['TPSA'] + epsilon)
            desc['MolLogP_vs_MolWt'] = desc['MolLogP'] / (desc['MolWt'] + epsilon)
            desc['RingCount_vs_HeavyAtomCount'] = desc['RingCount'] / (desc['HeavyAtomCount'] + epsilon)
            for key, value in desc.items():
                if not np.isfinite(value): desc[key] = np.nan
            rdkit_features.append({'fingerprint': fp, 'descriptors': desc})
        except:
            rdkit_features.append(None)

        # ChemBERTa Embedding Computation
        try:
            with torch.no_grad():
                inputs = chemberta_tokenizer(smiles, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
                embedding = chemberta_model(**inputs).last_hidden_state[0, 0, :].cpu().numpy()
            chemberta_embeddings.append(embedding)
        except:
            chemberta_embeddings.append(None)
    lg.setLevel(RDLogger.INFO) # Restore RDKit logger

    # Filter out samples where feature generation failed
    valid_indices = [i for i, (rdkit, embed) in enumerate(zip(rdkit_features, chemberta_embeddings)) if rdkit is not None and embed is not None]
    y_clean = y[valid_indices]
    df_clean = df.iloc[valid_indices].reset_index(drop=True)

    # Assemble RDKit feature matrix (X_rdkit)
    X_fp = np.vstack([rdkit_features[i]['fingerprint'] for i in valid_indices])
    desc_df = pd.DataFrame([rdkit_features[i]['descriptors'] for i in valid_indices]).reindex(columns=COMPUTED_NUMERICAL_FEATURES)
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(imputer.fit_transform(desc_df))
    X_rdkit = np.concatenate([X_fp, X_num_scaled], axis=1)

    # Assemble ChemBERTa embedding matrix (X_embed)
    X_embed = np.vstack([chemberta_embeddings[i] for i in valid_indices])

    print(f"\nFinal dataset size: {len(y_clean)} metabolites.")
    print(f"  - RDKit feature matrix shape: {X_rdkit.shape}")
    print(f"  - ChemBERTa embedding matrix shape: {X_embed.shape}")

    return X_rdkit, X_embed, y_clean, df_clean, mlb, imputer, scaler

# ==============================================================================
# STEP 4: EXPERT AND META-MODEL DEFINITIONS
# ==============================================================================
print("Defining expert and meta-models...")

class PinnacleEnsembleModel(BaseEstimator, ClassifierMixin):
    """Wrapper for a collection of one-vs-rest classifiers."""
    def __init__(self, estimators: List[Any]):
        self.estimators_ = estimators
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)
    def predict_proba(self, X):
        return np.hstack([est.predict_proba(X)[:, 1].reshape(-1, 1) for est in self.estimators_])

class OptunaEarlyStoppingCallback:
    """Callback to stop Optuna study if validation score does not improve."""
    def __init__(self, early_stopping_rounds: int):
        self._iter = 0
        self.best_value = -float('inf')
        self.early_stopping_rounds = early_stopping_rounds
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if trial.value is not None and trial.value > self.best_value:
            self.best_value = trial.value
            self._iter = 0
        else:
            self._iter += 1
        if self._iter >= self.early_stopping_rounds:
            study.stop()

class MLPEmbeddingExpert(nn.Module):
    """MLP classifier for ChemBERTa embeddings."""
    def __init__(self, input_dim: int, num_classes: int):
        super(MLPEmbeddingExpert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.network(x)

def train_mlp_expert(X_train_embed, y_train, X_val_embed, y_val, binarizer, device):
    """Trains the MLP expert on embedding data."""
    print("\n--- Training Expert B: MLP on ChemBERTa Embeddings ---")
    input_dim = X_train_embed.shape[1]
    num_classes = len(binarizer.classes_)
    model = MLPEmbeddingExpert(input_dim, num_classes).to(device)

    train_dataset = TensorDataset(torch.FloatTensor(X_train_embed), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=MLP_BATCH_SIZE, shuffle=True)
    val_dataset = TensorDataset(torch.FloatTensor(X_val_embed), torch.FloatTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=MLP_BATCH_SIZE, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=MLP_LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=3)

    best_val_f1 = -1
    for epoch in range(MLP_EPOCHS):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                preds = torch.sigmoid(outputs)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        y_pred_val = (np.vstack(all_preds) > 0.5).astype(int)
        val_f1_micro = f1_score(np.vstack(all_labels), y_pred_val, average='micro', zero_division=0)
        print(f"Epoch {epoch+1}/{MLP_EPOCHS}, Val Micro F1: {val_f1_micro:.6f}")
        scheduler.step(val_f1_micro)
        if val_f1_micro > best_val_f1:
            best_val_f1 = val_f1_micro
            torch.save(model.state_dict(), 'best_mlp_expert.pth')

    print(f"Best MLP Validation Micro F1: {best_val_f1:.6f}. Model state saved.")
    model.load_state_dict(torch.load('best_mlp_expert.pth'))
    return model

def get_mlp_predictions(model, X_embed, device):
    """Generates probability predictions from a trained MLP model."""
    model.eval()
    dataset = TensorDataset(torch.FloatTensor(X_embed))
    loader = DataLoader(dataset, batch_size=MLP_BATCH_SIZE * 2, shuffle=False)
    all_probas = []
    with torch.no_grad():
        for features in loader:
            features = features[0].to(device)
            outputs = model(features)
            probas = torch.sigmoid(outputs)
            all_probas.append(probas.cpu().numpy())
    return np.vstack(all_probas)

class StackingMetaModel:
    """Logistic regression meta-learner for stacking."""
    def __init__(self):
        self.models = [LogisticRegression(solver='liblinear', random_state=RANDOM_SEED) for _ in range(len(TARGET_BIOFLUIDS))]
    def fit(self, X, y):
        for i, model in enumerate(self.models):
            model.fit(X, y[:, i])
    def predict_proba(self, X):
        return np.hstack([model.predict_proba(X)[:, 1].reshape(-1, 1) for model in self.models])
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)

# ==============================================================================
# STEP 5: EVALUATION AND ANALYSIS FUNCTIONS
# ==============================================================================
print("Defining evaluation and utility functions...")

def print_full_evaluation_report(y_true, y_pred, y_pred_proba, model_name, binarizer):
    """Prints a comprehensive multi-label classification report."""
    print(f"\n--- {model_name} Performance Evaluation ---")
    print("\n" + "="*40 + f"\n      Global Correctness Metrics\n" + "="*40)
    print(f"| {'Metric':<20} | {'Score':<15} |\n|{'-'*22}|{'-'*17}|")
    print(f"| {'Subset Accuracy':<20} | {accuracy_score(y_true, y_pred):<15.6f} |")
    print(f"| {'Hamming Loss':<20} | {hamming_loss(y_true, y_pred):<15.6f} |")
    print("="*40 + "\n\n" + "="*53 + f"\n            Label-wise Performance Metrics\n" + "="*53)
    print(f"| {'Metric':<20} | {'Micro':<15} | {'Macro':<15} |\n|{'-'*22}|{'-'*17}|{'-'*17}|")
    metrics = {"F1 Score": f1_score, "Precision": precision_score, "Recall": recall_score, "ROC AUC": roc_auc_score}
    for name, func in metrics.items():
        if name == "ROC AUC":
            micro = func(y_true, y_pred_proba, average='micro')
            macro = func(y_true, y_pred_proba, average='macro')
        else:
            micro = func(y_true, y_pred, average='micro', zero_division=0)
            macro = func(y_true, y_pred, average='macro', zero_division=0)
        print(f"| {name:<20} | {micro:<15.6f} | {macro:<15.6f} |")
    print("="*53)

    print("\n\n--- Detailed Class-wise Performance Report ---")
    report = classification_report(y_true, y_pred, target_names=binarizer.classes_, output_dict=True, zero_division=0)
    print("\n" + "="*70)
    print(f"| {'Class':<30} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Support':<7} |")
    print(f"|{'-'*32}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*9}|")
    for class_name, class_metrics in report.items():
        if class_name in binarizer.classes_:
            print(f"| {class_name:<30} | {class_metrics['precision']:<12.4f} | {class_metrics['recall']:<12.4f} | {class_metrics['f1-score']:<12.4f} | {class_metrics['support']:<7} |")
    print("="*70)

def find_optimal_threshold(model, X_val, y_val) -> float:
    """Finds the optimal prediction threshold based on subset accuracy on validation data."""
    print("\n--- Optimizing Prediction Threshold ---")
    y_pred_proba_val = model.predict_proba(X_val)
    best_score, best_threshold = -1, 0.5
    for threshold in np.arange(0.1, 0.9, 0.01):
        score = accuracy_score(y_val, (y_pred_proba_val > threshold).astype(int))
        if score > best_score:
            best_score, best_threshold = score, threshold
    print(f"Optimal threshold found: {best_threshold:.2f} (Validation Subset Accuracy: {best_score:.6f})")
    return best_threshold

def display_class_distribution(y: np.ndarray, binarizer: MultiLabelBinarizer):
    """Displays the distribution of samples per class."""
    print("\n--- Biofluid Class Distribution ---")
    class_counts = np.sum(y, axis=0)
    total_samples = len(y)
    class_info = sorted(zip(binarizer.classes_, class_counts), key=lambda x: x[1], reverse=True)
    print("\n" + "="*56)
    print(f"| {'Biofluid Class':<30} | {'Count':<10} | {'Percentage (%)':<10} |")
    print(f"|{'-'*32}|{'-'*12}|{'-'*16}|")
    for class_name, count in class_info:
        percentage = (count / total_samples) * 100
        print(f"| {class_name:<30} | {count:<10} | {percentage:<14.2f} |")
    print("="*56)

def display_representative_predictions(df_test, y_test, y_pred_proba, binarizer, optimal_threshold, n_samples=10):
    """Displays a sample of test predictions for qualitative analysis."""
    print(f"\n--- Representative Test Predictions (n={n_samples}) ---")
    random_indices = random.sample(range(len(df_test)), min(n_samples, len(df_test)))
    for i, idx in enumerate(random_indices):
        true_labels = binarizer.classes_[y_test[idx] == 1]
        sample_probas = y_pred_proba[idx]
        pred_labels = binarizer.classes_[sample_probas >= optimal_threshold]
        print(f"\n--- Sample #{i+1} (Index: {df_test.index[idx]}) ---")
        print(f"  - SMILES            : {df_test.iloc[idx]['smiles']}")
        print(f"  - True Biofluids    : {', '.join(true_labels) if len(true_labels) > 0 else 'None'}")
        print(f"  - Predicted Biofluids : {', '.join(pred_labels) if len(pred_labels) > 0 else 'None'}")
        print("  - Confidence Scores :")
        for j, fluid in enumerate(binarizer.classes_):
            is_predicted = " [*]" if sample_probas[j] >= optimal_threshold else ""
            print(f"    - {fluid:<28}: {sample_probas[j]:.4f}{is_predicted}")

def display_per_class_feature_importance(models: List[Any], binarizer, descriptors: List[str], fp_bits: int):
    """Calculates and displays feature importance for each class in the XGBoost model."""
    print("\n--- XGBoost Expert: Per-Class Feature Importance ---")
    for i, class_name in enumerate(binarizer.classes_):
        print(f"\n{'='*50}\n       Feature Importance for Class: {class_name}\n{'='*50}")
        importances = models[i].feature_importances_
        # Aggregate fingerprint importance
        feature_map = {'Fingerprints': np.sum(importances[:fp_bits])}
        # Map descriptor importance
        for j, desc_name in enumerate(descriptors):
            feature_map[desc_name] = importances[fp_bits + j]
        total_importance = sum(feature_map.values())
        if total_importance == 0: continue
        # Sort and display feature contributions
        sorted_features = sorted([(name, (imp/total_importance)*100) for name, imp in feature_map.items()], key=lambda x: x[1], reverse=True)
        print(f"| {'Feature':<30} | {'Contribution (%)':<15} |\n|{'-'*32}|{'-'*17}|")
        for name, imp in sorted_features:
            print(f"| {name:<30} | {imp:<15.4f} |")
        print("="*50)

# ==============================================================================
# STEP 6: MAIN EXECUTION WORKFLOW
# ==============================================================================

if __name__ == "__main__":
    print("--- Starting Stacked Ensemble Workflow ---")

    print(f"Loading ChemBERTa model: '{CHEMBERTA_MODEL_NAME}'...")
    chemberta_tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL_NAME)
    chemberta_model = AutoModel.from_pretrained(CHEMBERTA_MODEL_NAME).to(device)
    chemberta_model.eval()

    if not os.path.exists(DRIVE_XML_FILE_PATH):
        print(f"CRITICAL ERROR: File not found at '{DRIVE_XML_FILE_PATH}'. Halting.")
    else:
        parsed_data = parse_hmdb_for_smiles_and_biofluids(DRIVE_XML_FILE_PATH, MAX_METABOLITES_TO_PARSE)
        X_rdkit, X_embed, y, df_clean, binarizer, imputer, scaler = compute_all_features(parsed_data, TARGET_BIOFLUIDS, chemberta_tokenizer, chemberta_model, device)
        display_class_distribution(y, binarizer)

        # Split data into training, validation, and test sets
        indices = np.arange(X_rdkit.shape[0])
        try: # Attempt stratified split
            train_full_idx, test_idx, y_train_full, y_test = train_test_split(indices, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)
            val_size_frac = 0.15 / 0.85
            train_idx, val_idx, y_train, y_val = train_test_split(train_full_idx, y_train_full, test_size=val_size_frac, random_state=RANDOM_SEED, stratify=y_train_full)
            print("\nData splitting complete (stratified).")
        except ValueError: # Fallback for insufficient samples in a class
            train_full_idx, test_idx, y_train_full, y_test = train_test_split(indices, y, test_size=0.15, random_state=RANDOM_SEED)
            val_size_frac = 0.15 / 0.85
            train_idx, val_idx, y_train, y_val = train_test_split(train_full_idx, y_train_full, test_size=val_size_frac, random_state=RANDOM_SEED)
            print("\nData splitting complete (non-stratified fallback).")

        X_train_rdkit, X_val_rdkit, X_test_rdkit = X_rdkit[train_idx], X_rdkit[val_idx], X_rdkit[test_idx]
        X_train_embed, X_val_embed, X_test_embed = X_embed[train_idx], X_embed[val_idx], X_embed[test_idx]
        X_train_full_rdkit = X_rdkit[train_full_idx]
        print(f"  - Samples: {len(train_idx)} train, {len(val_idx)} validation, {len(test_idx)} test")

        # ==========================================================================
        # STAGE 1A: TRAIN EXPERT A (XGBoost on RDKit Features)
        # ==========================================================================
        print("\n--- Training Expert A: XGBoost on RDKit Features ---")
        def xgb_objective(trial, X_train, y_train_single, X_val, y_val_single):
            params = {
                'tree_method': 'gpu_hist', 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'random_state': RANDOM_SEED, 'n_jobs': -1,
                'n_estimators': trial.suggest_int('n_estimators', 500, 2500, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 6, 12),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'gamma': trial.suggest_float('gamma', 0.01, 20.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 4, 30),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
            }
            n_pos = np.sum(y_train_single == 1)
            scale_pos_weight = (len(y_train_single) - n_pos) / n_pos if n_pos > 0 else 1
            model = xgb.XGBClassifier(**params, scale_pos_weight=scale_pos_weight)
            model.fit(X_train, y_train_single)
            return f1_score(y_val_single, model.predict(X_val), average='binary', zero_division=0)

        # Allocate optimization time based on class prevalence
        time_budgets = {'Blood': 0.30, 'Urine': 0.14, 'Feces': 0.14, 'Saliva': 0.14, 'Cerebrospinal Fluid (CSF)': 0.14, 'Sweat': 0.14}
        time_budgets = {k: int(v * XGB_TOTAL_OPTUNA_TIMEOUT_SECONDS) for k, v in time_budgets.items()}

        xgb_specialist_models = []
        # First, find good foundational parameters on the most prevalent class ('Blood')
        blood_idx = binarizer.classes_.tolist().index('Blood')
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: xgb_objective(t, X_train_rdkit, y_train[:, blood_idx], X_val_rdkit, y_val[:, blood_idx]), timeout=time_budgets['Blood'])
        foundational_params = study.best_params

        for i, class_name in enumerate(binarizer.classes_):
            print(f"-- Optimizing XGB for Class: {class_name} ({time_budgets[class_name]}s) --")
            study = optuna.create_study(direction='maximize')
            study.enqueue_trial(foundational_params) # Seed search with good params
            study.optimize(lambda t: xgb_objective(t, X_train_rdkit, y_train[:, i], X_val_rdkit, y_val[:, i]), timeout=time_budgets[class_name])
            final_params = {'tree_method': 'gpu_hist', 'objective': 'binary:logistic', 'random_state': RANDOM_SEED, 'n_jobs': -1, **study.best_params}
            y_train_full_single = y_train_full[:, i]
            n_pos = np.sum(y_train_full_single == 1)
            scale_pos_weight = (len(y_train_full_single) - n_pos) / n_pos if n_pos > 0 else 1
            specialist = xgb.XGBClassifier(**final_params, scale_pos_weight=scale_pos_weight)
            specialist.fit(X_rdkit[train_full_idx], y_train_full_single)
            xgb_specialist_models.append(specialist)

        xgb_expert_model = PinnacleEnsembleModel(estimators=xgb_specialist_models)
        print("--- Expert A (XGBoost) training complete. ---")

        # ==========================================================================
        # STAGE 1B: TRAIN EXPERT B (MLP on ChemBERTa Embeddings)
        # ==========================================================================
        mlp_expert_model = train_mlp_expert(X_train_embed, y_train, X_val_embed, y_val, binarizer, device)
        print("--- Expert B (MLP) training complete. ---")

        # ==========================================================================
        # STAGE 2: TRAIN META-MODEL
        # ==========================================================================
        print("\n--- Training Stage 2: Meta-Model ---")
        # Generate expert predictions on the validation set for meta-training
        xgb_val_preds = xgb_expert_model.predict_proba(X_val_rdkit)
        mlp_val_preds = get_mlp_predictions(mlp_expert_model, X_val_embed, device)
        X_meta_train = np.concatenate([xgb_val_preds, mlp_val_preds], axis=1)
        meta_model = StackingMetaModel()
        meta_model.fit(X_meta_train, y_val)
        print("--- Meta-Model training complete. ---")

        # ==========================================================================
        # FINAL EVALUATION
        # ==========================================================================
        print("\n--- Final Evaluation on Test Set ---")
        # Generate expert predictions on the test set
        xgb_test_preds = xgb_expert_model.predict_proba(X_test_rdkit)
        mlp_test_preds = get_mlp_predictions(mlp_expert_model, X_test_embed, device)
        X_meta_test = np.concatenate([xgb_test_preds, mlp_test_preds], axis=1)

        optimal_threshold = find_optimal_threshold(meta_model, X_meta_train, y_val)
        y_pred_proba_final = meta_model.predict_proba(X_meta_test)
        y_pred_final = (y_pred_proba_final > optimal_threshold).astype(int)

        print_full_evaluation_report(y_test, y_pred_final, y_pred_proba_final, "Stacked Ensemble Model", binarizer)
        display_per_class_feature_importance(xgb_expert_model.estimators_, binarizer, COMPUTED_NUMERICAL_FEATURES, FINGERPRINT_BITS)
        display_representative_predictions(df_clean.iloc[test_idx], y_test, y_pred_proba_final, binarizer, optimal_threshold)

        # ==========================================================================
        # SAVE ARTIFACTS
        # ==========================================================================
        print("\n--- Saving Model Artifacts ---")
        model_package = {
            'xgb_expert': xgb_expert_model,
            'mlp_expert_state_dict': mlp_expert_model.state_dict(),
            'meta_model': meta_model,
            'binarizer': binarizer,
            'rdkit_imputer': imputer,
            'rdkit_scaler': scaler,
            'optimal_threshold': optimal_threshold
        }
        model_filename = 'stacked_ensemble.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(model_package, f)

        df_clean.to_csv("complete_processed_dataset.csv", index=False)
        with zipfile.ZipFile(OUTPUT_ARCHIVE_NAME, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(model_filename)
            zf.write("complete_processed_dataset.csv")
        os.remove(model_filename)
        os.remove("complete_processed_dataset.csv")
        print(f"Artifacts saved to '{OUTPUT_ARCHIVE_NAME}'.")

        try:
            files.download(OUTPUT_ARCHIVE_NAME)
        except (ImportError, NameError):
            print(f"\nDownload '{OUTPUT_ARCHIVE_NAME}' from the file browser.")

    print("\n--- Workflow Execution Complete ---")
