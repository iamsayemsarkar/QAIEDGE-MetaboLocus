# ==============================================================================
# The Supreme Stacked Ensemble for World-Class Predictive Performance
# VERSION:  1.3 (World Class, 46k Master Dataset - Feature Importance Restored)
# PURPOSE:
#   To achieve an unprecedented level of predictive accuracy on the complete and
#   validated Human Metabolome Database (46,432 metabolites). This script
#   deploys a state-of-the-art Stacked Ensemble architecture, featuring
#   surgically optimized molecular descriptors, novel high-impact engineered
#   features, and a dual-objective optimization strategy. The primary goal is
#   to maximize Subset Accuracy to its theoretical peak, leveraging a 30-minute
#   intelligent hyperparameter search.
#
# KEY FEATURES:
#   1. MASTER DATASET UTILIZATION: Directly ingests the pristine 46,432
#      metabolite CSV, unlocking the full potential of the dataset.
#   2. STRATEGIC DESCRIPTOR ENGINEERING: Eliminates low-impact descriptors
#      (e.g., Kappa2, HallKierAlpha) while introducing high-impact analogues
#      (VSA_EState descriptors) and a novel engineered feature (Wt_vs_TPSA).
#   3. DUAL-OBJECTIVE OPTIMIZATION: A sophisticated two-phase strategy where
#      base "expert" models are optimized for F1-Score (to master rare
#      classes) and the final "meta-model" is optimized for Subset Accuracy.
#   4. SUPREME TIME ALLOCATION (30 MINS): A refined Asymmetric Timed
#      Optimization (ATO) protocol allocates the 1800-second budget
#      intelligently across the six biofluid classes for the deepest possible search.
#   5. FULL INTERPRETABILITY: Includes a comprehensive Per-Class Feature
#      Importance analysis for the structural expert model (XGBoost).
# ==============================================================================

# ==============================================================================
# STEP 1: INSTALL DEPENDENCIES
# ==============================================================================
print("--- Installing required libraries ---")
!pip install xgboost --quiet
!pip install rdkit --quiet
!pip install optuna --quiet
!pip install transformers torch --quiet
print("Libraries installed.\n")

# ==============================================================================
# STEP 2: IMPORTS & WORLD-CLASS CONFIGURATION
# ==============================================================================
print("--- Importing libraries and setting up the world-class configuration ---")
import os
import pickle
import warnings
import zipfile
import random
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import xgboost as xgb
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.EState import EState_VSA
from rdkit.Chem import Fragments

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

# --- Supreme Configuration for the 46k Master Dataset ---
MASTER_CSV_PATH: str = '/content/drive/My Drive/hmdb/hmdb_mother_dataset_validated_46432.csv'
METABOLITES_TO_USE: int = 500 # Default to 500 for test. Set to 46432 for the ultimate run.
OUTPUT_ARCHIVE_NAME: str = f'XGB_WorldClass_StackedEnsemble_Run_{METABOLITES_TO_USE}.zip'

TARGET_BIOFLUIDS: List[str] = ['Blood', 'Urine', 'Saliva', 'Cerebrospinal Fluid (CSF)', 'Feces', 'Sweat']
RANDOM_SEED: int = 42

# --- ChemBERTa Configuration ---
CHEMBERTA_MODEL_NAME: str = 'seyonec/ChemBERTa-zinc-base-v1'

# --- Optuna Configuration: Supreme ATO Protocol (30 Minutes) ---
TOTAL_OPTUNA_TIMEOUT_SECONDS: int = 1800 # 30 minutes
if METABOLITES_TO_USE <= 500: TOTAL_OPTUNA_TIMEOUT_SECONDS = 30 # Scale for test runs
XGB_OPTUNA_EARLY_STOPPING_ROUNDS: int = 20

# --- MLP Expert Configuration ---
MLP_EPOCHS: int = 30
MLP_BATCH_SIZE: int = 256
MLP_LEARNING_RATE: float = 5e-5

# --- Feature Engineering Configuration: Surgically Optimized Descriptors ---
FINGERPRINT_BITS: int = 2048
OPTIMIZED_DESCRIPTORS: List[str] = [
    'MolWt', 'ExactMolWt', 'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors',
    'NumValenceElectrons', 'NumAromaticRings', 'NumAliphaticRings', 'RingCount',
    'TPSA', 'MolLogP', 'MolMR', 'FractionCSP3', 'NumRotatableBonds', 'qed',
    'MaxPartialCharge', 'MinPartialCharge', 'BalabanJ', 'BertzCT',
    'fr_NH2', 'fr_COO', 'fr_phenol', 'fr_aldehyde', 'fr_ketone', 'fr_ether', 'fr_unbrch_alkane',
    'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5',
    'Wt_vs_TPSA'
]

# --- Environment Setup ---
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

try:
    from google.colab import drive, files
    drive.mount('/content/drive', force_remount=True)
    print("Google Drive mounted successfully.")
except ImportError:
    print("Could not mount Google Drive. Assuming non-Colab environment.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# ==============================================================================
# STEP 3: DATA INGESTION & FEATURE ENGINEERING
# ==============================================================================
print("--- Defining data ingestion and supreme feature engineering functions ---")

def load_and_prepare_data(csv_path: str, max_records: int, target_biofluids: List[str]) -> pd.DataFrame:
    print(f"Loading master dataset from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    if max_records < len(df):
        print(f"Subsetting to {max_records} records for this run.")
        df = df.head(max_records)
    df = df.dropna(subset=['smiles'] + target_biofluids).copy()
    df = df[['smiles'] + target_biofluids]
    print(f"Loaded {len(df)} valid records.")
    return df

def compute_all_features(df: pd.DataFrame, chemberta_tokenizer, chemberta_model, device) -> Tuple:
    print("\nEngineering all feature sets from SMILES...")
    y = df[TARGET_BIOFLUIDS].values
    mlb = MultiLabelBinarizer(classes=TARGET_BIOFLUIDS).fit(None)

    lg = RDLogger.logger(); lg.setLevel(RDLogger.CRITICAL)
    rdkit_features, chemberta_embeddings = [], []

    for smiles in tqdm(df['smiles'], desc="Computing All Features"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rdkit_features.append(None); chemberta_embeddings.append(None)
            continue
        try:
            Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
            fp = np.array(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FINGERPRINT_BITS).ToBitString()), dtype=np.uint8)
            vsa_descriptors = EState_VSA.EState_VSA_(mol)
            desc = {
                'MolWt': Descriptors.MolWt(mol), 'ExactMolWt': Descriptors.ExactMolWt(mol), 'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol), 'NumHDonors': Descriptors.NumHDonors(mol), 'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol), 'NumAliphaticRings': Descriptors.NumAliphaticRings(mol), 'RingCount': Descriptors.RingCount(mol),
                'TPSA': Descriptors.TPSA(mol), 'MolLogP': Descriptors.MolLogP(mol), 'MolMR': Descriptors.MolMR(mol), 'FractionCSP3': Descriptors.FractionCSP3(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol), 'qed': Descriptors.qed(mol), 'MaxPartialCharge': Descriptors.MaxPartialCharge(mol),
                'MinPartialCharge': Descriptors.MinPartialCharge(mol), 'BalabanJ': Descriptors.BalabanJ(mol), 'BertzCT': Descriptors.BertzCT(mol),
                'fr_NH2': Fragments.fr_NH2(mol), 'fr_COO': Fragments.fr_COO(mol), 'fr_phenol': Fragments.fr_phenol(mol), 'fr_aldehyde': Fragments.fr_aldehyde(mol),
                'fr_ketone': Fragments.fr_ketone(mol), 'fr_ether': Fragments.fr_ether(mol), 'fr_unbrch_alkane': Fragments.fr_unbrch_alkane(mol),
                'VSA_EState1': vsa_descriptors[0], 'VSA_EState2': vsa_descriptors[1], 'VSA_EState3': vsa_descriptors[2],
                'VSA_EState4': vsa_descriptors[3], 'VSA_EState5': vsa_descriptors[4]
            }
            desc['Wt_vs_TPSA'] = desc['MolWt'] / (desc['TPSA'] + 1e-6)
            for key, value in desc.items():
                if not np.isfinite(value): desc[key] = np.nan
            rdkit_features.append({'fingerprint': fp, 'descriptors': desc})
        except Exception: rdkit_features.append(None)
        try:
            with torch.no_grad():
                inputs = chemberta_tokenizer(smiles, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
                embedding = chemberta_model(**inputs).last_hidden_state[0, 0, :].cpu().numpy()
            chemberta_embeddings.append(embedding)
        except: chemberta_embeddings.append(None)
    lg.setLevel(RDLogger.INFO)

    valid_indices = [i for i, (r, e) in enumerate(zip(rdkit_features, chemberta_embeddings)) if r is not None and e is not None]
    y_clean, df_clean = y[valid_indices], df.iloc[valid_indices].reset_index(drop=True)
    X_fp = np.vstack([rdkit_features[i]['fingerprint'] for i in valid_indices])
    desc_df = pd.DataFrame([rdkit_features[i]['descriptors'] for i in valid_indices]).reindex(columns=OPTIMIZED_DESCRIPTORS)
    imputer, scaler = SimpleImputer(strategy='mean'), StandardScaler()
    X_num_scaled = scaler.fit_transform(imputer.fit_transform(desc_df))
    X_rdkit = np.concatenate([X_fp, X_num_scaled], axis=1)
    X_embed = np.vstack([chemberta_embeddings[i] for i in valid_indices])

    print(f"\nKept {len(y_clean)} metabolites. RDKit features: {X_rdkit.shape}, ChemBERTa features: {X_embed.shape}")
    return X_rdkit, X_embed, y_clean, df_clean, mlb, imputer, scaler

# ==============================================================================
# STEP 4: DEFINITION OF EXPERT & META MODELS
# ==============================================================================
print("--- Defining expert models (XGBoost Ensemble, MLP) and meta-model ---")
# --- XGBoost Classes ---
class PinnacleEnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators: List[Any]): self.estimators_ = estimators
    def predict(self, X, threshold=0.5): return (self.predict_proba(X) > threshold).astype(int)
    def predict_proba(self, X): return np.hstack([est.predict_proba(X)[:, 1].reshape(-1, 1) for est in self.estimators_])

class OptunaEarlyStoppingCallback:
    def __init__(self, early_stopping_rounds: int):
        self._iter, self.best_value, self.early_stopping_rounds = 0, -float('inf'), early_stopping_rounds
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if trial.value is not None and trial.value > self.best_value: self.best_value, self._iter = trial.value, 0
        else: self._iter += 1
        if self._iter >= self.early_stopping_rounds: study.stop()

# --- MLP Expert Definition ---
class MLPEmbeddingExpert(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPEmbeddingExpert, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
                                     nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
                                     nn.Linear(256, num_classes))
    def forward(self, x): return self.network(x)

def train_mlp_expert(X_train_embed, y_train, X_val_embed, y_val, binarizer, device):
    print("\n--- Training Expert B: MLP on ChemBERTa Embeddings (Optimizing for MACRO F1) ---")
    model = MLPEmbeddingExpert(X_train_embed.shape[1], len(binarizer.classes_)).to(device)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_embed), torch.FloatTensor(y_train)), batch_size=MLP_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_embed), torch.FloatTensor(y_val)), batch_size=MLP_BATCH_SIZE*2)
    optimizer = optim.AdamW(model.parameters(), lr=MLP_LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=3)
    best_val_f1 = -1
    for epoch in range(MLP_EPOCHS):
        model.train()
        for features, labels in train_loader:
            outputs = model(features.to(device)); loss = criterion(outputs, labels.to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                all_preds.append(torch.sigmoid(model(features.to(device))).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        val_f1_macro = f1_score(np.vstack(all_labels), (np.vstack(all_preds) > 0.5).astype(int), average='macro', zero_division=0)
        print(f"Epoch {epoch+1}/{MLP_EPOCHS}, Val Macro F1: {val_f1_macro:.6f}")
        scheduler.step(val_f1_macro)
        if val_f1_macro > best_val_f1:
            best_val_f1, _ = val_f1_macro, torch.save(model.state_dict(), 'best_mlp_expert.pth')
    print(f"Best MLP Validation Macro F1: {best_val_f1:.6f}. Model saved.")
    model.load_state_dict(torch.load('best_mlp_expert.pth'))
    return model

def get_mlp_predictions(model, X_embed, device):
    model.eval()
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_embed)), batch_size=MLP_BATCH_SIZE*2)
    all_probas = []
    with torch.no_grad():
        for features in loader: all_probas.append(torch.sigmoid(model(features[0].to(device))).cpu().numpy())
    return np.vstack(all_probas)

# --- Meta-Model Definition ---
class StackingMetaModel:
    def __init__(self, num_classes):
        self.models = [LogisticRegression(solver='liblinear', random_state=RANDOM_SEED, class_weight='balanced') for _ in range(num_classes)]
    def fit(self, X, y):
        for i, model in enumerate(self.models): model.fit(X, y[:, i])
    def predict_proba(self, X):
        return np.hstack([model.predict_proba(X)[:, 1].reshape(-1, 1) for model in self.models])
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)

# ==============================================================================
# STEP 5: EVALUATION AND ANALYSIS FUNCTIONS
# ==============================================================================
print("--- Defining supreme evaluation and utility functions ---")
def print_full_evaluation_report(y_true, y_pred, y_pred_proba, binarizer):
    print("\n\n" + "="*80 + "\n---                  SUPREME PERFORMANCE EVALUATION REPORT                  ---\n" + "="*80)
    print("\n" + "="*40 + f"\n      Global Correctness Metrics\n" + "="*40)
    print(f"| {'Metric':<20} | {'Score':<15} |\n|{'-'*22}|{'-'*17}|")
    print(f"| {'Subset Accuracy':<20} | {accuracy_score(y_true, y_pred):<15.6f} |")
    print(f"| {'Hamming Loss':<20} | {hamming_loss(y_true, y_pred):<15.6f} |")
    print("="*40 + "\n\n" + "="*53 + f"\n            Label-wise Performance Metrics\n" + "="*53)
    print(f"| {'Metric':<20} | {'Micro':<15} | {'Macro':<15} |\n|{'-'*22}|{'-'*17}|{'-'*17}|")
    metrics = {"F1 Score": f1_score, "Precision": precision_score, "Recall": recall_score, "ROC AUC": roc_auc_score}
    for name, func in metrics.items():
        if name == "ROC AUC": micro, macro = func(y_true, y_pred_proba, average='micro'), func(y_true, y_pred_proba, average='macro')
        else: micro, macro = func(y_true, y_pred, average='micro', zero_division=0), func(y_true, y_pred, average='macro', zero_division=0)
        print(f"| {name:<20} | {micro:<15.6f} | {macro:<15.6f} |")
    print("="*53)
    print("\n\n--- Detailed Class-wise Performance Report ---")
    report = classification_report(y_true, y_pred, target_names=binarizer.classes_, output_dict=True, zero_division=0)
    print("\n" + "="*70 + f"\n| {'Class':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<7} |\n|{'-'*32}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*9}|")
    for class_name in binarizer.classes_:
        metrics = report[class_name]
        print(f"| {class_name:<30} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | {metrics['f1-score']:<10.4f} | {int(metrics['support']):<7} |")
    print("="*70)

def find_optimal_threshold(meta_model, X_meta_val, y_val):
    print("\n--- Optimizing Final Prediction Threshold for SUBSET ACCURACY ---")
    y_pred_proba_val = meta_model.predict_proba(X_meta_val)
    best_score, best_threshold = -1, 0.5
    for threshold in tqdm(np.arange(0.2, 0.8, 0.01), desc="Finding Best Threshold"):
        score = accuracy_score(y_val, (y_pred_proba_val > threshold).astype(int))
        if score > best_score: best_score, best_threshold = score, threshold
    print(f"Optimal threshold found: {best_threshold:.2f} (achieved meta-validation subset accuracy: {best_score:.6f})")
    return best_threshold

def display_class_distribution(y: np.ndarray, binarizer: MultiLabelBinarizer):
    print("\n--- Biofluid Class Distribution Analysis ---")
    counts = np.sum(y, axis=0)
    info = sorted(zip(binarizer.classes_, counts), key=lambda x: x[1], reverse=True)
    print("\n" + "="*56 + f"\n| {'Biofluid Class':<30} | {'Count':<10} | {'Percentage (%)':<10} |\n|{'-'*32}|{'-'*12}|{'-'*16}|")
    for name, count in info: print(f"| {name:<30} | {count:<10} | {(count / len(y)) * 100:<14.2f} |")
    print("="*56)

def display_representative_predictions(df_test, y_test, y_pred_proba, binarizer, optimal_threshold, n_samples=10):
    print(f"\n--- Representative Test Predictions (n={n_samples}) ---")
    random_indices = random.sample(range(len(df_test)), min(n_samples, len(df_test)))
    for i, idx in enumerate(random_indices):
        true_labels, sample_probas = binarizer.classes_[y_test[idx] == 1], y_pred_proba[idx]
        pred_labels = binarizer.classes_[sample_probas >= optimal_threshold]
        print(f"\n--- Sample #{i+1} (SMILES: {df_test.iloc[idx]['smiles']}) ---")
        print(f"  - True Biofluids    : {', '.join(true_labels) if len(true_labels) > 0 else 'None'}")
        print(f"  - Predicted Biofluids : {', '.join(pred_labels) if len(pred_labels) > 0 else 'None'}")
        print("  - Confidence Scores :")
        for j, fluid in enumerate(binarizer.classes_):
            print(f"    - {fluid:<28}: {sample_probas[j]:.4f}{' [*]' if sample_probas[j] >= optimal_threshold else ''}")

# --- NEWLY RESTORED: Per-Class Feature Importance Analysis ---
def display_per_class_feature_importance(xgb_expert_model: PinnacleEnsembleModel, binarizer: MultiLabelBinarizer, descriptors: List[str], fp_bits: int):
    """Calculates and displays feature importance for the structural XGBoost expert model."""
    print("\n\n" + "="*80 + "\n---             STRUCTURAL EXPERT (XGBoost) FEATURE IMPORTANCE             ---\n" + "="*80)
    for i, class_name in enumerate(binarizer.classes_):
        print("\n" + "="*50)
        print(f"       Feature Importance for Class: {class_name}")
        print("="*50)
        specialist_model = xgb_expert_model.estimators_[i]
        importances = specialist_model.feature_importances_
        
        fp_importance = np.sum(importances[:fp_bits])
        desc_importances = importances[fp_bits:]
        
        feature_map = {'Morgan_Fingerprints': fp_importance}
        for j, desc_name in enumerate(descriptors):
            feature_map[desc_name] = desc_importances[j]
            
        total_importance = sum(feature_map.values())
        if total_importance == 0:
            print("  - All feature importances are zero for this class.")
            continue
            
        sorted_features = sorted(
            [(name, (imp / total_importance) * 100) for name, imp in feature_map.items() if imp > 0],
            key=lambda x: x[1], reverse=True
        )
        print(f"| {'Feature':<30} | {'Contribution (%)':<15} |")
        print(f"|{'-'*32}|{'-'*17}|")
        for name, imp in sorted_features:
            print(f"| {name:<30} | {imp:<15.4f} |")
        print("="*50)

# ==============================================================================
# STEP 6: MAIN EXECUTION WORKFLOW
# ==============================================================================
if __name__ == "__main__":
    print("--- Starting The Supreme Stacked Ensemble Workflow ---")
    print(f"--- Loading pre-trained ChemBERTa model: '{CHEMBERTA_MODEL_NAME}' ---")
    chemberta_tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL_NAME)
    chemberta_model = AutoModel.from_pretrained(CHEMBERTA_MODEL_NAME).to(device)
    chemberta_model.eval()

    df_raw = load_and_prepare_data(MASTER_CSV_PATH, METABOLITES_TO_USE, TARGET_BIOFLUIDS)
    X_rdkit, X_embed, y, df_clean, binarizer, imputer, scaler = compute_all_features(df_raw, chemberta_tokenizer, chemberta_model, device)
    display_class_distribution(y, binarizer)

    indices = np.arange(X_rdkit.shape[0])
    train_full_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=RANDOM_SEED)
    train_idx, val_idx = train_test_split(train_full_idx, test_size=(0.15/0.85), random_state=RANDOM_SEED)
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    X_train_rdkit, X_val_rdkit, X_test_rdkit = X_rdkit[train_idx], X_rdkit[val_idx], X_rdkit[test_idx]
    X_train_embed, X_val_embed, X_test_embed = X_embed[train_idx], X_embed[val_idx], X_embed[test_idx]
    print(f"\nData splitting complete. Training: {len(train_idx)}, Validation: {len(val_idx)}, Test: {len(test_idx)}")

    print("\n--- Training Expert A: XGBoost on RDKit Features (ATO Strategy) ---")
    def xgb_objective(trial, X_train, y_train_single, X_val, y_val_single):
        params = {'tree_method': 'gpu_hist', 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'random_state': RANDOM_SEED,
                  'n_estimators': trial.suggest_int('n_estimators', 500, 3000, step=100), 'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.2, log=True),
                  'max_depth': trial.suggest_int('max_depth', 6, 16), 'subsample': trial.suggest_float('subsample', 0.5, 0.9),
                  'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9), 'gamma': trial.suggest_float('gamma', 1e-2, 30.0, log=True),
                  'min_child_weight': trial.suggest_int('min_child_weight', 1, 40), 'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 20.0, log=True),
                  'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 20.0, log=True)}
        n_pos = np.sum(y_train_single == 1); scale = (len(y_train_single) - n_pos) / n_pos if n_pos > 0 else 1
        model = xgb.XGBClassifier(**params, scale_pos_weight=scale).fit(X_train, y_train_single)
        return f1_score(y_val_single, model.predict(X_val), average='binary', zero_division=0)

    time_budgets = {'Blood': 0.30, 'Urine': 0.14, 'Feces': 0.14, 'Saliva': 0.14, 'Cerebrospinal Fluid (CSF)': 0.14, 'Sweat': 0.14}
    xgb_specialist_models, foundational_params = [], {}
    print(f"-- Phase 1: Training Foundational Model on Blood ({time_budgets['Blood']*TOTAL_OPTUNA_TIMEOUT_SECONDS:.0f}s) --")
    blood_idx = binarizer.classes_.tolist().index('Blood')
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda t: xgb_objective(t, X_train_rdkit, y_train[:, blood_idx], X_val_rdkit, y_val[:, blood_idx]),
                   timeout=time_budgets['Blood']*TOTAL_OPTUNA_TIMEOUT_SECONDS, callbacks=[OptunaEarlyStoppingCallback(XGB_OPTUNA_EARLY_STOPPING_ROUNDS)])
    foundational_params = study.best_params
    print(f"  - Best Foundational F1-Score on Val: {study.best_value:.6f}")

    print("\n-- Phase 2: Specializing Models using Transfer Learning --")
    for i, class_name in enumerate(binarizer.classes_):
        budget = time_budgets[class_name] * TOTAL_OPTUNA_TIMEOUT_SECONDS
        print(f"-- Optimizing for Class: {class_name} ({budget:.0f}s budget) --")
        study = optuna.create_study(direction='maximize')
        study.enqueue_trial(foundational_params)
        study.optimize(lambda t: xgb_objective(t, X_train_rdkit, y_train[:, i], X_val_rdkit, y_val[:, i]), timeout=budget)
        best_params = study.best_params
        print(f"  - Best F1 for {class_name}: {study.best_value:.6f}")
        y_train_full_single = y[train_full_idx, i]
        n_pos = np.sum(y_train_full_single == 1); scale = (len(y_train_full_single) - n_pos) / n_pos if n_pos > 0 else 1
        specialist = xgb.XGBClassifier(**best_params, scale_pos_weight=scale).fit(X_rdkit[train_full_idx], y_train_full_single)
        xgb_specialist_models.append(specialist)
    xgb_expert_model = PinnacleEnsembleModel(estimators=xgb_specialist_models)
    print("--- Expert A (XGBoost) trained successfully. ---")

    mlp_expert_model = train_mlp_expert(X_train_embed, y_train, X_val_embed, y_val, binarizer, device)

    print("\n--- Training Stage 2: Meta-Model on Expert Predictions ---")
    xgb_val_preds = xgb_expert_model.predict_proba(X_val_rdkit)
    mlp_val_preds = get_mlp_predictions(mlp_expert_model, X_val_embed, device)
    X_meta_train = np.concatenate([xgb_val_preds, mlp_val_preds], axis=1)
    meta_model = StackingMetaModel(num_classes=y_val.shape[1])
    meta_model.fit(X_meta_train, y_val)
    print("--- Meta-Model trained successfully. ---")

    xgb_test_preds = xgb_expert_model.predict_proba(X_test_rdkit)
    mlp_test_preds = get_mlp_predictions(mlp_expert_model, X_test_embed, device)
    X_meta_test = np.concatenate([xgb_test_preds, mlp_test_preds], axis=1)
    optimal_threshold = find_optimal_threshold(meta_model, X_meta_train, y_val)
    y_pred_proba_final = meta_model.predict_proba(X_meta_test)
    y_pred_final = (y_pred_proba_final > optimal_threshold).astype(int)

    print_full_evaluation_report(y_test, y_pred_final, y_pred_proba_final, binarizer)
    display_representative_predictions(df_clean.iloc[test_idx], y_test, y_pred_proba_final, binarizer, optimal_threshold)
    display_per_class_feature_importance(xgb_expert_model, binarizer, OPTIMIZED_DESCRIPTORS, FINGERPRINT_BITS) # <-- RESTORED CALL

    print("\n--- Saving All Artifacts for Unimpeachable Reproducibility ---")
    model_package = {'xgb_expert': xgb_expert_model, 'mlp_expert_state_dict': mlp_expert_model.state_dict(), 'meta_model': meta_model, 'binarizer': binarizer,
                     'rdkit_imputer': imputer, 'rdkit_scaler': scaler, 'optimal_threshold': optimal_threshold, 'optimized_descriptors': OPTIMIZED_DESCRIPTORS}
    with open('supreme_stacked_ensemble.pkl', 'wb') as f: pickle.dump(model_package, f)
    df_clean.to_csv("complete_processed_dataset.csv", index=False)
    with zipfile.ZipFile(OUTPUT_ARCHIVE_NAME, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write('supreme_stacked_ensemble.pkl'); zf.write("complete_processed_dataset.csv")
    os.remove('supreme_stacked_ensemble.pkl'); os.remove("complete_processed_dataset.csv")
    print(f"All artifacts saved to '{OUTPUT_ARCHIVE_NAME}'.")
    try: files.download(OUTPUT_ARCHIVE_NAME)
    except NameError: print(f"\nTo download, locate '{OUTPUT_ARCHIVE_NAME}' in the file browser.")
    print("\n--- The Supreme Stacked Ensemble Workflow Execution Complete ---")
