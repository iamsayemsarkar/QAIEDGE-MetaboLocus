# ==============================================================================
# The Ultimate XGBoost Pipeline for Small Datasets
# VERSION:  1.0
# PURPOSE:
#   To achieve the absolute theoretical peak predictive performance on a small,
#   high-stakes dataset of 1875 metabolites. This script implements a
#   precision-guided strategy: a curated "Critical Core" feature set, aggressive
#   regularization, and a focused optimization protocol to deliver unrivaled
#   performance and prevent overfitting.
#
# KEY FEATURES:
#   1. CRITICAL CORE FEATURE ENGINEERING: Utilizes a lean set of 10 fundamental
#      descriptors and a 1024-bit fingerprint to maximize signal and minimize noise.
#   2. AGGRESSIVE REGULARIZATION: The hyperparameter search space is strategically
#      constrained to force the discovery of a simple, robust, and highly
#      generalizable model.
#   3. TWO-STAGE GOAL-ALIGNED OPTIMIZATION: A focused 20-minute search for Subset
#      Accuracy is followed by dynamic threshold optimization to maximize all
#      evaluation metrics.
# ==============================================================================

# ==============================================================================
# STEP 1: INSTALL DEPENDENCIES
# ==============================================================================

print("--- Installing required libraries ---")
!pip install xgboost --quiet
!pip install lxml --quiet
!pip install rdkit --quiet
!pip install optuna --quiet
print("Libraries installed.\n")

# ==============================================================================
# STEP 2: IMPORTS & ULTIMATE PRECISION-GUIDED CONFIGURATION
# ==============================================================================

print("--- Importing libraries and setting up the ultimate precision-guided configuration ---")

import os
import pickle
import warnings
import zipfile
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from lxml import etree
from tqdm.notebook import tqdm

import xgboost as xgb
import optuna

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Fragments

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, precision_score, recall_score, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

# --- Global Constants: Calibrated for the 1875 Metabolite Dataset ---

DRIVE_XML_FILE_PATH: str = '/content/drive/My Drive/hmdb/hmdb_metabolites.xml'
# Set the exact number of metabolites for this high-precision run.
MAX_METABOLITES_TO_PARSE: int = 1875
OUTPUT_ARCHIVE_NAME: str = 'XGB_Ultimate_Run_1875.zip'

TARGET_BIOFLUIDS: List[str] = [
    'Blood', 'Urine', 'Saliva', 'Cerebrospinal Fluid (CSF)', 'Feces', 'Sweat'
]
RANDOM_SEED: int = 42

# --- Optuna Configuration: 20-Minute Focused Search Protocol ---

OPTUNA_TIMEOUT_SECONDS: int = 1200  # 20 minutes
OPTUNA_EARLY_STOPPING_ROUNDS: int = 30 # Reduced patience for a faster, focused search

# --- Feature Engineering Configuration: The "Critical Core 10" Descriptors ---

COMPUTED_NUMERICAL_FEATURES: List[str] = [
    'MolWt', 'HeavyAtomCount', 'MolLogP', 'TPSA', 'NumHDonors',
    'NumHAcceptors', 'NumRotatableBonds', 'RingCount', 'FractionCSP3', 'qed'
]
FINGERPRINT_BITS: int = 1024 # Leaner fingerprint for a smaller dataset

# --- Environment Setup ---

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(RANDOM_SEED)

try:
    from google.colab import drive, files
    drive.mount('/content/drive', force_remount=True)
    print("Google Drive mounted successfully.")
except ImportError:
    print("Could not mount Google Drive. Assuming non-Colab environment.")

print("Setup complete.\n")


# ==============================================================================
# STEP 3: WORKFLOW FUNCTIONS
# ==============================================================================

print("--- Defining workflow functions ---")

def parse_hmdb_for_smiles_and_biofluids(xml_path: str, max_records: int) -> List[Dict[str, Any]]:
    """
    Parses HMDB XML for metabolites that have both a SMILES string and at least
    one biofluid annotation, ensuring a high-quality initial dataset.
    """
    print(f"Parsing source XML to find {max_records} valid metabolites...")
    NAMESPACE, TAGS = 'http://www.hmdb.ca', {
        'metabolite': '{http://www.hmdb.ca}metabolite', 'smiles': '{http://www.hmdb.ca}smiles',
        'locations': '{http://www.hmdb.ca}biospecimen_locations', 'specimen': '{http://www.hmdb.ca}biospecimen'
    }
    parsed_data = []
    context = etree.iterparse(xml_path, events=('end',), tag=TAGS['metabolite'])
    for _, elem in tqdm(context, desc="Scanning Metabolite Records"):
        smiles_elem = elem.find(TAGS['smiles'])
        if smiles_elem is None or not smiles_elem.text:
            elem.clear(); continue
        locs_elem = elem.find('.//' + TAGS['locations'])
        if locs_elem is not None:
            biofluids = [spec.text for spec in locs_elem.findall(TAGS['specimen'])]
            if biofluids:
                parsed_data.append({'smiles': smiles_elem.text, 'biofluids': list(set(biofluids))})
        elem.clear()
        while elem.getprevious() is not None: del elem.getparent()[0]
        if len(parsed_data) >= max_records: break
    print(f"Found {len(parsed_data)} metabolites meeting quality criteria.")
    return parsed_data

def compute_features_from_smiles(smiles: str) -> Optional[Dict[str, Any]]:
    """
    Computes Morgan fingerprints and the "Critical Core 10" numerical descriptors.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FINGERPRINT_BITS)
        fp_array = np.array(list(fp.ToBitString()), dtype=np.uint8)
        descriptors = {
            'MolWt': Descriptors.MolWt(mol), 'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
            'MolLogP': Descriptors.MolLogP(mol), 'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Descriptors.NumHDonors(mol), 'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol), 'RingCount': Descriptors.RingCount(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol), 'qed': Descriptors.qed(mol)
        }
        for key, value in descriptors.items():
            if not np.isfinite(value): descriptors[key] = np.nan
        return {'fingerprint': fp_array, 'descriptors': descriptors}
    except Exception:
        return None

def preprocess_and_engineer_features(data: List[Dict[str, Any]], target_biofluids: List[str]) -> Tuple:
    """Creates a feature matrix and target labels from parsed data."""
    print("\nPreprocessing data and engineering features from SMILES...")
    df = pd.DataFrame(data)
    df['biofluids'] = df['biofluids'].apply(lambda lst: [b for b in lst if b in target_biofluids])
    df = df[df['biofluids'].apply(len) > 0].reset_index(drop=True)
    mlb = MultiLabelBinarizer(classes=target_biofluids)
    y = mlb.fit_transform(df['biofluids'])
    print("Computing fingerprints and molecular descriptors...")
    lg = RDLogger.logger(); lg.setLevel(RDLogger.CRITICAL)
    feature_results = [compute_features_from_smiles(s) for s in tqdm(df['smiles'], desc="Computing Features")]
    lg.setLevel(RDLogger.INFO)
    valid_indices = [i for i, res in enumerate(feature_results) if res is not None]
    if not valid_indices:
        print("\nCRITICAL ERROR: Feature computation failed for all parsed metabolites.")
        return None, None, None, None, None, None
    y_clean = y[valid_indices]
    df_clean = df.iloc[valid_indices].reset_index(drop=True)
    X_fp = np.vstack([feature_results[i]['fingerprint'] for i in valid_indices])
    desc_df = pd.DataFrame([feature_results[i]['descriptors'] for i in valid_indices]).reindex(columns=COMPUTED_NUMERICAL_FEATURES)
    print(f"Kept {len(y_clean)} metabolites after final feature validation.")
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_num_imputed = imputer.fit_transform(desc_df)
    X_num_scaled = scaler.fit_transform(X_num_imputed)
    X_hybrid = np.concatenate([X_fp, X_num_scaled], axis=1)
    print(f"Successfully created hybrid feature matrix with shape: {X_hybrid.shape}")
    return X_hybrid, y_clean, df_clean, mlb, imputer, scaler

def print_full_evaluation_report(y_true, y_pred, y_pred_proba, model_name):
    """Calculates and prints multi-label evaluation metrics."""
    print(f"\n--- {model_name} Performance on Test Set ---")
    metrics = {
        "F1 Score (Micro)": f1_score(y_true, y_pred, average='micro', zero_division=0),
        "Subset Accuracy": accuracy_score(y_true, y_pred),
        "Hamming Loss": hamming_loss(y_true, y_pred),
        "Precision (Micro)": precision_score(y_true, y_pred, average='micro', zero_division=0),
        "Recall (Micro)": recall_score(y_true, y_pred, average='micro', zero_division=0),
        "ROC AUC (Micro)": roc_auc_score(y_true, y_pred_proba, average='micro')
    }
    for name, score in metrics.items():
        print(f"  - {name:<20}: {score:.6f}")

# ==============================================================================
# STEP 4: CUSTOM WRAPPER, CALLBACKS, AND NEW OPTIMIZATION FUNCTIONS
# ==============================================================================

class XGBoostMultiOutputWrapper(BaseEstimator, ClassifierMixin):
    """A custom wrapper for multi-label classification with dynamic scale_pos_weight."""
    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.estimators_ = []
    def fit(self, X, y):
        self.estimators_ = []
        for i in range(y.shape[1]):
            y_col = y[:, i]
            n_pos = np.sum(y_col == 1)
            scale_pos_weight = np.sum(y_col == 0) / n_pos if n_pos > 0 else 1
            estimator = xgb.XGBClassifier(**self.xgb_params, scale_pos_weight=scale_pos_weight)
            estimator.fit(X, y_col)
            self.estimators_.append(estimator)
        return self
    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return (probas > threshold).astype(int)
    def predict_proba(self, X):
        return np.hstack([est.predict_proba(X)[:, 1].reshape(-1, 1) for est in self.estimators_])

class OptunaEarlyStoppingCallback:
    """Callback to stop Optuna study if no improvement is seen for a given number of trials."""
    def __init__(self, early_stopping_rounds: int):
        self.early_stopping_rounds = early_stopping_rounds
        self._iter = 0
        self.best_value = -float('inf')
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if trial.value is not None and trial.value > self.best_value:
            self.best_value = trial.value
            self._iter = 0
        else:
            self._iter += 1
        if self._iter >= self.early_stopping_rounds:
            print(f"\n--- Stopping study early: No improvement in {self.early_stopping_rounds} trials. ---")
            study.stop()

def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Finds the optimal probability threshold to maximize subset accuracy on the validation set.
    """
    best_score = -1
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.01):
        preds = (y_pred_proba > threshold).astype(int)
        score = accuracy_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    print(f"Optimal threshold found: {best_threshold:.2f} (achieved validation subset accuracy: {best_score:.6f})")
    return best_threshold

# ==============================================================================
# STEP 5: ULTIMATE OPTUNA OBJECTIVE FUNCTION
# ==============================================================================

def objective(trial, X_train, y_train, X_val, y_val):
    """
    Defines the Optuna objective function, maximizing SUBSET ACCURACY with a
    search space aggressively regularized for small datasets.
    """
    param_grid = {
        'tree_method': 'gpu_hist', 'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'random_state': RANDOM_SEED, 'n_jobs': -1,
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.4, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'gamma': trial.suggest_float('gamma', 1e-3, 40.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 40),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
    }
    model = XGBoostMultiOutputWrapper(**param_grid)
    model.fit(X_train, y_train)
    preds = model.predict(X_val, threshold=0.5)
    return accuracy_score(y_val, preds)

# ==============================================================================
# STEP 6: MAIN EXECUTION WORKFLOW
# ==============================================================================

if __name__ == "__main__":
    print("--- Starting The Ultimate Precision-Guided XGBoost Workflow ---")
    if not os.path.exists(DRIVE_XML_FILE_PATH):
        print(f"CRITICAL ERROR: File not found at '{DRIVE_XML_FILE_PATH}'. Halting.")
    else:
        parsed_data = parse_hmdb_for_smiles_and_biofluids(DRIVE_XML_FILE_PATH, MAX_METABOLITES_TO_PARSE)
        if not parsed_data:
            print("\nCRITICAL ERROR: No data parsed. Halting.")
        else:
            X, y, df_clean, binarizer, imputer, scaler = preprocess_and_engineer_features(parsed_data, TARGET_BIOFLUIDS)
            if X is not None:
                try:
                    X_train_full, X_test, y_train_full, y_test, df_train_full, df_test = train_test_split(
                        X, y, df_clean, test_size=0.15, random_state=RANDOM_SEED, stratify=y
                    )
                    val_size_fraction = 0.15 / 0.85
                    X_train, X_val, y_train, y_val, df_train, df_val = train_test_split(
                        X_train_full, y_train_full, df_train_full, test_size=val_size_fraction, random_state=RANDOM_SEED, stratify=y_train_full
                    )
                    print("\nData splitting complete (stratified).")
                except ValueError:
                    warnings.warn("Stratified split failed. Falling back to non-stratified split.")
                    X_train_full, X_test, y_train_full, y_test, df_train_full, df_test = train_test_split(
                        X, y, df_clean, test_size=0.15, random_state=RANDOM_SEED
                    )
                    val_size_fraction = 0.15 / 0.85
                    X_train, X_val, y_train, y_val, df_train, df_val = train_test_split(
                        X_train_full, y_train_full, df_train_full, test_size=val_size_fraction, random_state=RANDOM_SEED
                    )
                    print("\nData splitting complete (non-stratified fallback).")

                print(f"  - Total Valid Metabolites: {X.shape[0]}")
                print(f"  - Training samples   : {X_train.shape[0]}")
                print(f"  - Validation samples : {X_val.shape[0]}")
                print(f"  - Final Test samples : {X_test.shape[0]}\n")

                print(f"--- Starting {OPTUNA_TIMEOUT_SECONDS/60:.0f}-Minute Focused Hyperparameter Search (Optimizing for Subset Accuracy) ---")
                early_stopping_callback = OptunaEarlyStoppingCallback(OPTUNA_EARLY_STOPPING_ROUNDS)
                study = optuna.create_study(direction='maximize')
                study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val),
                               timeout=OPTUNA_TIMEOUT_SECONDS, callbacks=[early_stopping_callback])

                print("\n--- Hyperparameter Search Complete ---")
                print(f"Best Subset Accuracy on Validation Set: {study.best_value:.6f}")
                print("Best parameters found for peak performance:")
                best_params = study.best_params
                for key, value in best_params.items():
                    print(f"  - {key}: {value}")

                print("\nTraining the final model on all available training data with best parameters...")
                final_params = {'tree_method': 'gpu_hist', 'objective': 'binary:logistic', 'random_state': RANDOM_SEED, 'n_jobs': -1}
                final_params.update(best_params)
                final_model = XGBoostMultiOutputWrapper(**final_params)
                final_model.fit(X_train_full, y_train_full)
                print("Final model training complete.")

                print("\n--- Stage 2: Optimizing Prediction Threshold ---")
                y_pred_proba_val = final_model.predict_proba(X_val)
                optimal_threshold = find_optimal_threshold(y_val, y_pred_proba_val)

                y_pred_proba_test = final_model.predict_proba(X_test)
                y_pred_test = final_model.predict(X_test, threshold=optimal_threshold)
                print_full_evaluation_report(y_test, y_pred_test, y_pred_proba_test, "The Ultimate Precision-Guided Model")

                print("\n--- Saving All Artifacts for Unimpeachable Reproducibility ---")
                model_package = {
                    'model': final_model, 'binarizer': binarizer, 'imputer': imputer, 'scaler': scaler,
                    'optimal_threshold': optimal_threshold,
                    'computed_numerical_features_order': COMPUTED_NUMERICAL_FEATURES,
                    'fingerprint_bits': FINGERPRINT_BITS
                }
                model_filename = 'xgb_ultimate_model_1875.pkl'
                with open(model_filename, 'wb') as f:
                    pickle.dump(model_package, f)

                train_df, val_df, test_df = df_train.copy(), df_val.copy(), df_test.copy()
                for i, fluid in enumerate(binarizer.classes_):
                    train_df[f'label_{fluid}'], val_df[f'label_{fluid}'], test_df[f'label_{fluid}'] = y_train[:, i], y_val[:, i], y_test[:, i]

                with zipfile.ZipFile(OUTPUT_ARCHIVE_NAME, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.write(model_filename)
                    print(f"  - Saved and compressed '{model_filename}'")
                    for name, df in [('train_dataset', train_df), ('validation_dataset', val_df), ('test_dataset', test_df)]:
                        csv_filename = f"{name}.csv"
                        df.to_csv(csv_filename, index=False)
                        zf.write(csv_filename)
                        os.remove(csv_filename)
                        print(f"  - Saved and compressed '{csv_filename}'")
                os.remove(model_filename)

                print(f"\nAll artifacts (model, datasets) have been saved to '{OUTPUT_ARCHIVE_NAME}'.")
                try:
                    files.download(OUTPUT_ARCHIVE_NAME)
                except (ImportError, NameError):
                    print(f"\nTo download the archive, please locate '{OUTPUT_ARCHIVE_NAME}' in the file browser.")

    print("\n--- The Ultimate Precision-Guided XGBoost Workflow Complete ---")