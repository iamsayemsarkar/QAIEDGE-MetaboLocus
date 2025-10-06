# ==============================================================================
# XGBoost model for predicting metabolite biofluid distribution.
# VERSION: 1.0
#
# PURPOSE:
#   Build an XGBoost model to predict metabolite biofluid distribution
#   using features derived from SMILES strings. This script includes
#   hyperparameter optimization constrained to a 15-minute training window.
#
# KEY FEATURES:
#   1. SMILES-BASED FEATURE ENGINEERING: Derives physico-chemical descriptors
#      and Morgan fingerprints directly from SMILES strings.
#   2. IMBALANCE HANDLING: Utilizes a multi-output wrapper that calculates
#      `scale_pos_weight` for each target biofluid independently.
#   3. COMPREHENSIVE EVALUATION: Reports F1 score, subset accuracy, Hamming loss,
#      precision, recall, and micro-averaged ROC-AUC score.
#   4. HYPERPARAMETER OPTIMIZATION: Uses Optuna for hyperparameter search.
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
# STEP 2: IMPORTS & GLOBAL CONFIGURATION
# ==============================================================================
print("--- Importing libraries and setting up configuration ---")

import os
import pickle
import warnings
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from lxml import etree
from tqdm.notebook import tqdm

import xgboost as xgb
import optuna

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, precision_score, recall_score, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

# --- Global Constants ---
DRIVE_XML_FILE_PATH: str = '/content/drive/My Drive/hmdb/hmdb_metabolites.xml'
MODEL_OUTPUT_FILENAME: str = 'xgb_biofluid_model_v6.0_15min.pkl'
MAX_METABOLITES_TO_PARSE: int = 30000
TARGET_BIOFLUIDS: List[str] = [
    'Blood', 'Urine', 'Saliva', 'Cerebrospinal Fluid (CSF)', 'Feces', 'Sweat'
]
OPTUNA_TIMEOUT_SECONDS: int = 900
RANDOM_SEED: int = 42

# --- Feature Engineering Configuration ---
COMPUTED_NUMERICAL_FEATURES: List[str] = [
    'MolWt', 'ExactMolWt', 'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors',
    'NumValenceElectrons', 'NumAromaticRings', 'NumSaturatedRings',
    'NumAliphaticRings', 'RingCount', 'TPSA', 'MolLogP', 'MolMR',
    'FractionCSP3', 'NumRotatableBonds', 'qed'
]
FINGERPRINT_BITS: int = 2048

# --- Environment Setup ---
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(RANDOM_SEED)

try:
    from google.colab import drive
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
    """Parses HMDB XML for SMILES strings and biofluid locations."""
    print(f"Parsing up to {max_records} records from '{os.path.basename(xml_path)}'...")
    NAMESPACE, TAGS = 'http://www.hmdb.ca', {
        'metabolite': '{http://www.hmdb.ca}metabolite', 'smiles': '{http://www.hmdb.ca}smiles',
        'locations': '{http://www.hmdb.ca}biospecimen_locations', 'specimen': '{http://www.hmdb.ca}biospecimen'
    }
    parsed_data = []
    context = etree.iterparse(xml_path, events=('end',), tag=TAGS['metabolite'])
    for _, elem in tqdm(context, desc="Processing Metabolites", total=max_records):
        smiles_elem = elem.find(TAGS['smiles'])
        if smiles_elem is None or not smiles_elem.text: elem.clear(); continue
        locs_elem = elem.find('.//' + TAGS['locations'])
        if locs_elem is not None:
            biofluids = [spec.text for spec in locs_elem.findall(TAGS['specimen'])]
            if biofluids:
                parsed_data.append({'smiles': smiles_elem.text, 'biofluids': list(set(biofluids))})
        elem.clear()
        while elem.getprevious() is not None: del elem.getparent()[0]
        if len(parsed_data) >= max_records: break
    return parsed_data

def compute_features_from_smiles(smiles: str) -> Optional[Dict[str, Any]]:
    """Computes Morgan fingerprints and numerical descriptors from a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        
        # 1. Fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FINGERPRINT_BITS)
        fp_array = np.array(list(fp.ToBitString()), dtype=np.uint8)
        
        # 2. Numerical Descriptors
        descriptors = {
            'MolWt': Descriptors.MolWt(mol), 'ExactMolWt': Descriptors.ExactMolWt(mol),
            'HeavyAtomCount': Descriptors.HeavyAtomCount(mol), 'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumHDonors': Descriptors.NumHDonors(mol), 'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol), 'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol), 'RingCount': Descriptors.RingCount(mol),
            'TPSA': Descriptors.TPSA(mol), 'MolLogP': Descriptors.MolLogP(mol), 'MolMR': Descriptors.MolMR(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol), 'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'qed': Descriptors.qed(mol) # Quantitative Estimate of Drug-likeness
        }
        return {'fingerprint': fp_array, 'descriptors': descriptors}
    except Exception:
        return None

def preprocess_and_engineer_features(data: List[Dict[str, Any]], target_biofluids: List[str]) -> Tuple:
    """Creates a feature matrix and target labels from parsed data."""
    print("\nPreprocessing data and engineering features from SMILES...")
    df = pd.DataFrame(data)
    df['biofluids'] = df['biofluids'].apply(lambda lst: [b for b in lst if b in target_biofluids])
    df = df[df['biofluids'].apply(len) > 0].reset_index(drop=True)
    if df.empty: return None, None, None, None, None

    mlb = MultiLabelBinarizer(classes=target_biofluids); y = mlb.fit_transform(df['biofluids'])
    
    print("Computing fingerprints and molecular descriptors...")
    lg = RDLogger.logger(); lg.setLevel(RDLogger.CRITICAL)
    feature_results = [compute_features_from_smiles(s) for s in tqdm(df['smiles'], desc="Computing Features")]
    lg.setLevel(RDLogger.INFO)
    
    valid_indices = [i for i, res in enumerate(feature_results) if res is not None]
    if not valid_indices: return None, None, None, None, None
    
    y_clean = y[valid_indices]
    
    X_fp = np.vstack([feature_results[i]['fingerprint'] for i in valid_indices])
    
    desc_df = pd.DataFrame([feature_results[i]['descriptors'] for i in valid_indices]).reindex(columns=COMPUTED_NUMERICAL_FEATURES)
    X_num = desc_df.values
    
    print(f"Kept {len(y_clean)} metabolites after feature computation.")
    
    imputer = SimpleImputer(strategy='mean'); scaler = StandardScaler()
    X_num_imputed = imputer.fit_transform(X_num); X_num_scaled = scaler.fit_transform(X_num_imputed)
    
    X_hybrid = np.concatenate([X_fp, X_num_scaled], axis=1)
    print(f"Successfully created hybrid feature matrix with shape: {X_hybrid.shape}")
    
    return X_hybrid, y_clean, mlb, imputer, scaler

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
    for name, score in metrics.items(): print(f"  - {name:<20}: {score:.4f}")

# ==============================================================================
# STEP 4: CUSTOM XGBOOST WRAPPER
# ==============================================================================
class XGBoostMultiOutputWrapper(BaseEstimator, ClassifierMixin):
    """A custom wrapper for multi-label classification with dynamic `scale_pos_weight`."""
    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.estimators_ = []
    def fit(self, X, y):
        self.estimators_ = []
        for i in range(y.shape[1]):
            y_col = y[:, i]
            scale_pos_weight = np.sum(y_col == 0) / np.sum(y_col == 1) if np.sum(y_col == 1) > 0 else 1
            estimator = xgb.XGBClassifier(**self.xgb_params, scale_pos_weight=scale_pos_weight)
            estimator.fit(X, y_col)
            self.estimators_.append(estimator)
        return self
    def predict(self, X):
        return np.hstack([est.predict(X).reshape(-1, 1) for est in self.estimators_])
    def predict_proba(self, X):
        # Returns probabilities for the positive class (class 1) for each label.
        return np.hstack([est.predict_proba(X)[:, 1].reshape(-1, 1) for est in self.estimators_])

# ==============================================================================
# STEP 5: OPTUNA HYPERPARAMETER OPTIMIZATION
# ==============================================================================
def objective(trial, X_train, y_train, X_val, y_val):
    """Defines the Optuna objective function for hyperparameter search."""
    param_grid = {
        'tree_method': 'gpu_hist', 'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'random_state': RANDOM_SEED, 'n_jobs': -1,
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 6, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 20.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
    }
    model = XGBoostMultiOutputWrapper(**param_grid); model.fit(X_train, y_train)
    return f1_score(y_val, model.predict(X_val), average='micro', zero_division=0)

# ==============================================================================
# STEP 6: MAIN EXECUTION WORKFLOW
# ==============================================================================
if __name__ == "__main__":
    print("--- Starting XGBoost Execution Workflow ---")

    if not os.path.exists(DRIVE_XML_FILE_PATH):
        print(f"CRITICAL ERROR: File not found at '{DRIVE_XML_FILE_PATH}'. Halting.")
    else:
        parsed_data = parse_hmdb_for_smiles_and_biofluids(DRIVE_XML_FILE_PATH, MAX_METABOLITES_TO_PARSE)
        if not parsed_data:
            print("\nCRITICAL ERROR: No data parsed. Halting.")
        else:
            X, y, binarizer, imputer, scaler = preprocess_and_engineer_features(parsed_data, TARGET_BIOFLUIDS)
            if X is not None:
                # Store exact dataset sizes
                total_samples = X.shape[0]
                test_set_size = int(total_samples * 0.15)
                val_set_size_from_train_full = int((total_samples - test_set_size) * (0.15 / 0.85))
                
                X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_set_size, random_state=RANDOM_SEED)
                X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_set_size_from_train_full, random_state=RANDOM_SEED)

                print("\nData splitting complete:")
                print(f"  - Total Valid Metabolites: {total_samples}")
                print(f"  - Training samples   : {X_train.shape[0]}")
                print(f"  - Validation samples : {X_val.shape[0]}")
                print(f"  - Final Test samples : {X_test.shape[0]}\n")

                print(f"--- Starting 15-Minute Hyperparameter Search with Optuna ---")
                study = optuna.create_study(direction='maximize')
                study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), timeout=OPTUNA_TIMEOUT_SECONDS)

                print("\n--- Hyperparameter Search Complete ---")
                print(f"Best F1 Score on Validation Set: {study.best_value:.4f}")
                print("Best parameters found:")
                best_params = study.best_params
                for key, value in best_params.items(): print(f"  - {key}: {value}")

                print("\nTraining the final model on all available training data with best parameters...")
                final_params = {'tree_method': 'gpu_hist', 'objective': 'binary:logistic', 'random_state': RANDOM_SEED, 'n_jobs': -1}
                final_params.update(best_params)

                final_model = XGBoostMultiOutputWrapper(**final_params)
                final_model.fit(X_train_full, y_train_full)
                print("Final model training complete.")

                y_pred_test = final_model.predict(X_test)
                y_pred_proba_test = final_model.predict_proba(X_test)
                print_full_evaluation_report(y_test, y_pred_test, y_pred_proba_test, "Optimized XGBoost Model")

                print("\n--- Example Prediction Confidences ---")
                for i in range(min(5, len(X_test))):
                    print(f"\nSample #{i+1}:")
                    true_labels = binarizer.inverse_transform(y_test[i].reshape(1, -1))[0]
                    print(f"  - True Biofluids  : {true_labels if true_labels else 'None'}")
                    confidences = y_pred_proba_test[i]
                    pred_labels = [TARGET_BIOFLUIDS[j] for j, conf in enumerate(confidences) if conf > 0.5]
                    print(f"  - Predicted Biofluids: {tuple(pred_labels) if pred_labels else 'None'}")
                    print("  - Prediction Confidences:")
                    for j, fluid in enumerate(TARGET_BIOFLUIDS):
                        print(f"    - {fluid:<28}: {confidences[j]:.4f}")

                model_package = {
                    'model': final_model, 'binarizer': binarizer, 'imputer': imputer, 'scaler': scaler,
                    'computed_numerical_features_order': COMPUTED_NUMERICAL_FEATURES,
                    'fingerprint_bits': FINGERPRINT_BITS
                }
                with open(MODEL_OUTPUT_FILENAME, 'wb') as f:
                    pickle.dump(model_package, f)
                print(f"\nModel package saved successfully to '{MODEL_OUTPUT_FILENAME}'.")

                try:
                    from google.colab import files
                    files.download(MODEL_OUTPUT_FILENAME)
                except (ImportError, NameError):
                    print(f"\nTo download the model ('{MODEL_OUTPUT_FILENAME}'), please locate it in the file browser.")

    print("\n--- XGBoost Workflow Execution Complete ---")