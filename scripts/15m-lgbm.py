# ==============================================================================
# SCRIPT: PREDICTING METABOLITE BIOFLUID DISTRIBUTION WITH LIGHTGBM
# VERSION: 4.0 (Hybrid Features, GPU-Accelerated, and Fully Optimized)
#
# PURPOSE:
#   To build a state-of-the-art LightGBM model that achieves maximum predictive
#   accuracy for metabolite biofluid distribution within a strict 15-minute
#   training window on a T4 GPU.
#
# WORKFLOW:
#   1. Robustly parses the HMDB XML database, extracting both chemical
#      structures (SMILES) and a rich set of physico-chemical properties.
#   2. Engineers a powerful hybrid feature set by combining 1024-bit ECFP4
#      fingerprints with the scaled numerical properties.
#   3. Executes a time-budgeted (15-minute) hyperparameter search with Optuna
#      to discover the optimal LightGBM configuration for this task.
#   4. Trains the final, optimized multi-output LightGBM classifier on the
#      entire training dataset using the best parameters found.
#   5. Conducts a rigorous final evaluation on an unseen test set using a
#      comprehensive suite of multi-label classification metrics.
#   6. Saves the trained model and all necessary data processors (binarizer,
#      scaler) as a single, production-ready package.
#
# EXPECTED RUNTIME:
#   Precisely 15 minutes for hyperparameter search, followed by a final
#   training and evaluation phase, on a standard cloud instance with a T4 GPU.
# ==============================================================================


# ==============================================================================
# STEP 1: INSTALL PREREQUISITE LIBRARIES
# ==============================================================================
print("--- STEP 1: INSTALLING PREREQUISITE LIBRARIES ---")
!pip install lxml --quiet
!pip install lightgbm --install-option=--gpu --quiet
!pip install rdkit --quiet
!pip install optuna --quiet
print("All libraries installed successfully.\n")


# ==============================================================================
# STEP 2: IMPORT LIBRARIES AND SET GLOBAL CONFIGURATION
# ==============================================================================
print("--- STEP 2: IMPORTING LIBRARIES & DEFINING CONFIGURATION ---")

# --- Core Python & Data Handling ---
import os
import pickle
import warnings
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from lxml import etree
from tqdm.notebook import tqdm

# --- Machine Learning ---
import lightgbm as lgb
import optuna

# --- Chemistry & Feature Engineering ---
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

# --- Model Evaluation & Utility ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, precision_score, recall_score

# --- Global Constants ---
DRIVE_XML_FILE_PATH: str = '/content/drive/My Drive/hmdb/hmdb_metabolites.xml'
MODEL_OUTPUT_FILENAME: str = 'lgbm_biofluid_model_v4.0_15min_final.pkl'
MAX_METABOLITES_TO_PARSE: int = 25000
TARGET_BIOFLUIDS: List[str] = [
    'Blood', 'Urine', 'Saliva', 'Cerebrospinal Fluid (CSF)', 'Feces', 'Sweat'
]
OPTUNA_TIMEOUT_SECONDS: int = 900  # 15 minutes
RANDOM_SEED: int = 42

# --- Feature Engineering Configuration ---
NUMERICAL_FEATURES: List[str] = [
    'average_molecular_weight', 'monisotopic_molecular_weight', 'logp', 'logs',
    'solubility', 'polar_surface_area', 'refractivity', 'polarizability',
    'rotatable_bond_count', 'acceptor_count', 'donor_count', 'physiological_charge'
]

# --- Environment Setup ---
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(RANDOM_SEED)

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    print("Google Drive mounted successfully.")
except ImportError:
    print("Could not mount Google Drive. Assuming non-Colab environment.")

print("Configuration and environment setup complete.\n")


# ==============================================================================
# STEP 3: DATA EXTRACTION AND FEATURE ENGINEERING FUNCTIONS
# ==============================================================================
print("--- STEP 3: DEFINING DATA PARSING & FEATURE ENGINEERING FUNCTIONS ---")

def parse_hmdb_with_hybrid_features(xml_path: str, max_records: int) -> List[Dict[str, Any]]:
    """
    Efficiently parses HMDB XML, extracting SMILES, biofluids, and a curated
    set of physico-chemical properties for hybrid feature generation.
    """
    print(f"Parsing up to {max_records} records from '{os.path.basename(xml_path)}'...")
    NAMESPACE = 'http://www.hmdb.ca'
    TAGS = {
        'metabolite': f'{{{NAMESPACE}}}metabolite', 'smiles': f'{{{NAMESPACE}}}smiles',
        'bio_props': f'{{{NAMESPACE}}}biological_properties',
        'locations': f'{{{NAMESPACE}}}biospecimen_locations',
        'specimen': f'{{{NAMESPACE}}}biospecimen',
        'pred_props': f'{{{NAMESPACE}}}predicted_properties',
        'prop': f'{{{NAMESPACE}}}property', 'kind': f'{{{NAMESPACE}}}kind',
        'value': f'{{{NAMESPACE}}}value'
    }
    parsed_data = []
    context = etree.iterparse(xml_path, events=('end',), tag=TAGS['metabolite'])

    for _, elem in tqdm(context, desc="Processing Metabolites", total=max_records):
        smiles_elem = elem.find(TAGS['smiles'])
        if smiles_elem is None or not smiles_elem.text:
            elem.clear(); continue

        biofluids = []
        bio_props_elem = elem.find(TAGS['bio_props'])
        if bio_props_elem is not None:
            locs_elem = bio_props_elem.find(TAGS['locations'])
            if locs_elem is not None:
                biofluids = [spec.text for spec in locs_elem.findall(TAGS['specimen'])]

        if not biofluids:
            elem.clear(); continue

        # Extract numerical features
        features = {}
        for child in elem:
            if child.tag.endswith('average_molecular_weight') and child.text:
                features['average_molecular_weight'] = float(child.text)
            elif child.tag.endswith('monisotopic_molecular_weight') and child.text:
                features['monisotopic_molecular_weight'] = float(child.text)

        pred_props_elem = elem.find(TAGS['pred_props'])
        if pred_props_elem is not None:
            for prop in pred_props_elem.findall(TAGS['prop']):
                kind_elem, value_elem = prop.find(TAGS['kind']), prop.find(TAGS['value'])
                if kind_elem is not None and value_elem is not None and kind_elem.text in NUMERICAL_FEATURES:
                    try:
                        features[kind_elem.text] = float(value_elem.text)
                    except (ValueError, TypeError):
                        continue

        parsed_data.append({
            'smiles': smiles_elem.text,
            'biofluids': list(set(biofluids)),
            'features': features
        })
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        if len(parsed_data) >= max_records:
            break
    return parsed_data

def smiles_to_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """Converts a SMILES string to a 1024-bit ECFP4 fingerprint as a numpy array."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            return np.array(list(fp.ToBitString()), dtype=int)
    except Exception:
        return None
    return None

def preprocess_and_create_hybrid_features(df: pd.DataFrame, target_biofluids: List[str]) -> Tuple:
    """
    Cleans data, generates a hybrid feature matrix of fingerprints and scaled
    numerical features, creates binary labels, and returns all necessary processors.
    """
    print("\nPreprocessing data and engineering hybrid features...")
    df['biofluids'] = df['biofluids'].apply(lambda lst: [b for b in lst if b in target_biofluids])
    df = df[df['biofluids'].apply(len) > 0].reset_index(drop=True)

    if df.empty:
        print("CRITICAL: No metabolites found with the specified target biofluids.")
        return None, None, None, None, None

    # 1. Binarize labels
    mlb = MultiLabelBinarizer(classes=target_biofluids)
    y = mlb.fit_transform(df['biofluids'])

    # 2. Generate fingerprints (handling errors)
    print("Generating molecular fingerprints...")
    lg = RDLogger.logger(); lg.setLevel(RDLogger.CRITICAL)
    df['fingerprint'] = df['smiles'].apply(smiles_to_fingerprint)
    lg.setLevel(RDLogger.INFO)
    
    # 3. Process numerical features
    numerical_df = pd.DataFrame([row.get('features', {}) for row in df.to_dict('records')])
    numerical_df = numerical_df[NUMERICAL_FEATURES] # Ensure consistent column order

    # 4. Filter out any rows where fingerprint generation failed
    valid_indices = df['fingerprint'].notna()
    if not valid_indices.any():
        print("CRITICAL: Fingerprint generation failed for all SMILES strings.")
        return None, None, None, None, None

    X_fp = np.vstack(df.loc[valid_indices, 'fingerprint'].values)
    X_num = numerical_df.loc[valid_indices].values
    y_clean = y[valid_indices]

    # 5. Impute and Scale numerical features
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_num_imputed = imputer.fit_transform(X_num)
    X_num_scaled = scaler.fit_transform(X_num_imputed)

    # 6. Combine into a single feature matrix
    X_hybrid = np.concatenate([X_fp, X_num_scaled], axis=1)

    print(f"Successfully created hybrid feature matrix for {X_hybrid.shape[0]} metabolites.")
    print(f"Feature matrix shape: {X_hybrid.shape}")
    
    return X_hybrid, y_clean, mlb, imputer, scaler

print("All workflow functions defined.\n")


# ==============================================================================
# STEP 4: OPTUNA HYPERPARAMETER OPTIMIZATION
# ==============================================================================
def objective(trial: optuna.trial.Trial, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """
    Optuna objective function to find the best LightGBM hyperparameters for
    multi-label classification, maximizing the F1 score.
    """
    param_grid = {
        "device": "gpu",
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 5, 16),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "random_state": RANDOM_SEED
    }

    lgbm = lgb.LGBMClassifier(**param_grid)
    model = MultiOutputClassifier(lgbm, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    f1 = f1_score(y_val, y_pred_val, average='micro', zero_division=0)
    return f1


# ==============================================================================
# STEP 5: MAIN EXECUTION WORKFLOW
# ==============================================================================
if __name__ == "__main__":
    print("--- STARTING MAIN LIGHTGBM EXECUTION WORKFLOW ---")

    if not os.path.exists(DRIVE_XML_FILE_PATH):
        print(f"CRITICAL ERROR: The file was not found at '{DRIVE_XML_FILE_PATH}'. Halting execution.")
    else:
        # Step 5.1: Data Ingestion and Hybrid Feature Engineering
        metabolite_list = parse_hmdb_with_hybrid_features(DRIVE_XML_FILE_PATH, MAX_METABOLITES_TO_PARSE)
        if not metabolite_list:
            print("\nCRITICAL ERROR: No data was parsed from the XML file. Halting execution.")
        else:
            raw_df = pd.DataFrame(metabolite_list)
            X, y, binarizer, imputer, scaler = preprocess_and_create_hybrid_features(raw_df, TARGET_BIOFLUIDS)

            if X is not None:
                # Step 5.2: Create Data Splits (Train+Val for search, Test for final eval)
                X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_SEED)
                X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1765, random_state=RANDOM_SEED)

                print("\nData splitting complete:")
                print(f"  - Training samples   : {X_train.shape[0]}")
                print(f"  - Validation samples : {X_val.shape[0]}")
                print(f"  - Final Test samples : {X_test.shape[0]}\n")

                # Step 5.3: Execute the Time-Budgeted Hyperparameter Search
                print(f"--- Starting 15-Minute Hyperparameter Search with Optuna ---")
                study = optuna.create_study(direction='maximize')
                study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), timeout=OPTUNA_TIMEOUT_SECONDS)

                print("\n--- Hyperparameter Search Complete ---")
                print(f"Best F1 Score on Validation Set: {study.best_value:.4f}")
                print("Best parameters found:")
                best_params = study.best_params
                for key, value in best_params.items():
                    print(f"  - {key}: {value}")

                # Step 5.4: Train the Final, Most Powerful Model
                print("\nTraining the final model on all available training data with the best parameters...")
                final_params = {
                    "device": "gpu", "objective": "binary",
                    "metric": "binary_logloss", "verbosity": -1,
                    "random_state": RANDOM_SEED
                }
                final_params.update(best_params)

                final_lgbm = lgb.LGBMClassifier(**final_params)
                final_model = MultiOutputClassifier(final_lgbm, n_jobs=-1)
                final_model.fit(X_train_full, y_train_full)
                print("Final model training complete.")

                # Step 5.5: Final Evaluation on Unseen Test Set
                y_pred_test = final_model.predict(X_test)
                print_evaluation_report(y_test, y_pred_test, "Final Optimized LightGBM Model")

                # Step 5.6: Save the Production-Ready Model Package
                model_package = {
                    'model': final_model,
                    'binarizer': binarizer,
                    'imputer': imputer,
                    'scaler': scaler
                }
                with open(MODEL_OUTPUT_FILENAME, 'wb') as f:
                    pickle.dump(model_package, f)
                print(f"\nModel package saved successfully to '{MODEL_OUTPUT_FILENAME}'.")

                try:
                    from google.colab import files
                    files.download(MODEL_OUTPUT_FILENAME)
                except (ImportError, NameError):
                    print(f"\nTo download the model ('{MODEL_OUTPUT_FILENAME}'), please locate it in the file browser.")

    print("\n--- LIGHTGBM WORKFLOW EXECUTION COMPLETE ---")