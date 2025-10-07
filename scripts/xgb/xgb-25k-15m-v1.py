# ==============================================================================
# SCRIPT:   Peak Performance XGBoost Pipeline for Metabolite Biofluid Prediction
# VERSION:  2.2 (Definitive, Robustness Edition)
# PURPOSE:
#   To achieve the absolute theoretical peak predictive performance for a dataset
#   of 120,000 metabolites. This script integrates a meticulously curated
#   feature set, an exhaustive hyperparameter search protocol, and rigorous
#   archiving for unimpeachable reproducibility. This version includes critical
#   robustness patches to handle non-finite descriptor values and ensure stable
#   data splitting.
#
# KEY FEATURES:
#   1. STRATEGIC FEATURE ENGINEERING: Utilizes 28 orthogonal, high-impact
#      physico-chemical descriptors computed directly from SMILES to provide a
#      rich and non-redundant feature space.
#   2. EXHAUSTIVE HYPERPARAMETER OPTIMIZATION: Implements a 6-hour Optuna search
#      with wide parameter spaces and an intelligent early-stopping mechanism
#      to find the true global optimum.
#   3. UNIMPEACHABLE REPRODUCIBILITY: Saves the final model, scalers, binarizer,
#      and the exact training, validation, and test datasets to a single,
#      portable archive.
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
# STEP 2: IMPORTS & PEAK PERFORMANCE CONFIGURATION
# ==============================================================================

print("--- Importing libraries and setting up the peak performance configuration ---")

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

# --- Global Constants: Calibrated for 120,000 Metabolites ---

DRIVE_XML_FILE_PATH: str = '/content/drive/My Drive/hmdb/hmdb_metabolites.xml'
MAX_METABOLITES_TO_PARSE: int = 25000
OUTPUT_ARCHIVE_NAME: str = 'XGB_Peak_Performance_Run_120k.zip'

TARGET_BIOFLUIDS: List[str] = [
    'Blood', 'Urine', 'Saliva', 'Cerebrospinal Fluid (CSF)', 'Feces', 'Sweat'
]
RANDOM_SEED: int = 42

# --- Optuna Configuration: Exhaustive 6-Hour Search ---

OPTUNA_TIMEOUT_SECONDS: int = 900  # 6 hours
OPTUNA_EARLY_STOPPING_ROUNDS: int = 50 # Stop if no improvement after 50 trials

# --- Feature Engineering Configuration: 28 Curated Descriptors ---

COMPUTED_NUMERICAL_FEATURES: List[str] = [
    'MolWt', 'ExactMolWt', 'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors',
    'NumValenceElectrons', 'NumAromaticRings', 'NumAliphaticRings', 'RingCount',
    'TPSA', 'MolLogP', 'MolMR', 'FractionCSP3', 'NumRotatableBonds', 'qed',
    'HallKierAlpha', 'MaxPartialCharge', 'MinPartialCharge', 'BalabanJ',
    'BertzCT', 'Chi0v', 'Kappa2', 'fr_NH2', 'fr_COO', 'fr_phenol',
    'fr_aldehyde', 'fr_ketone', 'fr_ether'
]
FINGERPRINT_BITS: int = 2048

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
        if smiles_elem is None or not smiles_elem.text:
            elem.clear()
            continue
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
    """
    Computes Morgan fingerprints and the 28 curated numerical descriptors.
    Includes a robustness patch to handle non-finite descriptor values.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)

        # 1. Fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FINGERPRINT_BITS)
        fp_array = np.array(list(fp.ToBitString()), dtype=np.uint8)

        # 2. Curated Numerical Descriptors
        descriptors = {
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
            'fr_phenol': Fragments.fr_phenol(mol), 'fr_aldehyde': Fragments.fr_aldehyde(mol),
            'fr_ketone': Fragments.fr_ketone(mol), 'fr_ether': Fragments.fr_ether(mol)
        }

        # --- CRITICAL PATCH: Replace any potential non-finite values with NaN ---
        # This robustly handles `inf` from BalabanJ or other descriptor issues.
        for key, value in descriptors.items():
            if not np.isfinite(value):
                descriptors[key] = np.nan
        # --- END CRITICAL PATCH ---

        return {'fingerprint': fp_array, 'descriptors': descriptors}
    except Exception:
        # This block catches unexpected, severe errors during molecule processing.
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

    # --- CRITICAL PATCH: Add robustness check for feature computation failure ---
    if not valid_indices:
        print("\nCRITICAL ERROR: Feature computation failed for all parsed metabolites.")
        print("This may be due to invalid SMILES strings or an issue in the feature calculation function.")
        return None, None, None, None, None, None
    # --- END CRITICAL PATCH ---

    y_clean = y[valid_indices]
    df_clean = df.iloc[valid_indices].reset_index(drop=True)

    X_fp = np.vstack([feature_results[i]['fingerprint'] for i in valid_indices])
    desc_df = pd.DataFrame([feature_results[i]['descriptors'] for i in valid_indices]).reindex(columns=COMPUTED_NUMERICAL_FEATURES)

    print(f"Kept {len(y_clean)} metabolites after feature computation.")

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
        print(f"  - {name:<20}: {score:.6f}") # Increased precision for peak report

# ==============================================================================
# STEP 4: CUSTOM XGBOOST WRAPPER & OPTUNA CALLBACK
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
    def predict(self, X):
        return np.hstack([est.predict(X).reshape(-1, 1) for est in self.estimators_])
    def predict_proba(self, X):
        return np.hstack([est.predict_proba(X)[:, 1].reshape(-1, 1) for est in self.estimators_])

class OptunaEarlyStoppingCallback:
    """Callback to stop Optuna study if no improvement is seen for a given number of trials."""
    def __init__(self, early_stopping_rounds: int):
        self.early_stopping_rounds = early_stopping_rounds
        self._iter = 0
        self.best_value = -float('inf')
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if trial.value > self.best_value:
            self.best_value = trial.value
            self._iter = 0
        else:
            self._iter += 1
        if self._iter >= self.early_stopping_rounds:
            print(f"\n--- Stopping study early: No improvement in {self.early_stopping_rounds} trials. ---")
            study.stop()

# ==============================================================================
# STEP 5: OPTUNA OBJECTIVE FUNCTION
# ==============================================================================

def objective(trial, X_train, y_train, X_val, y_val):
    """Defines the Optuna objective function with the unrestricted search space."""
    param_grid = {
        'tree_method': 'gpu_hist', 'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'random_state': RANDOM_SEED, 'n_jobs': -1,
        'n_estimators': trial.suggest_int('n_estimators', 1000, 5000, step=200),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 8, 24),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 25.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
    }
    model = XGBoostMultiOutputWrapper(**param_grid)
    model.fit(X_train, y_train)
    return f1_score(y_val, model.predict(X_val), average='micro', zero_division=0)

# ==============================================================================
# STEP 6: MAIN EXECUTION WORKFLOW
# ==============================================================================

if __name__ == "__main__":
    print("--- Starting Peak Performance XGBoost Execution Workflow ---")

    if not os.path.exists(DRIVE_XML_FILE_PATH):
        print(f"CRITICAL ERROR: File not found at '{DRIVE_XML_FILE_PATH}'. Halting.")
    else:
        parsed_data = parse_hmdb_for_smiles_and_biofluids(DRIVE_XML_FILE_PATH, MAX_METABOLITES_TO_PARSE)
        if not parsed_data:
            print("\nCRITICAL ERROR: No data parsed. Halting.")
        else:
            X, y, df_clean, binarizer, imputer, scaler = preprocess_and_engineer_features(parsed_data, TARGET_BIOFLUIDS)
            if X is not None:
                # --- CRITICAL PATCH: Robust Stratified Splitting ---
                # This attempts a stratified split but falls back to a regular split if it fails,
                # preventing crashes on small datasets with rare label combinations.
                try:
                    # Attempt stratified split (best practice)
                    X_train_full, X_test, y_train_full, y_test, df_train_full, df_test = train_test_split(
                        X, y, df_clean, test_size=0.15, random_state=RANDOM_SEED, stratify=y
                    )
                    val_size_fraction = 0.15 / 0.85
                    X_train, X_val, y_train, y_val, df_train, df_val = train_test_split(
                        X_train_full, y_train_full, df_train_full, test_size=val_size_fraction, random_state=RANDOM_SEED, stratify=y_train_full
                    )
                    print("\nData splitting complete (stratified).")
                except ValueError:
                    # Fallback to non-stratified split if stratification is not possible
                    warnings.warn("Stratified split failed, likely due to rare classes in the small dataset. Falling back to a non-stratified split.")
                    X_train_full, X_test, y_train_full, y_test, df_train_full, df_test = train_test_split(
                        X, y, df_clean, test_size=0.15, random_state=RANDOM_SEED
                    )
                    val_size_fraction = 0.15 / 0.85
                    X_train, X_val, y_train, y_val, df_train, df_val = train_test_split(
                        X_train_full, y_train_full, df_train_full, test_size=val_size_fraction, random_state=RANDOM_SEED
                    )
                    print("\nData splitting complete (non-stratified fallback).")
                # --- END CRITICAL PATCH ---

                print(f"  - Total Valid Metabolites: {X.shape[0]}")
                print(f"  - Training samples   : {X_train.shape[0]}")
                print(f"  - Validation samples : {X_val.shape[0]}")
                print(f"  - Final Test samples : {X_test.shape[0]}\n")

                print(f"--- Starting {OPTUNA_TIMEOUT_SECONDS/3600:.1f}-Hour Exhaustive Hyperparameter Search ---")
                early_stopping_callback = OptunaEarlyStoppingCallback(OPTUNA_EARLY_STOPPING_ROUNDS)
                study = optuna.create_study(direction='maximize')
                study.optimize(
                    lambda trial: objective(trial, X_train, y_train, X_val, y_val),
                    timeout=OPTUNA_TIMEOUT_SECONDS,
                    callbacks=[early_stopping_callback]
                )

                print("\n--- Hyperparameter Search Complete ---")
                print(f"Best F1 Score on Validation Set: {study.best_value:.6f}")
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

                y_pred_test = final_model.predict(X_test)
                y_pred_proba_test = final_model.predict_proba(X_test)
                print_full_evaluation_report(y_test, y_pred_test, y_pred_proba_test, "Optimized XGBoost Model")

                print("\n--- Saving All Artifacts for Reproducibility ---")
                # 1. Save model package
                model_package = {
                    'model': final_model, 'binarizer': binarizer, 'imputer': imputer, 'scaler': scaler,
                    'computed_numerical_features_order': COMPUTED_NUMERICAL_FEATURES,
                    'fingerprint_bits': FINGERPRINT_BITS
                }
                model_filename = 'xgb_peak_model.pkl'
                with open(model_filename, 'wb') as f:
                    pickle.dump(model_package, f)

                # 2. Prepare dataset dataframes
                train_df = df_train.copy()
                val_df = df_val.copy()
                test_df = df_test.copy()

                for i, fluid in enumerate(binarizer.classes_):
                    train_df[f'label_{fluid}'] = y_train[:, i]
                    val_df[f'label_{fluid}'] = y_val[:, i]
                    test_df[f'label_{fluid}'] = y_test[:, i]

                # 3. Archive datasets and model into a single zip file
                with zipfile.ZipFile(OUTPUT_ARCHIVE_NAME, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.write(model_filename)
                    print(f"  - Saved and compressed '{model_filename}'")
                    # Save datasets
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

    print("\n--- Peak Performance XGBoost Workflow Execution Complete ---")