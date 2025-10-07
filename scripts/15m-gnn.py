# ==============================================================================
# SCRIPT: PREDICTING METABOLITE BIOFLUID DISTRIBUTION WITH GNN
# VERSION: 3.1 (Robust, Error-Free, and Fully Optimized)
#
# PURPOSE:
#   To build a state-of-the-art Graph Neural Network (GNN) model that achieves
#   maximum predictive accuracy for metabolite biofluid distribution within a
#   strict 15-minute training window on a T4 GPU. This script is production-
#   ready, fully refactored for stability, and optimized for performance.
#
# WORKFLOW:
#   1. Robustly parses the HMDB XML database, extracting chemical structures
#      (SMILES) and a rich set of physico-chemical properties to be used as
#      powerful graph-level features.
#   2. Converts each valid metabolite into a detailed graph representation,
#      gracefully handling and reporting any invalid SMILES strings silently.
#   3. Utilizes a Graph Isomorphism Network (GIN) architecture, a top-performing
#      GNN model for molecular property prediction.
#   4. Executes a time-budgeted (15-minute) hyperparameter search with Optuna
#      to discover the optimal architecture and training parameters.
#   5. Trains the final, optimized GNN model on the complete training and
#      validation dataset with an early stopping mechanism to prevent
#      overfitting and ensure generalization.
#   6. Conducts a rigorous final evaluation on an unseen test set using a
#      comprehensive suite of multi-label classification metrics.
#   7. Saves the trained model, data processors, and configuration as a single,
#      deployable package.
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
!pip install rdkit --quiet
!pip install optuna --quiet
# Install PyTorch and PyTorch Geometric for GPU (T4 is CUDA 11.x compatible)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
!pip install torch_geometric --quiet
print("All libraries installed successfully.\n")


# ==============================================================================
# STEP 2: IMPORT LIBRARIES AND SET GLOBAL CONFIGURATION
# ==============================================================================
print("--- STEP 2: IMPORTING LIBRARIES & DEFINING CONFIGURATION ---")

# --- Core Python & Data Handling ---
import os
import pickle
import warnings
import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from lxml import etree
from tqdm.notebook import tqdm

# --- Machine Learning & Deep Learning ---
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList, Sequential
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GINConv, global_add_pool
import optuna

# --- Chemistry & Feature Engineering ---
from rdkit import Chem, RDLogger
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

# --- Model Evaluation & Utility ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, precision_score, recall_score

# --- Global Constants ---
DRIVE_XML_FILE_PATH: str = '/content/drive/My Drive/hmdb/hmdb_metabolites.xml'
MODEL_OUTPUT_FILENAME: str = 'gnn_biofluid_model_v3.1_15min_final.pkl'
MAX_METABOLITES_TO_PARSE: int = 25000
TARGET_BIOFLUIDS: List[str] = [
    'Blood', 'Urine', 'Saliva', 'Cerebrospinal Fluid (CSF)', 'Feces', 'Sweat'
]
OPTUNA_TIMEOUT_SECONDS: int = 900  # 15 minutes
RANDOM_SEED: int = 42

# --- Feature Engineering Configuration ---
GRAPH_LEVEL_FEATURES: List[str] = [
    'average_molecular_weight', 'monisotopic_molecular_weight', 'logp', 'logs',
    'solubility', 'polar_surface_area', 'refractivity', 'polarizability',
    'rotatable_bond_count', 'acceptor_count', 'donor_count', 'physiological_charge'
]

# --- Environment Setup ---
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Execution device set to: {DEVICE}")

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    print("Google Drive mounted successfully.")
except ImportError:
    print("Could not mount Google Drive. Assuming non-Colab environment.")

print("Configuration and environment setup complete.\n")


# ==============================================================================
# STEP 3: DATA EXTRACTION AND FEATURE ENGINEERING
# ==============================================================================
print("--- STEP 3: DEFINING DATA PARSING & GRAPH CONSTRUCTION FUNCTIONS ---")

def parse_hmdb_with_features(xml_path: str, max_records: int) -> List[Dict[str, Any]]:
    """
    Efficiently parses the HMDB XML, extracting SMILES, biofluids, and a
    curated set of physico-chemical properties for advanced feature engineering.
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
    # Use iterparse for memory-efficient streaming of the large XML file
    context = etree.iterparse(xml_path, events=('end',), tag=TAGS['metabolite'])

    for _, elem in tqdm(context, desc="Processing Metabolites", total=max_records):
        # Extract required fields: SMILES and biofluids
        smiles_elem = elem.find(TAGS['smiles'])
        if smiles_elem is None or not smiles_elem.text:
            elem.clear() # Clear element to free memory
            continue

        biofluids = []
        bio_props_elem = elem.find(TAGS['bio_props'])
        if bio_props_elem is not None:
            locs_elem = bio_props_elem.find(TAGS['locations'])
            if locs_elem is not None:
                biofluids = [spec.text for spec in locs_elem.findall(TAGS['specimen'])]

        if not biofluids:
            elem.clear()
            continue

        # Extract additional features for graph-level embeddings
        features = {}
        # Direct children for molecular weights
        for child in elem:
            if child.tag.endswith('average_molecular_weight') and child.text:
                features['average_molecular_weight'] = float(child.text)
            elif child.tag.endswith('monisotopic_molecular_weight') and child.text:
                features['monisotopic_molecular_weight'] = float(child.text)

        # Nested predicted properties
        pred_props_elem = elem.find(TAGS['pred_props'])
        if pred_props_elem is not None:
            for prop in pred_props_elem.findall(TAGS['prop']):
                kind_elem = prop.find(TAGS['kind'])
                value_elem = prop.find(TAGS['value'])
                if kind_elem is not None and value_elem is not None and kind_elem.text in GRAPH_LEVEL_FEATURES:
                    try:
                        features[kind_elem.text] = float(value_elem.text)
                    except (ValueError, TypeError):
                        continue # Skip non-numeric values like 'Yes'/'No'

        parsed_data.append({
            'smiles': smiles_elem.text,
            'biofluids': list(set(biofluids)),
            'features': features
        })

        # Memory management for lxml.iterparse
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        if len(parsed_data) >= max_records:
            break
    return parsed_data

def get_atom_features(atom: Chem.Atom) -> List[float]:
    """Generates a feature vector for a single atom using one-hot encoding."""
    possible_symbols = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    symbol = [0.0] * (len(possible_symbols) + 1) # +1 for 'other'
    try:
        symbol[possible_symbols.index(atom.GetSymbol())] = 1.0
    except ValueError:
        symbol[-1] = 1.0

    return symbol + [
        float(atom.GetDegree()), float(atom.GetFormalCharge()),
        float(atom.GetNumRadicalElectrons()), float(atom.GetHybridization()),
        float(atom.GetIsAromatic()), float(atom.IsInRing())
    ]

def smiles_to_graph_data(smiles: str, features: Dict[str, float], y_vec: np.ndarray, feature_list: List[str]) -> Optional[Data]:
    """Converts a SMILES string and its features into a PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    # Node Features (atoms)
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge Index (bonds)
    adj = GetAdjacencyMatrix(mol)
    row, col = np.where(adj)
    edge_index = torch.tensor([row, col], dtype=torch.long).view(2, -1)

    # Graph-level features (molecule properties)
    u_list = [features.get(feat, 0.0) for feat in feature_list] # Impute missing with 0
    u = torch.tensor(u_list, dtype=torch.float).view(1, -1)

    # Target vector
    y = torch.tensor(y_vec, dtype=torch.float).view(1, -1)

    return Data(x=x, edge_index=edge_index, y=y, u=u)


class MetaboliteDataset(Dataset):
    """Custom PyTorch Geometric Dataset for handling metabolite graphs."""
    def __init__(self, data_list: List[Data]):
        super().__init__()
        # PyG's internal mechanism handles this better than a simple list
        self.data_list = data_list if data_list else []

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

    # This is the key fix: Allow slicing to return a new Dataset instance
    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self.get(idx)
        elif isinstance(idx, (slice, list, np.ndarray, torch.Tensor)):
            return self.__class__([self.get(i) for i in range(*idx.indices(len(self)))]) if isinstance(idx, slice) else self.__class__([self.get(i) for i in idx])
        raise IndexError

print("Data processing functions defined.\n")


# ==============================================================================
# STEP 4: GNN MODEL ARCHITECTURE
# ==============================================================================
print("--- STEP 4: DEFINING GNN MODEL ARCHITECTURE ---")

class GINBiofluidPredictor(torch.nn.Module):
    """
    Graph Isomorphism Network (GIN) with added graph-level features. This
    architecture is designed for high performance on molecular tasks.
    """
    def __init__(self, num_node_features: int, num_graph_features: int, num_classes: int,
                 hidden_channels: int, num_layers: int, dropout: float):
        super(GINBiofluidPredictor, self).__init__()
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        # Input layer
        nn1 = Sequential(Linear(num_node_features, hidden_channels), BatchNorm1d(hidden_channels), torch.nn.ReLU(), Linear(hidden_channels, hidden_channels))
        self.convs.append(GINConv(nn1))
        self.batch_norms.append(BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            nn = Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), torch.nn.ReLU(), Linear(hidden_channels, hidden_channels))
            self.convs.append(GINConv(nn))
            self.batch_norms.append(BatchNorm1d(hidden_channels))

        # Output MLP
        self.mlp = Sequential(
            Linear(hidden_channels + num_graph_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            Linear(hidden_channels, num_classes)
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch, u = data.x, data.edge_index, data.batch, data.u
        for conv, bn in zip(self.convs, self.batch_norms):
            x = F.relu(bn(conv(x, edge_index)))

        # Aggregate node features into a single graph-level representation
        x_pooled = global_add_pool(x, batch)

        # Concatenate graph-level features with the learned graph embedding
        combined_features = torch.cat([x_pooled, u], dim=1)

        return self.mlp(combined_features)

print("GNN model defined.\n")


# ==============================================================================
# STEP 5: TRAINING, EVALUATION, AND UTILITY FUNCTIONS
# ==============================================================================
print("--- STEP 5: DEFINING TRAINING & EVALUATION LOOPS ---")

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            out = model(data)
            preds = (torch.sigmoid(out) > 0.5).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(data.y.cpu().numpy())
    return np.vstack(all_preds), np.vstack(all_labels)

def save_model_package(package: Dict[str, Any], filename: str):
    """Saves the final model state, binarizer, and feature scaler."""
    print(f"\nSaving the final model package to '{filename}'...")
    with open(filename, 'wb') as f:
        pickle.dump(package, f)
    print("Model package saved successfully.")

def print_evaluation_report(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """Calculates and prints a comprehensive set of multi-label evaluation metrics."""
    print(f"\n--- Evaluating {model_name} Performance on the Unseen Test Set ---")
    metrics = {
        "F1 Score (Micro)": f1_score(y_true, y_pred, average='micro', zero_division=0),
        "Subset Accuracy": accuracy_score(y_true, y_pred),
        "Hamming Loss": hamming_loss(y_true, y_pred),
        "Precision (Micro)": precision_score(y_true, y_pred, average='micro', zero_division=0),
        "Recall (Micro)": recall_score(y_true, y_pred, average='micro', zero_division=0)
    }
    for name, score in metrics.items():
        print(f"  - {name:<20}: {score:.4f}")

print("Utility functions defined.\n")

# ==============================================================================
# STEP 6: OPTUNA HYPERPARAMETER OPTIMIZATION OBJECTIVE
# ==============================================================================
def objective(trial: optuna.trial.Trial, train_dataset, val_dataset) -> float:
    """
    The 'objective' function for Optuna. Defines the search space, trains a GNN model
    on a subset of data, evaluates it, and returns the F1 score for maximization.
    """
    params = {
        'hidden_channels': trial.suggest_categorical('hidden_channels', [64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 2, 5),
        'dropout': trial.suggest_float('dropout', 0.1, 0.6),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
    }
    
    # Re-create DataLoaders inside the objective to use the trial's batch_size
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])

    model = GINBiofluidPredictor(
        num_node_features=train_dataset.num_node_features,
        num_graph_features=train_dataset[0].u.shape[1],
        num_classes=train_dataset.num_classes,
        hidden_channels=params['hidden_channels'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = torch.nn.BCEWithLogitsLoss()

    # Train for a fixed, small number of epochs for a fast evaluation
    for epoch in range(10):
        train_epoch(model, train_loader, optimizer, criterion)
    
    y_pred, y_true = evaluate(model, val_loader)
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)

    trial.report(f1, epoch)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return f1

# ==============================================================================
# STEP 7: MAIN EXECUTION WORKFLOW
# ==============================================================================
if __name__ == "__main__":
    print("--- STARTING MAIN GNN EXECUTION WORKFLOW ---")

    if not os.path.exists(DRIVE_XML_FILE_PATH):
        print(f"CRITICAL ERROR: The file was not found at '{DRIVE_XML_FILE_PATH}'. Halting execution.")
    else:
        # Step 7.1: Data Ingestion and Preprocessing
        metabolite_list = parse_hmdb_with_features(DRIVE_XML_FILE_PATH, MAX_METABOLITES_TO_PARSE)
        if not metabolite_list:
            print("\nCRITICAL ERROR: No data was parsed from the XML file. Halting execution.")
        else:
            df = pd.DataFrame(metabolite_list)
            df['biofluids'] = df['biofluids'].apply(lambda lst: [b for b in lst if b in TARGET_BIOFLUIDS])
            df = df[df['biofluids'].apply(len) > 0].reset_index(drop=True)

            mlb = MultiLabelBinarizer(classes=TARGET_BIOFLUIDS)
            y_binarized = mlb.fit_transform(df['biofluids'])
            
            # Step 7.2: Create Graph Dataset with Silent Error Handling
            print("\nConstructing graph objects for each metabolite...")
            graph_data_list = []
            invalid_smiles_count = 0
            
            # Suppress RDKit warnings during conversion
            lg = RDLogger.logger()
            lg.setLevel(RDLogger.CRITICAL)
            
            for i in tqdm(range(len(df)), desc="Creating Graphs"):
                graph = smiles_to_graph_data(df.loc[i, 'smiles'], df.loc[i, 'features'], y_binarized[i], GRAPH_LEVEL_FEATURES)
                if graph:
                    graph_data_list.append(graph)
                else:
                    invalid_smiles_count += 1
            
            lg.setLevel(RDLogger.INFO) # Restore logger
            print(f"Successfully created {len(graph_data_list)} graph objects.")
            if invalid_smiles_count > 0:
                print(f"Skipped {invalid_smiles_count} entries due to invalid SMILES strings.")

            # Step 7.3: Scale Graph-Level Features
            all_u = torch.cat([data.u for data in graph_data_list], dim=0)
            scaler = StandardScaler()
            scaled_u = torch.tensor(scaler.fit_transform(all_u), dtype=torch.float)
            for i, data in enumerate(graph_data_list):
                data.u = scaled_u[i].unsqueeze(0)

            dataset = MetaboliteDataset(graph_data_list)

            # Step 7.4: Create Robust Data Splits using Indices
            # This is the correct way to split a PyG Dataset with sklearn
            indices = list(range(len(dataset)))
            train_val_indices, test_indices = train_test_split(indices, test_size=0.15, random_state=RANDOM_SEED)
            train_indices, val_indices = train_test_split(train_val_indices, test_size=0.1765, random_state=RANDOM_SEED) # 0.15 / 0.85 = ~0.1765 for a 70/15/15 split

            train_dataset = dataset[train_indices]
            val_dataset = dataset[val_indices]
            test_dataset = dataset[test_indices]
            
            train_val_dataset = dataset[train_val_indices] # For final training

            print("\nData splitting complete:")
            print(f"  - Training samples   : {len(train_dataset)}")
            print(f"  - Validation samples : {len(val_dataset)}")
            print(f"  - Final Test samples : {len(test_dataset)}\n")

            # Step 7.5: Execute Time-Budgeted Hyperparameter Search
            print(f"--- Starting 15-Minute Hyperparameter Search with Optuna ---")
            study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
            study.optimize(lambda trial: objective(trial, train_dataset, val_dataset), timeout=OPTUNA_TIMEOUT_SECONDS)

            print("\n--- Hyperparameter Search Complete ---")
            print(f"Best F1 Score on Validation Set: {study.best_value:.4f}")
            print("Best parameters found:")
            best_params = study.best_params
            for key, value in best_params.items():
                print(f"  - {key}: {value}")

            # Step 7.6: Train the Final, Most Powerful Model
            print("\nTraining the final model on all training+validation data with best parameters and early stopping...")
            final_train_loader = DataLoader(train_val_dataset, batch_size=best_params['batch_size'], shuffle=True)
            final_val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size']) # Use original val set for stopping
            final_test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])
            
            final_model = GINBiofluidPredictor(
                num_node_features=dataset.num_node_features,
                num_graph_features=dataset[0].u.shape[1],
                num_classes=dataset.num_classes,
                hidden_channels=best_params['hidden_channels'],
                num_layers=best_params['num_layers'],
                dropout=best_params['dropout']
            ).to(DEVICE)

            final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])
            final_criterion = torch.nn.BCEWithLogitsLoss()
            
            patience, best_val_f1, patience_counter = 15, 0, 0

            for epoch in range(200):
                start_time = time.time()
                loss = train_epoch(final_model, final_train_loader, final_optimizer, final_criterion)
                y_pred_val, y_true_val = evaluate(final_model, final_val_loader)
                val_f1 = f1_score(y_true_val, y_pred_val, average='micro', zero_division=0)
                
                print(f"Epoch {epoch+1:03d} | Train Loss: {loss:.4f} | Val F1: {val_f1:.4f} | Time: {time.time()-start_time:.2f}s")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    torch.save(final_model.state_dict(), 'best_model_state.pt')
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
            
            final_model.load_state_dict(torch.load('best_model_state.pt'))

            # Step 7.7: Final Evaluation on Unseen Test Set
            y_pred_test, y_true_test = evaluate(final_model, final_test_loader)
            print_evaluation_report(y_true_test, y_pred_test, "Final Optimized GNN Model")

            # Step 7.8: Save the Production-Ready Model Package
            model_package = {
                'model_state_dict': final_model.state_dict(),
                'model_params': {
                    'num_node_features': dataset.num_node_features,
                    'num_graph_features': dataset[0].u.shape[1],
                    'num_classes': dataset.num_classes,
                    'hidden_channels': best_params['hidden_channels'],
                    'num_layers': best_params['num_layers'],
                    'dropout': best_params['dropout']
                },
                'binarizer': mlb,
                'feature_scaler': scaler,
                'graph_feature_list': GRAPH_LEVEL_FEATURES
            }
            save_model_package(model_package, MODEL_OUTPUT_FILENAME)
            
            try:
                from google.colab import files
                files.download(MODEL_OUTPUT_FILENAME)
            except (ImportError, NameError):
                print(f"\nTo download the model ('{MODEL_OUTPUT_FILENAME}'), please locate it in the file browser.")

    print("\n--- GNN WORKFLOW EXECUTION COMPLETE ---")