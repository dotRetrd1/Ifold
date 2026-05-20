import numpy as np
import json
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

# ---------for the Nx4 matrix---------
PHYSICS_DICT = {
    'A': [-1.22,  0.0,  1.8],   # Alanine
    'R': [ 0.94,  1.0, -4.5],   # Arginine
    'N': [-0.57,  0.0, -3.5],   # Asparagine
    'D': [-0.64, -1.0, -3.5],   # Aspartic Acid
    'C': [-0.71,  0.0,  2.5],   # Cysteine
    'E': [ 0.05, -1.0, -3.5],   # Glutamic Acid
    'Q': [ 0.19,  0.0, -3.5],   # Glutamine
    'G': [-1.95,  0.0, -0.4],   # Glycine
    'H': [ 0.43,  0.1, -3.2],   # Histidine
    'I': [ 0.77,  0.0,  4.5],   # Isoleucine
    'L': [ 0.77,  0.0,  3.8],   # Leucine
    'K': [ 0.82,  1.0, -3.9],   # Lysine
    'M': [ 0.67,  0.0,  1.9],   # Methionine
    'F': [ 1.36,  0.0,  2.8],   # Phenylalanine
    'P': [-0.60,  0.0, -1.6],   # Proline
    'S': [-1.21,  0.0, -0.8],   # Serine
    'T': [-0.52,  0.0, -0.7],   # Threonine
    'W': [ 2.33,  0.0, -0.9],   # Tryptophan
    'Y': [ 1.46,  0.0, -1.3],   # Tyrosine
    'V': [ 0.09,  0.0,  4.2],   # Valine
    'X': [ 0.00,  0.0,  0.0]    # Unknown/Dummy
}


script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
data_dir = project_root / "data" / "trainingData" / "ca_coords"

seq_file = data_dir / "sequences.json"
if not seq_file.exists():
    raise FileNotFoundError("sequences.json not found! Please run buildDataset.py first.")

with open(seq_file, "r") as f:
    sequences = json.load(f)

# ---------generate input features---------
def generate_input_features(sequence):
    """
    Convert 1D string sequence into N x 4 input tensor. (I still dont really get tensors)
    [Volume, Charge, Hydrophobicity, Sinusoidal_Position]
    """
    N = len(sequence)
    feature_matrix = np.zeros((N, 4), dtype=np.float32)
    
    for i, char in enumerate(sequence):
        chem_features = PHYSICS_DICT.get(char.upper(), PHYSICS_DICT['X'])
        
        # Upgraded Positional Encoding: Sinusoidal mapping
        pos_encoding = np.sin((i / (N - 1)) * np.pi) if N > 1 else 0.0
        
        feature_matrix[i, 0] = chem_features[0]
        feature_matrix[i, 1] = chem_features[1]
        feature_matrix[i, 2] = chem_features[2]
        feature_matrix[i, 3] = pos_encoding
        
    return feature_matrix

print("generating Input and Ground Truth Matrices")
processed_count = 0

for file in data_dir.glob("*_ca.npy"):
    pdb_id = file.stem.split('_')[0]
    sequence = sequences.get(pdb_id)
    
    if not sequence:
        print(f" -> no seq found for {pdb_id}, skipping.")
        continue
        
    coords = np.load(file)
    
    #generate and save Ground Truth (N x N Distance Matrix)
    distance_matrix = squareform(pdist(coords, metric='euclidean'))
    dist_path = data_dir / f"{pdb_id}_dist.npy"
    np.save(dist_path, distance_matrix)
    
    #generate and save Input Features (N x 4 Matrix)
    feature_matrix = generate_input_features(sequence)
    feat_path = data_dir / f"{pdb_id}_features.npy"
    np.save(feat_path, feature_matrix)
    
    processed_count += 1
    print(f" -> {pdb_id}: saved {feature_matrix.shape} input and {distance_matrix.shape} ground truth.")

print(f"\nprep complete, formatted {processed_count} protein pairs for training.")