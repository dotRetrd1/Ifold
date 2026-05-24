import numpy as np
import json
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

AA_TO_INDEX = {
    'A':0, 'R':1, 'N':2, 'D':3,
    'C':4, 'E':5, 'Q':6, 'G':7,
    'H':8, 'I':9, 'L':10,'K':11,
    'M':12,'F':13,'P':14,'S':15,
    'T':16,'W':17,'Y':18,'V':19,
    'X':20
}


script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
data_dir = project_root / "data" / "trainingData" / "ca_coords"

seq_file = data_dir / "sequences.json"
if not seq_file.exists():
    raise FileNotFoundError("sequences.json not found! Please run buildDataset.py first.")

with open(seq_file, "r") as f:
    sequences = json.load(f)

def generate_input_features(sequence):
    """
    Convert amino-acid sequence into integer residue tokens.
    Output shape: (N,)
    """

    feature_vector = np.array([
        AA_TO_INDEX.get(char.upper(), 20)
        for char in sequence
    ], dtype=np.int64)

    return feature_vector

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