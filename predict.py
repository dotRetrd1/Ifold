import argparse
import torch
import numpy as np
import sys
import yaml
import urllib.request
import json
import os
from pathlib import Path
from Bio.PDB.MMCIFParser import MMCIFParser
from datetime import datetime
from scripts.saveRes import save_output

project_root = Path(__file__).parent
config_path = project_root / "config.yaml"

if not config_path.exists():
    raise FileNotFoundError(f"Could not find {config_path}. Please ensure it is in the root directory.")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

sys.path.append(str(project_root / config['paths']['scripts_training']))
sys.path.append(str(project_root / config['paths']['scripts']))

from scripts.training.model import iFoldResNet
from scripts.reconstruct3D import distance_to_3d, visualize_protein_3d

#query pdb for the inserted sequence 
def get_sequence_from_pdb(pdb_id):
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.upper()}/1"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data['entity_poly']['pdbx_seq_one_letter_code_can'].replace('\n', '')
    except Exception:
        return None

def search_pdb_by_sequence(sequence):
    query = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {"evalue_cutoff": 0.1, "identity_cutoff": 0.99, "sequence_type": "protein", "value": sequence}
        },
        "request_options": {"return_all_hits": False, "scoring_strategy": "sequence"},
        "return_type": "entry"
    }
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    try:
        req = urllib.request.Request(url, data=json.dumps(query).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            if result.get("result_set"):
                return result["result_set"][0]["identifier"]
    except Exception:
        pass
    return None

def fetch_ground_truth_from_pdb(pdb_id):
    temp_cif = f"{pdb_id}_temp.cif"
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        urllib.request.urlretrieve(url, temp_cif)
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id, temp_cif)
        
        longest_chain = None
        max_ca = 0
        for chain in structure[0].get_chains():
            ca_count = sum(1 for res in chain if 'CA' in res)
            if ca_count > max_ca:
                max_ca = ca_count
                longest_chain = chain
                
        if not longest_chain: 
            return None
            
        ca_coords = []
        for residue in longest_chain:
            if residue.get_id()[0] == ' ' and 'CA' in residue:
                ca_coords.append(residue['CA'].get_coord())
        return np.array(ca_coords)
    except Exception:
        return None
    finally:
        if os.path.exists(temp_cif):
            os.remove(temp_cif)


#the model takes in an (N,4) so this just converts the sequence to it 
def sequence_to_tensor(sequence):
    AA_TO_INDEX = {
        'A':0, 'C':1, 'D':2, 'E':3,
        'F':4, 'G':5, 'H':6, 'I':7,
        'K':8, 'L':9, 'M':10,'N':11,
        'P':12,'Q':13,'R':14,'S':15,
        'T':16,'V':17,'W':18,'Y':19,
        'X':20
    }

    def sequence_to_tensor(sequence):
        indices = [
            AA_TO_INDEX.get(aa.upper(), 20)
            for aa in sequence
        ]

        return torch.tensor(indices, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser(description="Predict 3D protein structure from sequence")
    parser.add_argument("--sequence", type=str, default=None, help="Raw amino acid sequence string")
    parser.add_argument("--pdb_id", type=str, default=None, help="Optional PDB ID to query")
    args = parser.parse_args()

    if not args.sequence and not args.pdb_id:
        print("Error: You must provide either --sequence or --pdb_id.")
        return

    target_sequence = args.sequence
    found_pdb_id = args.pdb_id

    #sync seq and PDB ID
    if found_pdb_id and not target_sequence:
        target_sequence = get_sequence_from_pdb(found_pdb_id)
        if not target_sequence:
            print("Could not retrieve sequence from PDB. Please check the PDB ID and try again.")
            return
            
    elif target_sequence and not found_pdb_id:
        print("Searching PDB for a matching sequence")
        found_pdb_id = search_pdb_by_sequence(target_sequence)
        if found_pdb_id:
            print(f"Match found; PDB ID: {found_pdb_id}")
        else:
            print("No exact match found in PDB. there wont be a ground truth comparison. Proceeding with prediction only.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    models_dir = project_root / config['paths']['models']
    weights_path = models_dir / config['paths']['weights_file']

    if not weights_path.exists():
        print(f"Error: Could not find weights at {weights_path}")
        return

    input_feat = config['model']['input_features']
    hidden = config['model']['hidden_channels']
    blocks = config['model']['num_blocks']
    
    model = iFoldResNet(input_features=input_feat, hidden_channels=hidden, num_blocks=blocks).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval() 

    #prep data
    print(f"prepping data (translating the sequence to Nx4) {len(target_sequence)}...")
    features = sequence_to_tensor(target_sequence)
    features = features.unsqueeze(0).to(device)

    #Forward Pass
    print("Running forward pass")
    with torch.inference_mode():
        pred_dist_tensor = model(features).squeeze(0)
    
    pred_dist_numpy = pred_dist_tensor.cpu().numpy()

    #3D Reconstruction
    pred_coords = distance_to_3d(pred_dist_numpy)

    print(f"Finished!")

    #Fetch Ground Truth and Visualize
    visualise = input("Would you like to visualize the predicted structure? (y/n): ").strip().lower()
    true_coords = None
    if found_pdb_id:
        print(f"Fetching ground truth coordinates for {found_pdb_id} from PDB...")
        true_coords = fetch_ground_truth_from_pdb(found_pdb_id)
        if true_coords is None:
            print("Warning: Failed to parse ground truth structure from PDB.")

    if visualise != 'y':
        print("Exiting without visualization.")
        return
    
    print("Launching Visualizer...")
    fig = visualize_protein_3d(pred_coords=pred_coords, true_coords=true_coords)

    if (input("Would you like to save the results? (y/n): ").strip().lower() == 'y'):
        print("Saving prediction data and visualization")
        out_dir = project_root / config['paths'].get('inference_out', './results')
        folder_name = f"{found_pdb_id.upper()}_Prediction" if found_pdb_id else f"Seq_{datetime.now().strftime('%m%d_%H%M')}"
        save_path = out_dir / folder_name
        save_output(save_path, target_sequence, found_pdb_id, pred_coords, fig)
        print(f"Results saved to {save_path}")

    
if __name__ == "__main__":
    main()