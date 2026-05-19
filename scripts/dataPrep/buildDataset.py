import urllib.request
import os
import json
import random
import numpy as np
from pathlib import Path
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.Data.IUPACData import protein_letters_3to1
from rcsbapi.search import search_attributes as attrs

# ---------setup dirs---------
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

data_dir = project_root / "data" / "trainingData" / "ca_coords"
data_dir.mkdir(parents=True, exist_ok=True)

temp_dir = project_root / "data" / "trainingData" / "temp"
temp_dir.mkdir(parents=True, exist_ok=True)

# ---------find candidates (beta barrels)---------
print("Downloading pure CATH database classifications...")
cath_url = "https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt"
cath_txt_path = temp_dir / "cath-domain-list.txt"

urllib.request.urlretrieve(cath_url, cath_txt_path)

cath_barrel_ids = set()
with open(cath_txt_path, "r") as f:
    for line in f:
        if line.startswith("#"): 
            continue
        parts = line.split()
        if len(parts) >= 5 and parts[1] == "2" and parts[2] == "40" and parts[3] == "160":
            pdb_id = parts[0][:4].upper()
            cath_barrel_ids.add(pdb_id)

print("Querying the PDB for high-quality X-Ray structures...")
q_method = attrs.exptl.method == "X-RAY DIFFRACTION"
q_res = attrs.rcsb_entry_info.resolution_combined <= 3.5
q_lengthLow = attrs.entity_poly.rcsb_sample_sequence_length >= 100
q_lengthUpp = attrs.entity_poly.rcsb_sample_sequence_length <= 350

rcsb_query = q_method & q_res & q_lengthLow & q_lengthUpp
rcsb_ids = set(rcsb_query("entry"))

training_ids = list(cath_barrel_ids.intersection(rcsb_ids))
print(f"Found {len(training_ids)} total pristine candidates.")

if os.path.exists(cath_txt_path):
    os.remove(cath_txt_path)

# ---------selection---------
try:
    HOW_MANY_TO_DOWNLOAD = int(input(f"How many beta barrels to download? (1-{len(training_ids)})\n"))
except ValueError:
    HOW_MANY_TO_DOWNLOAD = 10

target_ids = random.sample(training_ids, min(HOW_MANY_TO_DOWNLOAD, len(training_ids)))
print(f"Selected {len(target_ids)} random targets for processing.\n")

# ---------gather data---------
parser = MMCIFParser(QUIET=True)
resolved_sequences = {} #Dictionary to hold the physically resolved sequences

for pdb_id in target_ids:
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    temp_cif_path = temp_dir / f"{pdb_id}.cif"
    npy_path = data_dir / f"{pdb_id}_ca.npy"

    try:
        print(f"Processing {pdb_id}...")
        urllib.request.urlretrieve(url, str(temp_cif_path))
        structure = parser.get_structure(pdb_id, str(temp_cif_path))
        
        model = structure[0]
        longest_chain = None
        max_ca_count = 0
        
        for chain in model.get_chains():
            ca_count = sum(1 for residue in chain if 'CA' in residue)
            if ca_count > max_ca_count:
                max_ca_count = ca_count
                longest_chain = chain
        
        if longest_chain is None:
            continue

        #extract coords abd the true physical sequence simultaneously
        ca_coords = []
        actual_sequence = ""
        
        for residue in longest_chain:
            if 'CA' in residue:
                try:
                    aa_char = protein_letters_3to1.get(residue.get_resname(), 'X')
                    actual_sequence += aa_char
                    ca_coords.append(residue['CA'].get_coord())
                except KeyError:
                    # Skip unrecognized ligands/modified residues gracefully
                    continue

        ca_matrix = np.array(ca_coords)
        
        if ca_matrix.shape[0] < 50:
             print(f" -> Skipping {pdb_id}: Main chain too short after resolution check.")
             continue

        #save coordinates to disk
        np.save(npy_path, ca_matrix)
        
        #save sequence to dictionary
        resolved_sequences[pdb_id] = actual_sequence
        print(f" -> Success! Saved {npy_path.name} | Resolved Length: {len(actual_sequence)}")

    except Exception as e:
        print(f" -> Failed to extract data for {pdb_id}: {e}")
    
    finally:
        if temp_cif_path.exists():
            os.remove(temp_cif_path)

#save the dictionary so prepData.py can access the physical sequences
seq_path = data_dir / "sequences.json"
with open(seq_path, "w") as f:
    json.dump(resolved_sequences, f, indent=4)

print(f"\nData pipeline complete! Saved {len(resolved_sequences)} valid targets to {data_dir}")