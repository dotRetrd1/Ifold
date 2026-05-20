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

print("Querying the PDB for high-quality structures...")
q_method = attrs.exptl.method == "X-RAY DIFFRACTION"
q_res = attrs.rcsb_entry_info.resolution_combined <= 3.5
q_lengthLow = attrs.entity_poly.rcsb_sample_sequence_length >= 100
q_lengthUpp = attrs.entity_poly.rcsb_sample_sequence_length <= 350

rcsb_query = q_method & q_res & q_lengthLow & q_lengthUpp
rcsb_ids = set(rcsb_query("entry"))

if os.path.exists(cath_txt_path):
    os.remove(cath_txt_path)

#mode selection
print("\n==================================================")
print("SELECT DATASET MODE:")
print("1: General Pre-Training (Thousands of diverse proteins)")
print("2: Specialized Fine-Tuning (Strictly Beta-Barrels)")
print("==================================================")
mode = input("Enter 1 or 2: ")

if mode == "1":
    training_ids = list(rcsb_ids)
    print(f"\n[MODE 1] Found {len(training_ids)} total diverse candidates.")
else:
    training_ids = list(cath_barrel_ids.intersection(rcsb_ids))
    print(f"\n[MODE 2] Found {len(training_ids)} pristine beta-barrel candidates.")

# ---------selection---------
try:
    HOW_MANY_TO_DOWNLOAD = int(input(f"How many successful pristine targets do you want? (1-{len(training_ids)}): "))
except ValueError:
    HOW_MANY_TO_DOWNLOAD = 10

target_ids = list(training_ids)
random.shuffle(target_ids)
print(f"\nInitiating hunt for {HOW_MANY_TO_DOWNLOAD} pristine targets...\n")

# ---------gather data & QC ----------
parser = MMCIFParser(QUIET=True)
resolved_sequences = {} 

successful_count = 0
id_index = 0

while successful_count < HOW_MANY_TO_DOWNLOAD and id_index < len(target_ids):
    pdb_id = target_ids[id_index]
    id_index += 1
    
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    temp_cif_path = temp_dir / f"{pdb_id}.cif"
    npy_path = data_dir / f"{pdb_id}_ca.npy"

    try:
        print(f"[{successful_count}/{HOW_MANY_TO_DOWNLOAD}] Inspecting {pdb_id}...")
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

        # --- QC --- (continuous backbone, excessive unknowns, and minimum length) 
        ca_coords = []
        actual_sequence = ""
        expected_res_id = None
        is_continuous = True
        unknown_aa_count = 0
        
        for residue in longest_chain:
            # Skip hetero-atoms (water, ligands)
            if residue.get_id()[0] != ' ':
                continue
                
            if 'CA' in residue:
                current_res_id = residue.get_id()[1]
                
                #missing physical gaps
                if expected_res_id is not None and current_res_id != expected_res_id:
                    is_continuous = False
                    break 
                
                try:
                    res_name = residue.get_resname().capitalize()
                    aa_char = protein_letters_3to1.get(res_name, 'X')
                    
                    if aa_char == 'X':
                        unknown_aa_count += 1
                        
                    actual_sequence += aa_char
                    ca_coords.append(residue['CA'].get_coord())
                    expected_res_id = current_res_id + 1
                except KeyError:
                    continue

        if not is_continuous:
            print(f" -> Skipping {pdb_id}: QC Failed (Invisible gap in backbone).")
            continue
            
        if unknown_aa_count > 3:
            print(f" -> Skipping {pdb_id}: QC Failed (Too many unknown 'X' amino acids).")
            continue

        ca_matrix = np.array(ca_coords)
        if ca_matrix.shape[0] < 50:
             print(f" -> Skipping {pdb_id}: QC Failed (Main chain too short).")
             continue
        # -----------------------

        #save cooards
        np.save(npy_path, ca_matrix)
        resolved_sequences[pdb_id] = actual_sequence
        successful_count += 1
        print(f" -> Success! Saved {npy_path.name} | Resolved Length: {len(actual_sequence)}")

    except Exception as e:
        print(f" -> Failed to extract data for {pdb_id}: {e}")
    
    finally:
        if temp_cif_path.exists():
            os.remove(temp_cif_path)

seq_path = data_dir / "sequences.json"
with open(seq_path, "w") as f:
    json.dump(resolved_sequences, f, indent=4)

print(f"\nData pipeline complete! Saved {len(resolved_sequences)} pristine targets to {data_dir}")