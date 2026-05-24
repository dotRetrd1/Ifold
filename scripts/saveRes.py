import json
from datetime import datetime

def save_output(save_dir, sequence, pdb_id, pred_coords, fig):
    save_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pdb_id": pdb_id if pdb_id else "None",
        "sequence_length": len(sequence),
        "sequence": sequence,
        "files_generated": ["metadata.json", "interactive_plot.html", "prediction.pdb"]
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    if fig:
        fig.write_html(str(save_dir / "interactive_plot.html"))
        
    #(.pdb format)
    AA_MAP = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'}
    
    with open(save_dir / "prediction.pdb", "w") as f:
        for i, (coord, aa) in enumerate(zip(pred_coords, sequence)):
            res = AA_MAP.get(aa.upper(), 'UNK')
            f.write(f"ATOM  {i+1:>5}  CA  {res:>3} A{i+1:>4}    {coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}  1.00  0.00           C  \n")
