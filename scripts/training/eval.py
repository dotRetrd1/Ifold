import torch
import matplotlib.pyplot as plt
from pathlib import Path

from BBDataSet import BBDataset
from model import iFoldResNet

def evaluate_ifold():
    #setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data" / "trainingData" / "ca_coords"
    weights_path = script_dir / "ifold_weights.pth"                 #currently just takes the latest weight stored in the training folder

    if not weights_path.exists():
        print("Error: Could not find ifold_weights.pth. chceck that training finished and the weights are in the right place.")
        return

    #I hate having an AMD gpu Ihate haivng an AMD gpu I hate having an AMD gpu
    device = torch.device("cpu")
    print("Loading iFold Architecture...")

    #initialize model and load the trained weights
    model = iFoldResNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    
    model.eval() 

    print("Loading a test protein...")
    dataset = BBDataset(data_dir=data_dir, maxlen=350)
    
    #grab the very first protein in the dataset
    features, true_dist, mask = dataset[0]
    
    features = features.unsqueeze(0).to(device)
    
    #find the actual physical length of this protein (count the True values in the mask)
    seq_len = int(mask[0, :].sum().item())
    print(f"Testing on protein with length {seq_len} Amino Acids.")

    print("Running Inference...")
    with torch.no_grad(): 
        pred_dist = model(features)
        
    #remove batch dimension and slice off the zero-padding so only physical protein
    pred_dist = pred_dist.squeeze(0)[:seq_len, :seq_len].numpy()
    true_dist = true_dist[:seq_len, :seq_len].numpy()

    print("Rendering comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    #plot Ground Truth
    im1 = axes[0].imshow(true_dist, cmap='viridis')
    axes[0].set_title("True Physical Distances (Ground Truth)")
    axes[0].set_xlabel("Residue Index i")
    axes[0].set_ylabel("Residue Index j")
    fig.colorbar(im1, ax=axes[0], label="Distance (Å)")
    
    #plot iFold Prediction
    im2 = axes[1].imshow(pred_dist, cmap='viridis')
    axes[1].set_title("iFold Predictions")
    axes[1].set_xlabel("Residue Index i")
    axes[1].set_ylabel("Residue Index j")
    fig.colorbar(im2, ax=axes[1], label="Distance (Å)")
    
    plt.suptitle("eval map", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_ifold()