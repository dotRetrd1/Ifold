import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os

from BBDataSet import BBDataset
from model import iFoldResNet

os.environ["MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK"] = "0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"
os.environ["MIOPEN_LOG_LEVEL"] = "1"  #Suppress MIOpen warnings

def evaluate_ifold():
    #setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data" / "trainingData" / "ca_coords"
    weights_path = project_root / "data" / "models" / "ifold_weights.pth"                 

    if not weights_path.exists():
        print("Error: Could not find ifold_weights.pth. chceck that training finished and the weights are in the right place.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    #remove batch dimension
    pred_dist = pred_dist.squeeze(0)

    #crop to real sequence length
    pred_dist = pred_dist[:seq_len, :seq_len]
    true_dist = true_dist[:seq_len, :seq_len]
    mask = mask[:seq_len, :seq_len]

    #move to cpu numpy
    pred_dist = pred_dist.cpu().numpy()
    true_dist = true_dist.cpu().numpy()
    mask = mask.cpu().numpy()

    #clean prediction
    pred_dist = (pred_dist + pred_dist.T) / 2.0
    pred_dist = pred_dist * mask

    #metrics
    abs_error = abs(pred_dist - true_dist)

    mae = abs_error[mask].mean()
    rmse = ((abs_error[mask] ** 2).mean()) ** 0.5

    print(f"\n2D Distance Map Metrics:")
    print(f"MAE  : {mae:.4f} Å")
    print(f"RMSE : {rmse:.4f} Å")

    print("Rendering comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    #ground truth
    im1 = axes[0].imshow(true_dist, cmap='viridis')
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("Residue i")
    axes[0].set_ylabel("Residue j")
    fig.colorbar(im1, ax=axes[0])

    #prediction
    im2 = axes[1].imshow(pred_dist, cmap='viridis')
    axes[1].set_title("Prediction")
    axes[1].set_xlabel("Residue i")
    axes[1].set_ylabel("Residue j")
    fig.colorbar(im2, ax=axes[1])

    #error map
    im3 = axes[2].imshow(abs_error, cmap='hot')
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("Residue i")
    axes[2].set_ylabel("Residue j")
    fig.colorbar(im3, ax=axes[2])

    plt.suptitle(
        f"iFold Evaluation | MAE={mae:.2f}Å | RMSE={rmse:.2f}Å",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_ifold()