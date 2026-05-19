import torch
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Import your custom modules
from BBDataSet import BBDataset
from model import iFoldResNet

def distance_to_3d(dist_matrix):
    """
    Translates a 2D distance matrix into 3D coordinates using Eigendecomposition.
    """
    N = dist_matrix.shape[0]
    
    # 1. Square the distance matrix
    D_sq = dist_matrix ** 2
    
    # 2. Create the Centering Matrix (J) to move the protein to the origin
    # J = I - (1/N) * 1 * 1^T
    J = np.eye(N) - np.ones((N, N)) / N
    
    # 3. Double-center to calculate the Gram Matrix (G)
    G = -0.5 * J.dot(D_sq).dot(J)
    
    # 4. Eigendecomposition (Finding the spatial basis vectors)
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    
    # Sort eigenvalues and eigenvectors in descending order (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 5. Extract the top 3 dimensions for X, Y, Z space
    # (Clamp negative eigenvalues to 0 to prevent complex numbers if the network's geometry is slightly imperfect)
    top_3_evals = np.maximum(eigenvalues[:3], 0)
    top_3_evecs = eigenvectors[:, :3]
    
    # Coordinates = Eigenvectors * sqrt(Eigenvalues)
    coords = top_3_evecs * np.sqrt(top_3_evals)
    
    return coords


def visualize_protein_3d(true_coords, pred_coords):
    """
    Renders an interactive 3D plot comparing the Ground Truth to the iFold Prediction.
    """
    fig = go.Figure()

    # Ground Truth Trace (Blue)
    fig.add_trace(go.Scatter3d(
        x=true_coords[:, 0], y=true_coords[:, 1], z=true_coords[:, 2],
        mode='lines+markers',
        marker=dict(size=4, color='blue', opacity=0.8),
        line=dict(color='blue', width=4),
        name='Ground Truth (Actual)'
    ))

    # iFold Prediction Trace (Red)
    fig.add_trace(go.Scatter3d(
        x=pred_coords[:, 0], y=pred_coords[:, 1], z=pred_coords[:, 2],
        mode='lines+markers',
        marker=dict(size=4, color='red', opacity=0.8),
        line=dict(color='red', width=4),
        name='iFold Prediction'
    ))

    fig.update_layout(
        title="iFold 3D Reconstruction: Ground Truth vs. Prediction",
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.02, y=0.98)
    )
    
    # Opens the interactive plot in your default web browser
    fig.show()


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data" / "trainingData" / "ca_coords"
    weights_path = script_dir / "ifold_weights.pth"

    if not weights_path.exists():
        print("Error: Could not find ifold_weights.pth.")
        return

    device = torch.device("cpu")
    print("Loading Trained iFold Network...")

    # Load Model
    model = iFoldResNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    # Load Dataset
    print("Loading Test Protein...")
    dataset = BBDataset(data_dir=data_dir, maxlen=350)
    
    # Let's test it on the first protein in the dataset
    features, true_dist, mask = dataset[0]
    seq_len = int(mask[0, :].sum().item())
    
    # Forward Pass
    with torch.no_grad():
        pred_dist = model(features.unsqueeze(0).to(device)).squeeze(0)
    
    # Slice off the padding
    pred_dist = pred_dist[:seq_len, :seq_len].numpy()
    true_dist = true_dist[:seq_len, :seq_len].numpy()

    print("Executing Eigendecomposition to map 2D distances to 3D space...")
    pred_coords = distance_to_3d(pred_dist)
    true_coords = distance_to_3d(true_dist)

    print("Rendering 3D Visualization...")
    visualize_protein_3d(true_coords, pred_coords)

if __name__ == "__main__":
    main()