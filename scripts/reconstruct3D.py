import numpy as np
import plotly.graph_objects as go

""""" TO DO: Set up config instead of pathing and hardcode
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
"""

def distance_to_3d(dist_matrix):

#translate 2D distance matrix to 3D coors (eigendecomposition of the Gram matrix)
    N = dist_matrix.shape[0]
    
    D_sq = dist_matrix ** 2
    #create the Centering Matrix (J) to move the protein to the origin
    #J = I - (1/N) * 1 * 1^T
    J = np.eye(N) - np.ones((N, N)) / N
    
    #calc the Gram Matrix (G) (double centering)
    G = -0.5 * J.dot(D_sq).dot(J)
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    
    #sortin descending order (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    #(prevent complex numbers)
    top_3_evals = np.maximum(eigenvalues[:3], 0)
    top_3_evecs = eigenvectors[:, :3]
    
    coords = top_3_evecs * np.sqrt(top_3_evals)
    
    return coords

def kabsch_align(P, Q):
    """
    Aligns coordinate matrix P (Prediction) to Q (Ground Truth).
    Both matrices must have shape (N, 3).
    """
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    #Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    #Calculate the covariance matrix
    H = p.T @ q

    U, S, Vt = np.linalg.svd(H)

    #calc the optimal rot matrix
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    #apply and translate back
    P_aligned = (R @ p.T).T + centroid_Q
    
    return P_aligned


def visualize_protein_3d(true_coords, pred_coords):
    fig = go.Figure()

    #Apply Kabsch Alignment (if true_coords exist)
    if true_coords is not None:
        if true_coords.shape == pred_coords.shape:
            pred_coords = kabsch_align(pred_coords, true_coords)
        else:
            print(f"[!] Shape mismatch: Prediction is {pred_coords.shape}, Ground Truth is {true_coords.shape}.")
            print("[!] Skipping 3D alignment. Displaying Prediction only.")
            true_coords = None #so doesnt crash

    #iFold Prediction Trace (Red)
    fig.add_trace(go.Scatter3d(
        x=pred_coords[:, 0], y=pred_coords[:, 1], z=pred_coords[:, 2],
        mode='lines+markers',
        marker=dict(size=4, color='red', opacity=0.8),
        line=dict(color='red', width=4),
        name='iFold Prediction'
    ))

    #Ground truth trace (Blue)
    if true_coords is not None:
        fig.add_trace(go.Scatter3d(
            x=true_coords[:, 0], y=true_coords[:, 1], z=true_coords[:, 2],
            mode='lines+markers',
            marker=dict(size=4, color='blue', opacity=0.8),
            line=dict(color='blue', width=4),
            name='Ground Truth (Actual)'
        ))

    fig.update_layout(
        title="iFold 3D Reconstruction",
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.02, y=0.98)
    )
    
    fig.show()
    return fig