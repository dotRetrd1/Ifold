import numpy as np
import plotly.graph_objects as go


def distance_to_3d(dist_matrix):

    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.maximum(dist_matrix, 0.0)

    N = dist_matrix.shape[0]

    D_sq = dist_matrix ** 2

    J = np.eye(N) - np.ones((N, N)) / N

    G = -0.5 * J.dot(D_sq).dot(J)

    eigenvalues, eigenvectors = np.linalg.eigh(G)

    idx = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    top_3_evals = np.maximum(eigenvalues[:3], 1e-8)
    top_3_evecs = eigenvectors[:, :3]

    coords = top_3_evecs @ np.diag(np.sqrt(top_3_evals))

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


def visualize_protein_3d(pred_coords, true_coords=None):
    fig = go.Figure()

    if true_coords is not None:
        pred_len = pred_coords.shape[0]
        true_len = true_coords.shape[0]

        min_len = min(pred_len, true_len)

        if pred_len != true_len:
            print(f"[!] Shape mismatch: Prediction is {pred_coords.shape}, Ground Truth is {true_coords.shape}.")
            print(f"[!] Aligning first {min_len} residues.")

        #align only overlapping region
        aligned_overlap = kabsch_align(
            pred_coords[:min_len],
            true_coords[:min_len]
        )

        #replace aligned section back into prediction
        pred_coords[:min_len] = aligned_overlap

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
    
    from pathlib import Path
    import tempfile
    import webbrowser
    import os
    import time

    temp_dir = tempfile.gettempdir()
    temp_path = Path(temp_dir) / "protein_view.html"

    fig.write_html(temp_path)

    webbrowser.open(temp_path.as_uri())

    #give browser a moment to open file
    time.sleep(2)

    #delete temp file
    if temp_path.exists():
        os.remove(temp_path)
    return fig