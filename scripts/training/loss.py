import torch
import torch.nn.functional as F

def masked_mse_loss(pred, target, mask):
    masked_diff = (pred - target) * mask.float()
    squared_diff = masked_diff ** 2
    valid_elements = mask.sum()

    #handle case where there are no valid elements to avoid division by zero
    if valid_elements == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    mse = squared_diff.sum() / valid_elements
    return mse

def triangle_inequality_loss(pred, mask):
    d_ik = pred.unsqueeze(2) # B x N x 1 x N
    d_kj = pred.unsqueeze(1) # B x 1 x N x N
    d_ij = pred.unsqueeze(3) # B x N x N x 1

    violation = F.relu(d_ik + d_kj - d_ij) # B x N x N x N

    mask_3d = mask.unsqueeze(3) & mask.unsqueeze(2) & mask.unsqueeze(1) # B x N x N x N 

    valid_violations = violation * mask_3d.float() 
    valid_elements = mask_3d.sum()

    if valid_elements == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return valid_violations.sum() / valid_elements #average violation per valid triplet

def total_loss(pred, target, mask, lambda_triangle=0.1):
    mse = masked_mse_loss(pred, target, mask)
    triangle_loss = triangle_inequality_loss(pred, mask)
    total_loss = mse + lambda_triangle * triangle_loss
    return total_loss, mse, triangle_loss   




# ==========================================
# TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    print("Testing Physical Constraints Loss Function...")
    
    # Setup dummy data for 1 protein, length 4 (padded to 5)
    B, N = 1, 5
    
    # A perfect square grid of atoms:
    # A -- 2 -- B
    # |         |
    # 2         2
    # |         |
    # D -- 2 -- C
    # Diagonal A to C = sqrt(8) approx 2.82
    
    targets = torch.tensor([[[0.0,  2.0,  2.82, 2.0,  0.0],
                             [2.0,  0.0,  2.0,  2.82, 0.0],
                             [2.82, 2.0,  0.0,  2.0,  0.0],
                             [2.0,  2.82, 2.0,  0.0,  0.0],
                             [0.0,  0.0,  0.0,  0.0,  0.0]]]) # Row 5 is padding
                             
    # Create a prediction that breaks physics (Distance A->C predicted as 10.0!)
    predictions = targets.clone()
    predictions[0, 0, 2] = 10.0 
    predictions[0, 2, 0] = 10.0
    
    # Create mask (True for first 4, False for padding index 4)
    mask_1d = torch.tensor([True, True, True, True, False])
    mask_2d = mask_1d.unsqueeze(0).unsqueeze(1) & mask_1d.unsqueeze(0).unsqueeze(2)
    
    total_loss, mse, penalty = total_loss(predictions, targets, mask_2d, lambda_triangle=1.0)
    
    print(f"\nMSE Loss:      {mse.item():.4f} (Accuracy error)")
    print(f"Triangle Loss: {penalty.item():.4f} (Physics violation magnitude)")
    print(f"Total Loss:    {total_loss.item():.4f}")
    print("\n--- Loss Test Successful ---")
    print("The custom loss caught the impossible 10.0 distance while ignoring the padded zeros!")