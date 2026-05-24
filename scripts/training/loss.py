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

    violation = F.relu(d_ij - (d_ik + d_kj)) # B x N x N x N

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

