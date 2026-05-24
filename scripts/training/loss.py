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

def triangle_inequality_loss(pred, mask, chunk_size):
    B, N, _ = pred.shape
    total_violation = 0.0
    total_valid_elements = 0.0
    
    d_kj = pred.unsqueeze(1)
    mask_kj = mask.unsqueeze(1)
    
    for start_i in range(0, N, chunk_size):
        end_i = min(start_i + chunk_size, N)
        
        d_ik_chunk = pred[:, start_i:end_i, :].unsqueeze(2)
        d_ij_chunk = pred[:, start_i:end_i, :].unsqueeze(3)
        
        mask_ik_chunk = mask[:, start_i:end_i, :].unsqueeze(2)
        mask_ij_chunk = mask[:, start_i:end_i, :].unsqueeze(3)
        
        violation_chunk = F.relu(d_ij_chunk - (d_ik_chunk + d_kj))
        mask_3d_chunk = mask_ij_chunk & mask_ik_chunk & mask_kj
        
        valid_violations_chunk = violation_chunk * mask_3d_chunk.float()
        
        total_violation += valid_violations_chunk.sum()
        total_valid_elements += mask_3d_chunk.sum()

    if total_valid_elements == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return total_violation / total_valid_elements

def total_loss(pred, target, mask, chunk_size, lambda_triangle=0.1):
    mse = masked_mse_loss(pred, target, mask)
    triangle_loss = triangle_inequality_loss(pred, mask, chunk_size) 
    total_loss = mse + lambda_triangle * triangle_loss
    return total_loss, mse, triangle_loss

