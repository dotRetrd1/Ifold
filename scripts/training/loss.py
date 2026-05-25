import torch
import torch.nn.functional as F
"""""
def masked_mse_loss(pred, target, mask):
    masked_diff = (pred - target) * mask.float()
    squared_diff = masked_diff ** 2
    valid_elements = mask.sum()

    #handle case where there are no valid elements to avoid division by zero
    if valid_elements == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    mse = squared_diff.sum() / valid_elements
    return mse
"""
def masked_l1_loss(pred, target, mask):
    masked_diff = (pred - target) * mask.float()
    abs_diff = torch.abs(masked_diff)
    valid_elements = mask.sum()

    #handle case where there are no valid elements to avoid division by zero
    if valid_elements == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    l1 = abs_diff.sum() / valid_elements
    return l1

def triangle_inequality_loss(pred, mask, chunk_size):
    B, N, _ = pred.shape
    total_violation = 0.0
    total_valid_elements = 0.0
    
    
    
    for start_i in range(0, N, chunk_size):
        end_i = min(start_i + chunk_size, N)

        d_kj = pred.unsqueeze(1)
        mask_kj = mask.unsqueeze(1)
        
        d_ik_chunk = pred[:, start_i:end_i, :].unsqueeze(2)
        d_ij_chunk = pred[:, start_i:end_i, :].unsqueeze(3)
        
        mask_ik_chunk = mask[:, start_i:end_i, :].unsqueeze(2)
        mask_ij_chunk = mask[:, start_i:end_i, :].unsqueeze(3)
        
        violation_chunk = F.relu(d_ij_chunk - (d_ik_chunk + d_kj))
        mask_3d_chunk = mask_ij_chunk & mask_ik_chunk & mask_kj
        
        valid_violations_chunk = violation_chunk * mask_3d_chunk.float()
        
        num_violations = (valid_violations_chunk > 0).float().sum()
        
        total_violation += valid_violations_chunk.sum()
        total_valid_elements += num_violations

    if total_valid_elements == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
    return total_violation / (total_valid_elements + 1e-8)

def local_distance_loss(pred, target, mask, local_radius=6):
    B, N, _ = pred.shape

    device = pred.device

    indices = torch.arange(N, device=device)

    #|i-j|
    seq_sep = torch.abs(
        indices.unsqueeze(0) - indices.unsqueeze(1)
    )

    local_mask = seq_sep <= local_radius
    local_mask = local_mask.unsqueeze(0) #(1,N,N)

    combined_mask = mask & local_mask

    masked_diff = torch.abs(pred - target) * combined_mask.float()

    valid = combined_mask.sum()

    if valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return masked_diff.sum() / valid

def total_loss(
    pred,
    target,
    mask,
    chunk_size,
    lambda_triangle=0.02,
    lambda_local=2.0
):
    l1 = masked_l1_loss(pred, target, mask)

    local_loss = local_distance_loss(
        pred,
        target,
        mask,
        local_radius=6
    )

    triangle_loss = triangle_inequality_loss(
        pred,
        mask,
        chunk_size
    )

    total_loss = (
        l1
        + lambda_local * local_loss
        + lambda_triangle * triangle_loss
    )

    return total_loss, l1, local_loss, triangle_loss

