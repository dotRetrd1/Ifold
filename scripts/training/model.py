import torch.nn as nn
import torch
import torch.nn.functional as F

class dimUp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, N, C = x.shape

        #expand to N x N x C by repeating along new dimensions
        x_j = x.unsqueeze(2).expand(-1, -1, N, -1) # B x N x N x C
        x_i = x.unsqueeze(1).expand(-1, N, -1, -1) # B x N x N x C
        #concat along feature dimension
        pair_representation = torch.cat([x_i, x_j], dim=-1) # B x N x N x 2C

        return pair_representation.permute(0,3,1,2)

class DilatedResidualBlock(nn.Module):
    #2D convolutional block with skip connections and dilation
    def __init__(self, channels, dilation=1):
        super().__init__()
        #padding scale with dilation to keep the N x N grid the same size
        padding = dilation
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        #skip connection: Add the original input back to the processed output
        out += residual
        return F.relu(out)


class iFoldResNet(nn.Module):
    #main architecture class, takes in (B, N, 4) and outputs (B, N, N) distance matrix
    def __init__(self, input_features=4, hidden_channels=64, num_blocks=4):
        super().__init__()
        
        self.dim_jump = dimUp()
        
        #4 features per amino acid * 2 (concatenated) = 8 initial channels
        self.initial_conv = nn.Conv2d(input_features * 2, hidden_channels, kernel_size=1)
        
        blocks = []
        for i in range(num_blocks):
            #increasing dilation (1, 2, 4, 8)
            dilation = 2 ** i 
            blocks.append(DilatedResidualBlock(hidden_channels, dilation=dilation))
            
        self.resnet_blocks = nn.Sequential(*blocks)
        
        #final layer collapse the 64 hidden channels down to 1 distance prediction
        self.final_conv = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

#main
    def forward(self, x):
        #1D to 2D Jump
        out = self.dim_jump(x) # Shape: (B, 8, N, N)
        
        #Expand channels
        out = self.initial_conv(out) # Shape: (B, 64, N, N)
        
        #Deep Dilated Convolutions
        out = self.resnet_blocks(out) # Shape: (B, 64, N, N)
        
        #Output Distance Map
        out = self.final_conv(out) # Shape: (B, 1, N, N)
        
        #Distances cannot be negative, apply ReLU
        out = F.relu(out)
        
        #Squeeze the empty channel dimension: (B, 1, N, N) -> (B, N, N)
        return out.squeeze(1)


# ==========================================
# TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    print("Initializing iFold Architecture...")
    model = iFoldResNet()
    
    # Create a dummy batch of 4 proteins, each padded to 350 length, with 4 features
    dummy_input = torch.randn(4, 350, 4)
    print(f"Dummy Input Shape:  {dummy_input.shape} -> (Batch, N, Features)")
    
    # Pass it through the model
    predicted_distances = model(dummy_input)
    
    print(f"Model Output Shape: {predicted_distances.shape} -> Expected: (Batch, N, N)")
    print("\n--- Architecture Test Successful ---")
    print("The model successfully jumped from 1D sequence data to a 2D distance matrix!")