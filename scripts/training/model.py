import yaml
import torch.nn as nn
import torch
import torch.nn.functional as F
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
config_path = project_root / "config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

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
    
class RelativePositionEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, N, device):
        indices = torch.arange(N, device=device)

        #pairwise |i-j|
        rel_dist = torch.abs(
            indices.unsqueeze(0) - indices.unsqueeze(1)
        ).float()

        #normalize to [0,1]
        rel_dist = rel_dist / float(max(N - 1, 1))

        #shape: (1,1,N,N)
        return rel_dist.unsqueeze(0).unsqueeze(0)

class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()

        padding = dilation

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation
        )

        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        return F.relu(out)

input_feat = config['model']['input_features']
hidden = config['model']['hidden_channels']
blocks = config['model']['num_blocks']

class iFoldResNet(nn.Module):
    #takes in (B, N, 4) ; out (B, N, N) distance matrix
    def __init__(self, input_features=input_feat, hidden_channels=hidden, num_blocks=blocks):
        super().__init__()
        
        self.embedding_dim = 16

        #21 residue types including unknown token
        self.embedding = nn.Embedding(
            21,
            self.embedding_dim,
            padding_idx=20
        )

        self.dim_jump = dimUp()
        self.rel_pos = RelativePositionEncoding()

        #two embedded residues concatenated
        self.initial_conv = nn.Conv2d(
            self.embedding_dim * 2 + 1,
            hidden_channels,
            kernel_size=1
        )
        
        blocks = []
        for i in range(num_blocks):
            #[1, 2, 4, 8, 16, 32]
            dilation = min(2 ** i, 32)
            blocks.append(DilatedResidualBlock(hidden_channels, dilation=dilation))
            
        self.resnet_blocks = nn.Sequential(*blocks)
        
        #final layer collapse the 64 hidden channels down to 1 distance prediction
        self.final_conv = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

#main
    def forward(self, x):
        x = self.embedding(x)

        B, N, C = x.shape

        #pairwise residue features
        out = self.dim_jump(x)

        #relative sequence distance channel
        rel_pos = self.rel_pos(N, x.device)

        #expand across batch
        rel_pos = rel_pos.expand(B, -1, -1, -1)

        #append as extra channel
        out = torch.cat([out, rel_pos], dim=1)
        out = self.initial_conv(out) #shape: (B, 64, N, N)
        #Convolutions
        out = self.resnet_blocks(out) #shape: (B, 64, N, N)
        #Output Distance Map
        out = self.final_conv(out) #shape: (B, 1, N, N)
        out = (out + out.transpose(-1, -2)) / 2 #Symmetrize the output
        out = F.relu(out)
        out = torch.clamp(out, 0.0, 32.0)
        #(B, 1, N, N) -> (B, N, N)
        return out.squeeze(1)
