import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

from BBDataSet import BBDataset
from model import iFoldResNet
from loss import total_loss as ifold_loss

def train_ifold():
    #setup paths and hyperparams
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data" / "trainingData" / "ca_coords"
    weight_path = project_root / "data" / "models"
    
    BATCH_SIZE = config["training"]["batch_size"]
    LEARNING_RATE = config["training"]["learning_rate"]
    EPOCHS = config["training"]["epochs"]
    CHUNK_SIZE = config["training"]["loss_chunk_size"]
    USE_PRETRAINED = config["training"].get("use_pretrained", False)
    TRANSFER_LEARNING_CHECKPOINT = config["training"]["pretrained_model"]
    
    if USE_PRETRAINED:
        TRANSFER_LEARNING_CHECKPOINT = config["training"]["pretrained_model"]
        print(f"--> TRANSFER LEARNING ENABLED: {TRANSFER_LEARNING_CHECKPOINT}")
    else:
        TRANSFER_LEARNING_CHECKPOINT = None
        print("--> TRANSFER LEARNING DISABLED: Training from scratch")
        
        
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}\n")

    print("Loading Dataset...")
    dataset = BBDataset(data_dir=data_dir, maxlen=350)
    
    if len(dataset) == 0:
        print("Error: no data found; Make sure buildDataset.py and prepData.py ran successfully")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset loaded, Total proteins: {len(dataset)} | Batches per epoch: {len(dataloader)}")

    print("Initializing Model...")
    model = iFoldResNet().to(device)
    
    #load previous weights if Transfer Learning is active
    if TRANSFER_LEARNING_CHECKPOINT:
        checkpoint_file = weight_path / TRANSFER_LEARNING_CHECKPOINT
        if checkpoint_file.exists():
            model.load_state_dict(torch.load(checkpoint_file, map_location=device, weights_only=True))
            print(f"--> SUCCESSFULLY LOADED PRE-TRAINED WEIGHTS: {TRANSFER_LEARNING_CHECKPOINT}")
        else:
            print(f"--> Could not find {TRANSFER_LEARNING_CHECKPOINT}. Making new weights.")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(device.type, enabled=(device.type == 'cuda'))

    #training loop
    print("\n-------------------- TRAINING --------------------")
    
    for epoch in range(EPOCHS):
        model.train() 
        
        epoch_loss = 0.0
        epoch_l1 = 0.0
        #epoch_mse = 0.0
        epoch_triangle = 0.0
        
        for batch_idx, (features, targets, masks) in enumerate(dataloader):
            features = features.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            #(clear old memory)
            optimizer.zero_grad()
            
            #forward Pass
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                predictions = model(features)
                #l1 instead of mse
                total_loss, l1, penalty = ifold_loss(predictions, targets, masks, chunk_size=CHUNK_SIZE, lambda_triangle=7.5)
            
            #backward Pass
            scaler.scale(total_loss).backward()
            
            #update the weights
            scaler.step(optimizer)
            scaler.update()
            
            #for logging
            epoch_loss += total_loss.item()
            epoch_l1 += l1.item()
            epoch_triangle += penalty.item()
            
        avg_loss = epoch_loss / len(dataloader)
        avg_l1 = epoch_l1 / len(dataloader)
        avg_triangle = epoch_triangle / len(dataloader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Total Loss: {avg_loss:.4f} | L1: {avg_l1:.4f} | Triangle: {avg_triangle:.4f}")

        #save (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            checkpoint_path = weight_path / "recentTrain" / f"ifold_checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[!] Backup Checkpoint saved: {checkpoint_path.name}")

    save_path = weight_path / "recentTrain" / f"ifold_weights.pth"
    torch.save(model.state_dict(), save_path)
    print("\n-------------------- TRAINING COMPLETE --------------------")
    print(f"Model weights successfully saved to: {save_path.name}")

if __name__ == "__main__":
    train_ifold()