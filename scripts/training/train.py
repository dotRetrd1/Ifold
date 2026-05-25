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
    mode = input("would you like to train on general proteins(1) or just Beta Barrels(2)? ").strip().lower()
    if mode == "1":
        data_dir = project_root / "data" / "trainingData" / "ca_coords"
    else:
        data_dir = project_root / "data" / "trainingData" / "BBData_ca_coords"
    weight_path = project_root / "data" / "models"
    
    BATCH_SIZE = config["training"]["batch_size"]
    LEARNING_RATE = config["training"]["learning_rate"]
    EPOCHS = config["training"]["epochs"]
    CHUNK_SIZE = config["training"]["loss_chunk_size"]
    LAMBDA_TRIANGLE = config["training"]["lambda_triangle"]
    LAMBDA_LOCAL = config["training"]["lambda_local"]
    LAMBDA_CONTACT = config["training"]["lambda_contact"]
    USE_PRETRAINED = config["training"].get("use_pretrained", False)
    TRANSFER_LEARNING_CHECKPOINT = config["training"]["pretrained_model"]
    CHECKPOINT_INTERVAL = config["training"].get("checkpoint_interval", 10)
    
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
        
        if USE_PRETRAINED:
            current_triangle = LAMBDA_TRIANGLE
            current_local = LAMBDA_LOCAL
            current_contact = LAMBDA_CONTACT

        else:
            if epoch < 3:
                current_triangle = 0.0
                current_local = 1.0

            elif epoch < 10:
                current_triangle = 0.005
                current_local = 2.0

            else:
                current_triangle = LAMBDA_TRIANGLE
                current_local = LAMBDA_LOCAL

        epoch_loss = 0.0
        epoch_l1 = 0.0
        #epoch_mse = 0.0
        epoch_triangle = 0.0
        epoch_local = 0.0
        epoch_contact = 0.0
        
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
                loss, l1, local_loss, triangle_loss, contact_loss = ifold_loss(predictions, targets, masks, chunk_size=CHUNK_SIZE, lambda_triangle=current_triangle, lambda_local=current_local, lambda_contact=current_contact)
            
            #backward Pass
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            #update the weights
            scaler.step(optimizer)
            scaler.update()
            
            #for logging
            epoch_loss += loss.item()
            epoch_l1 += l1.item()
            epoch_local += local_loss.item()
            epoch_triangle += triangle_loss.item()
            epoch_contact += contact_loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        avg_l1 = epoch_l1 / len(dataloader)
        avg_local = epoch_local / len(dataloader)
        avg_triangle = epoch_triangle / len(dataloader)
        avg_contact = epoch_contact / len(dataloader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Total Loss: {avg_loss:.4f} | L1: {avg_l1:.4f} | Local: {avg_local:.4f} | Triangle: {avg_triangle:.4f} | Contact: {avg_contact:.4f}")

        #save (every CHECKPOINT_INTERVAL epochs)
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = weight_path / "recentTrain" / f"ifold_checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[!] Backup Checkpoint saved: {checkpoint_path.name}")

    save_path = weight_path / "recentTrain" / f"ifold_weights.pth"
    torch.save(model.state_dict(), save_path)
    print("\n-------------------- TRAINING COMPLETE --------------------")
    print(f"Model weights successfully saved to: {save_path.name}")

if __name__ == "__main__":
    train_ifold()