import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from BBDataSet import BBDataset
from model import iFoldResNet
from loss import total_loss as ifold_loss

def train_ifold():
    #setup paths and hyperparams
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data" / "trainingData" / "ca_coords"
    
    BATCH_SIZE = 15
    LEARNING_RATE = 5e-5
    EPOCHS = 100
    
    # -----------------------------------------------------------------
    # TRANSFER LEARNING SWITCH:
    # to load pre-trained weights, put the filename here
    # Otherwise, leave it as None to train a fresh brain.
    # Example: TRANSFER_LEARNING_CHECKPOINT = "ifold_weights_general.pth"
    # -----------------------------------------------------------------
    TRANSFER_LEARNING_CHECKPOINT = "ifold_checkpoint_epoch_30.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}\n")

    print("Loading Dataset...")
    dataset = BBDataset(data_dir=data_dir, maxlen=350)
    
    if len(dataset) == 0:
        print("Error: no data found. Make sure buildDataset.py and prepData.py ran successfully")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset loaded, Total proteins: {len(dataset)} | Batches per epoch: {len(dataloader)}")

    print("Initializing Model...")
    model = iFoldResNet().to(device)
    
    #load previous weights if Transfer Learning is active
    if TRANSFER_LEARNING_CHECKPOINT:
        weight_path = script_dir / TRANSFER_LEARNING_CHECKPOINT
        if weight_path.exists():
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
            print(f"--> SUCCESSFULLY LOADED PRE-TRAINED WEIGHTS: {TRANSFER_LEARNING_CHECKPOINT}")
        else:
            print(f"--> ERROR: Could not find {TRANSFER_LEARNING_CHECKPOINT}. Starting fresh.")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #training loop
    print("\n-------------------- TRAINING --------------------")
    
    for epoch in range(EPOCHS):
        model.train() 
        
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_triangle = 0.0
        
        for batch_idx, (features, targets, masks) in enumerate(dataloader):
            features = features.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            #(clear old memory)
            optimizer.zero_grad()
            
            #forward Pass
            predictions = model(features)
            
            total_loss, mse, penalty = ifold_loss(predictions, targets, masks, lambda_triangle=7.5)
            #backward Pass
            total_loss.backward()
            
            #update the weights
            optimizer.step()
            
            #for logging
            epoch_loss += total_loss.item()
            epoch_mse += mse.item()
            epoch_triangle += penalty.item()
            
        avg_loss = epoch_loss / len(dataloader)
        avg_mse = epoch_mse / len(dataloader)
        avg_triangle = epoch_triangle / len(dataloader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Total Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | Triangle: {avg_triangle:.4f}")

        #checkpoint (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            checkpoint_path = script_dir / f"ifold_checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    [!] Backup Checkpoint saved: {checkpoint_path.name}")

    #finally save
    save_path = script_dir / "ifold_weights_final.pth"
    torch.save(model.state_dict(), save_path)
    print("\n-------------------- TRAINING COMPLETE --------------------")
    print(f"Model weights successfully saved to: {save_path.name}")

if __name__ == "__main__":
    train_ifold()