import torch
import torch.optim as optim
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
    
    BATCH_SIZE = config["training"]["batch_size"]
    LEARNING_RATE = config["training"]["learning_rate"]
    EPOCHS = config["training"]["epochs"]
    TRANSFER_LEARNING_CHECKPOINT = config["training"]["pretrained_model"]
    
    if (not input("Start with a pretrained model? (y/n): ").lower() == 'y'):
        TRANSFER_LEARNING_CHECKPOINT = None
    elif(not input("is {TRANSFER_LEARNING_CHECKPOINT} the right model? (y/n): ").lower() == 'y'):
            TRANSFER_LEARNING_CHECKPOINT = input("Enter the filename of the checkpoint (e.g., ifold_weights_general.pth): ")
            print(f"--> TRANSFER LEARNING ENABLED: {TRANSFER_LEARNING_CHECKPOINT}")
    else:
        print(f"--> TRANSFER LEARNING ENABLED: {TRANSFER_LEARNING_CHECKPOINT}")
        
        
    
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
        weight_path = project_root / "data" / "trainingData" / "models" / TRANSFER_LEARNING_CHECKPOINT
        if weight_path.exists():
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
            print(f"--> SUCCESSFULLY LOADED PRE-TRAINED WEIGHTS: {TRANSFER_LEARNING_CHECKPOINT}")
        else:
            print(f"--> Could not find {TRANSFER_LEARNING_CHECKPOINT}. making new weights")
    
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

        #save (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            checkpoint_path = weight_path / "recentTrain" / f"ifold_checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[!] Backup Checkpoint saved: {checkpoint_path.name}")

    save_path = weight_path / "recentTrain" / f"ifold_weights_epoch_{EPOCHS}.pth"
    torch.save(model.state_dict(), save_path)
    print("\n-------------------- TRAINING COMPLETE --------------------")
    print(f"Model weights successfully saved to: {save_path.name}")

if __name__ == "__main__":
    train_ifold()