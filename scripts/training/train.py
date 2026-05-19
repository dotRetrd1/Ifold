import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from BBDataSet import BBDataset
from model import iFoldResNet
from loss import total_loss as ifold_loss

def train_ifold():
    #paths and hyperparams
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data" / "trainingData" / "ca_coords"
    
    BATCH_SIZE = 15
    LEARNING_RATE = 3e-4
    EPOCHS = 100
    
    #Did I say I dont like having an AMD gpu anymore? (this doenst do shit, its just gonna be the cpu cause f AMD ig)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}\n") #cpu

    print("Loading Dataset...")
    dataset = BBDataset(data_dir=data_dir, maxlen=350)
    
    if len(dataset) == 0:
        print("Error: no data found. Make sure buildDataset.py and prepData.py ran successfully")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset loaded, Total proteins: {len(dataset)} | Batches per epoch: {len(dataloader)}")

    print("Initializing Model...")
    model = iFoldResNet().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #the Training Loop
    print("\n-------------------- TRAINING --------------------")
    
    for epoch in range(EPOCHS):
        model.train() #set model to training mode
        
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_triangle = 0.0
        
        for batch_idx, (features, targets, masks) in enumerate(dataloader):
            #move data to GPU/CPU
            features = features.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            #(clear old memory)
            optimizer.zero_grad()
            
            #forward Pass (Predict)
            predictions = model(features)
            
            total_loss, mse, penalty = ifold_loss(predictions, targets, masks, lambda_triangle=10.0)
            
            #backward Pass (Calculate how to fix the errors)
            total_loss.backward()
            
            #update the weights
            optimizer.step()
            
            #accumulate metrics for logging
            epoch_loss += total_loss.item()
            epoch_mse += mse.item()
            epoch_triangle += penalty.item()
            
        #calculate epoch averages
        avg_loss = epoch_loss / len(dataloader)
        avg_mse = epoch_mse / len(dataloader)
        avg_triangle = epoch_triangle / len(dataloader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Total Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | Triangle: {avg_triangle:.4f}")

    #save the trained weights
    save_path = script_dir / "ifold_weights.pth"
    torch.save(model.state_dict(), save_path)
    print("\n-------------------- TRAINING COMPLETE --------------------")
    print(f"Model weights successfully saved to: {save_path.name}")

if __name__ == "__main__":
    train_ifold()