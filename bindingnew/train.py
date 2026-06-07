import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
from config import Config
from model import get_model
from dataset import get_dataloaders

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        drug = batch['drug'].to(device)
        protein = batch['protein'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(drug, protein)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in loader:
            drug = batch['drug'].to(device)
            protein = batch['protein'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(drug, protein)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    
    return total_loss / len(loader), np.array(predictions), np.array(actuals)

def main():
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # Model
    model = get_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        early_stopping(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(Config.MODEL_DIR, 'best_model.pt'))
            print(f"Saved best model with val_loss: {val_loss:.4f}")
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Save training history
    import json
    with open(os.path.join(Config.RESULTS_DIR, 'history.json'), 'w') as f:
        json.dump(history, f)
    
    print("Training complete!")

if __name__ == "__main__":
    main()