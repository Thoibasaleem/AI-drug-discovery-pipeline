import torch
from torch.utils.data import Dataset
import os
from config import Config

class KIBADataset(Dataset):
    def __init__(self, split='train'):
        data_path = os.path.join(Config.PROCESSED_DIR, f'{split}.pt')
        data = torch.load(data_path, weights_only=False)
        
        self.drug_features = data['drug_features']
        self.protein_features = data['protein_features']
        self.labels = data['labels']
        self.metadata = data['metadata']
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'drug': self.drug_features[idx],
            'protein': self.protein_features[idx],
            'label': self.labels[idx],
            'metadata': self.metadata.iloc[idx].to_dict() if hasattr(self.metadata, 'iloc') else {}
        }

def get_dataloaders():
    from torch.utils.data import DataLoader
    
    train_dataset = KIBADataset('train')
    val_dataset = KIBADataset('val')
    test_dataset = KIBADataset('test')
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader