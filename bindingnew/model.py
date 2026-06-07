import torch
import torch.nn as nn
from config import Config

class DTIModel(nn.Module):
    def __init__(self):
        super(DTIModel, self).__init__()
        
        # Drug branch: 2048 -> 512 -> 256
        self.drug_encoder = nn.Sequential(
            nn.Linear(Config.MORGAN_BITS, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Protein branch: 20 -> 128 -> 256
        self.protein_encoder = nn.Sequential(
            nn.Linear(Config.AA_FEATURES, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, drug_features, protein_features):
        drug_out = self.drug_encoder(drug_features)
        prot_out = self.protein_encoder(protein_features)
        combined = torch.cat([drug_out, prot_out], dim=1)
        return self.fusion(combined)

def get_model():
    model = DTIModel().to(Config.DEVICE)
    return model