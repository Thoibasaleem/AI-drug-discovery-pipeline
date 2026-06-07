import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import pickle
from tqdm import tqdm
from config import Config
import os

class KIBAPreprocessor:
    def __init__(self):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_dict = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.scaler = StandardScaler()
        
    def parse_kiba_file(self, filepath):
        """Parse the space-separated KIBA file"""
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    drug_id = parts[0]
                    protein_id = parts[1]
                    smiles = parts[2]
                    sequence = parts[3]
                    affinity = float(parts[4])
                    
                    data.append({
                        'drug_id': drug_id,
                        'protein_id': protein_id,
                        'smiles': smiles,
                        'sequence': sequence.upper(),
                        'affinity': affinity
                    })
        return pd.DataFrame(data)
    
    def smiles_to_morgan(self, smiles):
        """Convert SMILES to Morgan fingerprint"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(Config.MORGAN_BITS)
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=Config.MORGAN_RADIUS,
                nBits=Config.MORGAN_BITS
            )
            return np.array(fp)
        except:
            return np.zeros(Config.MORGAN_BITS)
    
    def sequence_to_features(self, sequence):
        """Convert protein sequence to amino acid composition features"""
        comp = np.zeros(20)
        
        seq_len = min(len(sequence), Config.MAX_SEQ_LENGTH)
        truncated_seq = sequence[:Config.MAX_SEQ_LENGTH]

        # ✅ FIXED INDENTATION
        for aa in truncated_seq:
            if aa in self.aa_dict:
                comp[self.aa_dict[aa]] += 1

        if seq_len > 0:
            comp = comp / seq_len

        return comp
    
    def process(self, input_file):
        print("Loading data...")
        df = self.parse_kiba_file(input_file)
        print(f"Loaded {len(df)} samples")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['drug_id', 'protein_id'])
        print(f"After deduplication: {len(df)} samples")
        
        # Process drugs
        print("Processing drug features (Morgan fingerprints)...")
        drug_features = []
        for smi in tqdm(df['smiles']):
            drug_features.append(self.smiles_to_morgan(smi))
        drug_features = np.array(drug_features)
        
        # Process proteins
        print("Processing protein features (AA composition)...")
        protein_features = []
        for seq in tqdm(df['sequence']):
            protein_features.append(self.sequence_to_features(seq))
        protein_features = np.array(protein_features)
        
        # Labels
        labels = df['affinity'].values.reshape(-1, 1)
        
        # Normalize features
        drug_features = self.scaler.fit_transform(drug_features)
        
        # Save scaler
        os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
        with open(os.path.join(Config.PROCESSED_DIR, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Train/Val/Test split
        indices = np.arange(len(df))
        train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
        
        splits = {
            'train': (train_idx, drug_features[train_idx], protein_features[train_idx], labels[train_idx]),
            'val': (val_idx, drug_features[val_idx], protein_features[val_idx], labels[val_idx]),
            'test': (test_idx, drug_features[test_idx], protein_features[test_idx], labels[test_idx])
        }
        
        for split_name, (idx, d_feat, p_feat, lab) in splits.items():
            torch.save({
                'indices': idx,
                'drug_features': torch.FloatTensor(d_feat),
                'protein_features': torch.FloatTensor(p_feat),
                'labels': torch.FloatTensor(lab),
                'metadata': df.iloc[idx].reset_index(drop=True)
            }, os.path.join(Config.PROCESSED_DIR, f'{split_name}.pt'))
        
        print("Preprocessing complete!")
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        return df


if __name__ == "__main__":
    preprocessor = KIBAPreprocessor()
    preprocessor.process(Config.DATA_PATH)