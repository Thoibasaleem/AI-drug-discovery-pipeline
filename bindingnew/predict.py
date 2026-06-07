import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import os
from config import Config

class KIBAPredictor:
    def __init__(self):
        self.device = Config.DEVICE
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_dict = {aa: i for i, aa in enumerate(self.amino_acids)}
        
    def _load_model(self):
        from model import DTIModel
        model = DTIModel().to(self.device)
        checkpoint = torch.load(os.path.join(Config.MODEL_DIR, 'best_model.pt'), map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def _load_scaler(self):
        with open(os.path.join(Config.PROCESSED_DIR, 'scaler.pkl'), 'rb') as f:
            return pickle.load(f)
    
    def _smiles_to_fingerprint(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, Config.MORGAN_RADIUS, Config.MORGAN_BITS)
        return np.array(fp).reshape(1, -1)
    
    def _sequence_to_features(self, sequence):
        sequence = sequence.upper()[:Config.MAX_SEQ_LENGTH]
        comp = np.zeros(20)
        for aa in sequence:
            if aa in self.aa_dict:
                comp[self.aa_dict[aa]] += 1
        if len(sequence) > 0:
            comp = comp / len(sequence)
        return comp.reshape(1, -1)
    
    def predict(self, smiles, sequence):
        """
        Predict binding affinity for a drug-protein pair
        
        Args:
            smiles: Drug SMILES string
            sequence: Protein amino acid sequence
            
        Returns:
            Predicted KIBA affinity score (float)
        """
        # Preprocess
        drug_feat = self._smiles_to_fingerprint(smiles)
        prot_feat = self._sequence_to_features(sequence)
        
        # Scale
        drug_feat = self.scaler.transform(drug_feat)
        
        # To tensors
        drug_tensor = torch.FloatTensor(drug_feat).to(self.device)
        prot_tensor = torch.FloatTensor(prot_feat).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(drug_tensor, prot_tensor)
        
        return prediction.item()

def main():
    predictor = KIBAPredictor()
    
    # Example usage with your data format
    test_smiles = "COC1=C(C=C2C(=C1)CCN=C2C3=CC(=C(C=C3)Cl)Cl)Cl"
    test_sequence = "MTVKTEAAKGTLTYSRMRGMVAILIAFMKQRRMGLNDFIQKIANNSYACKHPEVQSILKISQPQEPELMNANPSPPPSPSQQINLGPSSNPHAKPSDFHFLKVIGKGSFGKVLLARHKAEEVFYAVKVLQKKAILKKKEEKHIMSERNVLLKNVKHPFLVGLHFSFQTADKLYFVLDYINGGELFYHLQRERCFLEPRARFYAAEIASALGYLHSLNIVYRDLKPENILLDSQGHIVLTDFGLCKENIEHNSTTSTFCGTPEYLAPEVLHKQPYDRTVDWWCLGAVLYEMLYGLPPFYSRNTAEMYDNILNKPLQLKPNITNSARHLLEGLLQKDRTKRLGAKDDFMEIKSHVFFSLINWDDLINKKITPPFNPNVSGPNDLRHFDPEFTEEPVPNSIGKSPDSVLVTASVKEAAEAFLGFSYAPPTDSFL"
    
    affinity = predictor.predict(test_smiles, test_sequence)
    print(f"Predicted KIBA Affinity: {affinity:.4f}")
    
    # Interactive mode
    print("\nEnter your own data (or 'quit' to exit):")
    while True:
        smiles = input("SMILES (or 'quit'): ").strip()
        if smiles.lower() == 'quit':
            break
        sequence = input("Protein Sequence: ").strip()
        
        try:
            affinity = predictor.predict(smiles, sequence)
            print(f"Predicted Affinity: {affinity:.4f}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()