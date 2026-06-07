from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

from predict_example import predict_from_descriptors


def smiles_to_basic_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rtb = Descriptors.NumRotatableBonds(mol)

    return mw, logp, tpsa, hbd, hba, rtb


def predict_from_smiles(smiles: str):
    mw, logp, tpsa, hbd, hba, rtb = smiles_to_basic_descriptors(smiles)

    print("\nComputed descriptors:")
    print(f"  MolecularWeight : {mw}")
    print(f"  LogP            : {logp}")
    print(f"  TPSA            : {tpsa}")
    print(f"  HBD             : {hbd}")
    print(f"  HBA             : {hba}")
    print(f"  RotatableBonds  : {rtb}\n")

    return predict_from_descriptors(mw, logp, tpsa, hbd, hba, rtb)


if __name__ == "__main__":
    test_smiles = input("Enter SMILES: ").strip()
    label, prob = predict_from_smiles(test_smiles)
    print("Predicted Suitability:", label)
    print("Probability (drug-like):", prob)
