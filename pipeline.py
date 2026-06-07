import sys
import os

# =====================================
# PATH SETUP
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "druglikenessmodel"))
sys.path.append(os.path.join(BASE_DIR, "binding-affinity"))
sys.path.append(os.path.join(BASE_DIR, "diseasepred"))

# =====================================
# IMPORTS
# =====================================
from smiles_predict import predict_from_smiles
from predict import KibaPredictor

# 🔥 NEW: Import protein lookup
try:
    from protein_lookup import get_protein_id
    LOOKUP_AVAILABLE = True
except ImportError:
    LOOKUP_AVAILABLE = False
    print("⚠ Warning: protein_lookup not available, auto-detection disabled")

import json

# =====================================
# LOAD DISEASE DATA
# =====================================
disease_file = os.path.join(BASE_DIR, "diseasepred", "protein_disease_cache.json")

with open(disease_file, "r") as f:
    disease_data = json.load(f)["protein_to_diseases"]


def get_diseases(protein_id):
    """Get sorted diseases for a protein"""
    if protein_id not in disease_data:
        return None
    diseases = disease_data[protein_id]["diseases"]
    return sorted(diseases, key=lambda x: x["score"], reverse=True)


# =====================================
# MAIN PIPELINE
# =====================================
def run_pipeline(smiles, protein_seq, protein_id=None, auto_detect=True):
    """
    Run the full drug discovery pipeline.
    
    Args:
        smiles: SMILES string of the molecule
        protein_seq: Amino acid sequence of target protein
        protein_id: Optional UniProt ID (auto-detected if not provided)
        auto_detect: Whether to auto-detect protein ID from sequence
    """
    print("\n" + "=" * 50)
    print("DRUG DISCOVERY PIPELINE")
    print("=" * 50)

    # 🔥 AUTO-DETECT protein ID
    if protein_id is None and auto_detect and LOOKUP_AVAILABLE:
        print("\n[0] Auto-detecting Protein ID from sequence...")
        protein_id = get_protein_id(protein_seq)
        
        if protein_id:
            print(f"   ✓ Detected: {protein_id}")
        else:
            print("   ✗ Could not auto-detect ID")
    
    elif protein_id:
        print(f"\n[0] Using provided Protein ID: {protein_id}")
    else:
        print("\n[0] No Protein ID available (auto-detection disabled)")

    # ------------------------------- STEP 1: Drug-Likeness
    print("\n[1] Drug-Likeness Prediction")
    print("-" * 30)
    
    try:
        label, prob = predict_from_smiles(smiles)
        status = "Drug-like" if label == 1 else "Not Drug-like"
        print(f"   Result: {status}")
        print(f"   Confidence: {prob:.4f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return

    if label == 0:
        print("\n❌ PIPELINE STOPPED: Molecule is not drug-like")
        return

    # ------------------------------- STEP 2: Binding Affinity
    print("\n[2] Binding Affinity Prediction")
    print("-" * 30)
    
    try:
        predictor = KibaPredictor()
        kiba_score, confidence = predictor.predict(smiles, protein_seq)
        
        print(f"   KIBA Score: {kiba_score:.4f}")
        
        if kiba_score >= 12.1:
            binding = "HIGH"
        elif kiba_score >= 10:
            binding = "MODERATE"
        else:
            binding = "LOW"
        print(f"   Binding: {binding}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return

    # ------------------------------- STEP 3: Disease Prediction
    print("\n[3] Disease Association")
    print("-" * 30)
    
    if protein_id is None:
        print("   ⚠ Skipped: No Protein ID available")
        print("\n✅ Pipeline completed (Steps 1-2 only)")
        return

    diseases = get_diseases(protein_id)
    
    if diseases is None:
        print(f"   ⚠ No disease data for {protein_id}")
        print("\n✅ Pipeline completed (binding affinity only)")
        return

    print(f"\n   Top Diseases for {protein_id}:")
    for i, d in enumerate(diseases[:10], 1):
        print(f"   {i:2d}. {d['disease']:<40} (score: {d['score']:.3f})")

    print("\n" + "=" * 50)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 50)


# CLI ENTRY POINT
# =====================================
if __name__ == "__main__":
    print("=" * 50)
    print("Drug Discovery Pipeline")
    print("=" * 50)

    smiles = input("\nEnter SMILES: ").strip()
    protein_seq = input("Enter Protein Sequence: ").strip()

    run_pipeline(smiles, protein_seq)

