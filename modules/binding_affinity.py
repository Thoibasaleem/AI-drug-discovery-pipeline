import sys
sys.path.append("binding-affinity")

from predict import KibaPredictor

predictor = KibaPredictor()

def predict_binding(smiles, protein):

    kiba_score, raw = predictor.predict(smiles, protein)

    return kiba_score