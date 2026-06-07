import streamlit as st
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "druglikenessmodel"))
sys.path.append(os.path.join(BASE_DIR, "binding-affinity"))
sys.path.append(os.path.join(BASE_DIR, "diseasepred"))

# =========================
# IMPORTS
# =========================
from smiles_predict import predict_from_smiles
from predict import KibaPredictor
from protein_lookup import get_protein_id
from validate import get_diseases

from rdkit import Chem
from rdkit.Chem import Draw

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Drug Discovery", layout="wide")

# =========================
# STYLE
# =========================
st.markdown("""
<style>
body {
    background-color: #f4f8fb;
}
h1, h2, h3 {
    color: #2c7be5;
}
.stButton>button {
    background-color: #2c7be5;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🧠 AI Drug Discovery")

mode = st.sidebar.radio(
    "Select Mode",
    ["Full Pipeline", "Drug-Likeness", "Binding Affinity", "Disease Prediction"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("Run individual modules or full pipeline")

# =========================
# TITLE
# =========================
st.title("💊 AI Drug Discovery Pipeline")
st.caption("ML + DL + Knowledge-Based System")

# =========================
# INPUT SECTION
# =========================
col1, col2 = st.columns(2)

with col1:
    smiles = st.text_area("SMILES Input")

with col2:
    protein_seq = st.text_area("Protein Sequence")

run = st.button("🚀 Run Pipeline")

# =========================
# EXECUTION
# =========================
if run:

    if not smiles:
        st.error("Enter SMILES")
        st.stop()

    st.markdown("---")

    # =========================
    # MOLECULE VISUALIZATION
    # =========================
    st.subheader("🧪 Molecule Structure")

    try:
        mol = Chem.MolFromSmiles(smiles)
        st.image(Draw.MolToImage(mol), width=300)
    except:
        st.warning("Invalid SMILES")

    # =========================
    # RESULTS (SIDE-BY-SIDE)
    # =========================
    st.markdown("---")
    st.markdown("## 📊 Results")

    col1, col2, col3 = st.columns(3)

    # -------------------------------
    # DRUG LIKENESS
    with col1:
        st.markdown("### [1] Drug-Likeness")

        if mode in ["Full Pipeline", "Drug-Likeness"]:
            label, prob = predict_from_smiles(smiles)
            status = "Drug-like" if label == 1 else "Not Drug-like"

            st.success(status)
            st.write(f"Confidence: {prob:.4f}")
        else:
            st.info("Skipped")

    # -------------------------------
    # BINDING
    with col2:
        st.markdown("### [2] Binding Affinity")

        if mode in ["Full Pipeline", "Binding Affinity"]:
            predictor = KibaPredictor()
            kiba_score, _ = predictor.predict(smiles, protein_seq)

            st.write(f"KIBA Score: {kiba_score:.4f}")

            if kiba_score >= 12.1:
                st.success("HIGH")
            elif kiba_score >= 10:
                st.warning("MODERATE")
            else:
                st.error("LOW")
        else:
            st.info("Skipped")

    # -------------------------------
    # DISEASE
    with col3:
        st.markdown("### [3] Disease Association")

        if mode in ["Full Pipeline", "Disease Prediction"]:
            protein_id = get_protein_id(protein_seq)

            if protein_id:
                st.write(f"Protein ID: {protein_id}")

                diseases = get_diseases(protein_id)

                if diseases:
                    for d in diseases[:5]:
                        st.write(f"{d['disease']} ({d['score']:.2f})")
                else:
                    st.warning("No data")
            else:
                st.error("No ID found")
        else:
            st.info("Skipped")

    # =========================
    # DISEASE GRAPH
    # =========================
    if mode in ["Full Pipeline", "Disease Prediction"]:
        if protein_seq:
            protein_id = get_protein_id(protein_seq)
            diseases = get_diseases(protein_id)

            if diseases:
                st.markdown("---")
                st.subheader("📊 Disease Scores")

                names = [d["disease"] for d in diseases[:10]]
                scores = [d["score"] for d in diseases[:10]]

                fig, ax = plt.subplots()
                ax.barh(names, scores)
                ax.invert_yaxis()

                st.pyplot(fig)

    st.success("✅ Pipeline Completed")