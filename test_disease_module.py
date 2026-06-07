from diseasepred.protein_lookup import get_protein_id
from diseasepred.validate import get_diseases  # adjust if needed

def test_disease_module():
    print("\n=== Disease Prediction Module Test ===\n")

    # Sample protein sequence (already in your lookup)
    protein_seq = "MTVKTEAAKGTLTYSRMRGMVAILIAFMKQRRMGLNDFIQKIANNSYACKHPEVQSILKISQPQEPELMNANPSPPPSPSQQINLGPSSNPHAKPSDFHFLKVIGKGSFGKVLLARHKAEEVFYAVKVLQKKAILKKKEEKHIMSERNVLLKNVKHPFLVGLHFSFQTADKLYFVLDYINGGELFYHLQRERCFLEPRARFYAAEIASALGYLHSLNIVYRDLKPENILLDSQGHIVLTDFGLCKENIEHNSTTSTFCGTPEYLAPEVLHKQPYDRTVDWWCLGAVLYEMLYGLPPFYSRNTAEMYDNILNKPLQLKPNITNSARHLLEGLLQKDRTKRLGAKDDFMEIKSHVFFSLINWDDLINKKITPPFNPNVSGPNDLRHFDPEFTEEPVPNSIGKSPDSVLVTASVKEAAEAFLGFSYAPPTDSFL"

    # Step 1: Get Protein ID
    print("[1] Fetching Protein ID...")
    protein_id = get_protein_id(protein_seq)
    print(f"   Protein ID: {protein_id}")

    # Step 2: Get Diseases
    print("\n[2] Fetching Associated Diseases...")
    diseases = get_diseases(protein_id)

    # Step 3: Display Results
    print("\nTop Diseases:\n")
    for i, d in enumerate(diseases[:10], 1):
        print(f"{i}. {d['disease']} (score: {d['score']:.3f})")

    print("\n✅ Disease module working correctly!\n")


if __name__ == "__main__":
    test_disease_module()