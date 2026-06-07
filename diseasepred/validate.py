import json
import os


# =========================
# SINGLETON LOOKUP CLASS
# =========================
class DiseaseLookup:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DiseaseLookup, cls).__new__(cls)
            cls._instance._load_data()
        return cls._instance

    # =========================
    # LOAD JSON DATA
    # =========================
    def _load_data(self):

        current_dir = os.path.dirname(__file__)

        file_path = os.path.join(
            current_dir,
            "protein_disease_cache.json"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Protein-disease cache not found: {file_path}"
            )

        with open(file_path, "r") as f:
            data = json.load(f)

        # Handle both formats
        if "protein_to_diseases" in data:
            raw_data = data["protein_to_diseases"]
        else:
            raw_data = data

        # NORMALIZE KEYS
        self.protein_to_diseases = {
            str(k).strip().upper(): v
            for k, v in raw_data.items()
        }

        print(
            f"✓ Loaded {len(self.protein_to_diseases)} protein-disease mappings"
        )

    # =========================
    # CORE FUNCTION
    # =========================
    def get_diseases(self, protein_id, top_k=10):

        if protein_id is None:
            return []

        # NORMALIZE INPUT
        protein_id = str(protein_id).strip().upper()

        print(f"\n[DEBUG] Looking for protein: {protein_id}")

        # DEBUG SAMPLE KEYS
        sample_keys = list(self.protein_to_diseases.keys())[:5]
        print(f"[DEBUG] Sample keys: {sample_keys}")

        # CHECK EXISTENCE
        if protein_id not in self.protein_to_diseases:
            print(f"[DEBUG] Protein ID not found")
            return []

        diseases = self.protein_to_diseases[protein_id]

        print(f"[DEBUG] Raw disease data type: {type(diseases)}")

        # =========================
        # CASE 1:
        # {"diseases": [...]}
        # =========================
        if isinstance(diseases, dict):

            if "diseases" in diseases:
                diseases = diseases["diseases"]
            else:
                return []

        # =========================
        # CASE 2:
        # List format
        # =========================
        if isinstance(diseases, list):

            if len(diseases) == 0:
                return []

            # -------------------------
            # LIST OF DICTS
            # -------------------------
            if isinstance(diseases[0], dict):

                # Add default score if missing
                cleaned = []

                for d in diseases:

                    disease_name = d.get(
                        "disease",
                        "unknown"
                    )

                    score = float(
                        d.get("score", 0.0)
                    )

                    cleaned.append({
                        "disease": disease_name,
                        "score": score
                    })

                cleaned = sorted(
                    cleaned,
                    key=lambda x: x["score"],
                    reverse=True
                )

                return cleaned[:top_k]

            # -------------------------
            # LIST OF STRINGS
            # -------------------------
            elif isinstance(diseases[0], str):

                return [
                    {
                        "disease": d,
                        "score": 0.0
                    }
                    for d in diseases[:top_k]
                ]

        # =========================
        # INVALID FORMAT
        # =========================
        print("[DEBUG] Unsupported disease format")

        return []


# =========================
# PUBLIC FUNCTION
# =========================
def get_diseases(protein_id, top_k=10):

    lookup = DiseaseLookup()

    return lookup.get_diseases(
        protein_id,
        top_k
    )


# =========================
# QUICK TEST
# =========================
if __name__ == "__main__":

    test_protein = "O00141"

    print("\n=== Disease Validation Test ===\n")

    results = get_diseases(test_protein)

    if not results:
        print("❌ No diseases found.")

    else:
        for i, d in enumerate(results, 1):

            print(
                f"{i}. {d['disease']} "
                f"(score: {d['score']:.3f})"
            )