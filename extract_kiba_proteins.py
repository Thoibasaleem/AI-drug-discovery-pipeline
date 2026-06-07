import pandas as pd
import requests
import json
import time
import os
from collections import defaultdict

# =====================================
# CONFIGURATION
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KIBA_FILE = os.path.join(BASE_DIR, 'kiba.txt')
OUTPUT_DIR = BASE_DIR

# =====================================
# STEP 1: Extract KIBA proteins
# =====================================
def extract_kiba_proteins(kiba_file=KIBA_FILE):
    """Extract unique proteins from KIBA dataset and create lookup mappings"""
    print(f"Loading KIBA data from {kiba_file}...")
    
    if not os.path.exists(kiba_file):
        raise FileNotFoundError(f"KIBA file not found: {kiba_file}")
    
    df = pd.read_csv(kiba_file, sep=' ', header=None,
                     names=['ID1', 'ID2', 'SMILES', 'Sequence', 'Score'])
    
    print(f"Total interactions: {len(df)}")
    
    # Get unique proteins with their sequences
    unique_proteins = df[['ID2', 'Sequence']].drop_duplicates().reset_index(drop=True)
    
    print(f"Unique proteins found: {len(unique_proteins)}")
    print("\nSample protein IDs:")
    print(unique_proteins.head(10))
    
    # Save protein list
    proteins_csv = os.path.join(OUTPUT_DIR, 'kiba_proteins.csv')
    unique_proteins.to_csv(proteins_csv, index=False)
    print(f"\n✓ Saved protein list to {proteins_csv}")
    
    # 🔥 NEW: Create and save sequence-to-ID lookup
    sequence_lookup = {
        'sequence_to_id': dict(zip(unique_proteins['Sequence'], unique_proteins['ID2'])),
        'id_to_sequence': dict(zip(unique_proteins['ID2'], unique_proteins['Sequence']))
    }
    
    lookup_file = os.path.join(OUTPUT_DIR, 'protein_sequence_lookup.json')
    with open(lookup_file, 'w') as f:
        json.dump(sequence_lookup, f, indent=2)
    print(f"✓ Saved sequence lookup to {lookup_file}")
    
    return unique_proteins


# =====================================
# STEP 2: UniProt → Gene → Ensembl mapping
# =====================================
def map_uniprot_to_ensembl(uniprot_ids, cache_file='uniprot_ensembl_mapping.json'):
    """Map UniProt IDs to Ensembl IDs via OpenTargets API"""
    cache_path = os.path.join(OUTPUT_DIR, cache_file)
    
    # Load from cache if exists
    if os.path.exists(cache_path):
        print(f"\nLoading cached mapping from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    print(f"\nMapping {len(uniprot_ids)} UniProt IDs...")
    opentargets_url = "https://api.platform.opentargets.org/api/v4/graphql"
    mapping = {}

    for i, uid in enumerate(uniprot_ids):
        print(f"[{i+1}/{len(uniprot_ids)}] {uid}", end=' ')
        
        try:
            # Step 2A: Get gene name from UniProt
            uni_url = f"https://rest.uniprot.org/uniprotkb/{uid}.json"
            r = requests.get(uni_url, timeout=10)

            if r.status_code != 200:
                print("✖ UniProt fetch failed")
                continue

            data = r.json()
            genes = data.get("genes", [])

            if not genes:
                print("✖ No gene info")
                continue

            gene_name = genes[0].get("geneName", {}).get("value")

            if not gene_name:
                print("✖ No gene symbol")
                continue

            # Step 2B: Search OpenTargets using gene symbol
            query = """
            query search($q: String!) {
              search(queryString: $q, page: {index: 0, size: 1}) {
                hits {
                  object {
                    ... on Target {
                      id
                      approvedSymbol
                    }
                  }
                }
              }
            }
            """

            res = requests.post(
                opentargets_url,
                json={"query": query, "variables": {"q": gene_name}},
                headers={'Content-Type': 'application/json'},
                timeout=15
            )

            if res.status_code != 200:
                print("✖ OpenTargets API error")
                continue

            hits = res.json().get("data", {}).get("search", {}).get("hits", [])

            if hits:
                ensembl_id = hits[0]["object"]["id"]
                mapping[uid] = {
                    "ensembl_id": ensembl_id,
                    "symbol": gene_name,
                    "sequence": None  # Will be filled later
                }
                print(f"✓ {uid} → {ensembl_id} ({gene_name})")
            else:
                print("✖ Not found in OpenTargets")

        except Exception as e:
            print(f"✖ Error: {str(e)[:50]}")

        time.sleep(0.2)  # Rate limiting

    print(f"\nTotal mapped: {len(mapping)}/{len(uniprot_ids)}")
    
    # Save mapping
    with open(cache_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"✓ Saved mapping to {cache_path}")
    
    return mapping


# =====================================
# STEP 3: Fetch disease associations
# =====================================
def fetch_disease_associations(mapping, cache_file='protein_disease_cache.json'):
    """Fetch disease associations from OpenTargets"""
    cache_path = os.path.join(OUTPUT_DIR, cache_file)
    
    # Return cached data if exists
    if os.path.exists(cache_path):
        print(f"\nLoading cached disease data from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    print("\nFetching disease associations...")
    url = "https://api.platform.opentargets.org/api/v4/graphql"
    
    protein_to_diseases = {}
    disease_to_proteins = defaultdict(list)

    for i, (uid, info) in enumerate(mapping.items()):
        ensembl_id = info['ensembl_id']
        print(f"[{i+1}/{len(mapping)}] {uid} ({ensembl_id})", end=' ')

        query = """
        query targetDiseases($id: String!) {
          target(ensemblId: $id) {
            id
            approvedSymbol
            associatedDiseases(page: {index: 0, size: 10}) {
              rows {
                disease {
                  name
                }
                score
              }
            }
          }
        }
        """

        try:
            res = requests.post(
                url,
                json={"query": query, "variables": {"id": ensembl_id}},
                headers={'Content-Type': 'application/json'},
                timeout=15
            )

            data = res.json()

            if "errors" in data:
                print(f"❌ GraphQL error: {data['errors']}")
                continue

            target = data.get("data", {}).get("target")

            if not target:
                print("⚠ No target found")
                continue

            diseases = target.get("associatedDiseases", {}).get("rows", [])

            if not diseases:
                print("⚠ No diseases")
                continue

            disease_list = []
            for d in diseases:
                name = d["disease"]["name"]
                score = d["score"]
                disease_list.append({"disease": name, "score": score})
                disease_to_proteins[name].append({
                    "protein": uid,
                    "score": score
                })

            protein_to_diseases[uid] = {
                "ensembl_id": ensembl_id,
                "symbol": info.get("symbol"),
                "diseases": disease_list
            }
            print(f"✓ {len(diseases)} diseases")

        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")

        time.sleep(0.2)

    result = {
        "protein_to_diseases": protein_to_diseases,
        "disease_to_proteins": dict(disease_to_proteins),
        "metadata": {
            "total_proteins": len(protein_to_diseases),
            "total_diseases": len(disease_to_proteins),
            "source": "OpenTargets Platform"
        }
    }

    with open(cache_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Saved disease mappings to {cache_path}")
    return result


# =====================================
# STEP 4: Create summary statistics
# =====================================
def create_disease_summary(data, output_file='disease_summary.csv'):
    """Create summary of diseases with protein counts"""
    output_path = os.path.join(OUTPUT_DIR, output_file)
    
    disease_to_proteins = data.get('disease_to_proteins', {})

    if not disease_to_proteins:
        print("❌ No disease data found. Skipping summary.")
        return pd.DataFrame()

    summary = []
    for disease, proteins in disease_to_proteins.items():
        if not proteins:
            continue
        
        avg_score = sum(p['score'] for p in proteins) / len(proteins)
        summary.append({
            "disease": disease,
            "num_targets": len(proteins),
            "avg_score": round(avg_score, 3),
            "top_protein": max(proteins, key=lambda x: x['score'])['protein']
        })

    if not summary:
        print("❌ Summary empty after processing.")
        return pd.DataFrame()

    df = pd.DataFrame(summary)
    df = df.sort_values("num_targets", ascending=False)

    print("\nTop 10 diseases by protein count:")
    print(df.head(10).to_string())

    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved summary to {output_path}")
    
    return df


# =====================================
# 🔥 NEW: Helper function for pipeline integration
# =====================================
def load_sequence_lookup(lookup_file='protein_sequence_lookup.json'):
    """Load sequence-to-ID mapping for pipeline integration"""
    lookup_path = os.path.join(OUTPUT_DIR, lookup_file)
    
    if not os.path.exists(lookup_path):
        raise FileNotFoundError(
            f"Sequence lookup not found. Run extract_kiba_proteins() first."
        )
    
    with open(lookup_path, 'r') as f:
        data = json.load(f)
    
    return data['sequence_to_id'], data['id_to_sequence']


def get_protein_id_from_sequence(sequence, lookup_file='protein_sequence_lookup.json'):
    """Get UniProt ID from protein sequence (for pipeline integration)"""
    seq_to_id, _ = load_sequence_lookup(lookup_file)
    
    # Exact match
    if sequence in seq_to_id:
        return seq_to_id[sequence]
    
    # Partial match (for truncated sequences)
    for seq, pid in seq_to_id.items():
        if sequence in seq or seq in sequence:
            print(f"⚠ Partial sequence match: {pid}")
            return pid
    
    return None


def get_sequence_from_protein_id(protein_id, lookup_file='protein_sequence_lookup.json'):
    """Get sequence from UniProt ID (for pipeline integration)"""
    _, id_to_seq = load_sequence_lookup(lookup_file)
    return id_to_seq.get(protein_id)


# =====================================
# MAIN PIPELINE
# =====================================
if __name__ == "__main__":
    print("=" * 60)
    print("KIBA Protein-Disease Data Extraction Pipeline")
    print("=" * 60)
    
    # Step 1: Extract proteins
    proteins_df = extract_kiba_proteins(KIBA_FILE)
    uniprot_ids = proteins_df['ID2'].tolist()
    
    print(f"\nFound {len(uniprot_ids)} unique proteins")
    
    # Step 2: Map to Ensembl
    mapping = map_uniprot_to_ensembl(uniprot_ids)
    
    if not mapping:
        print("\n❌ ERROR: Mapping failed!")
        exit(1)
    
    # Step 3: Fetch diseases
    disease_data = fetch_disease_associations(mapping)
    
    # Step 4: Create summary
    summary = create_disease_summary(disease_data)
    
    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print(f"   - Proteins mapped: {len(mapping)}")
    print(f"   - Diseases found: {len(disease_data.get('disease_to_proteins', {}))}")
    print("=" * 60)
    
    print("\n--- Testing Helper Functions ---")
    test_seq = proteins_df['Sequence'].iloc[0]
    test_id = proteins_df['ID2'].iloc[0]
    
    print(f"Test sequence lookup: {test_id} → {get_protein_id_from_sequence(test_seq)[:20]}...")
    print(f"Test ID lookup: {test_id} → {get_sequence_from_protein_id(test_id)[:20]}...")