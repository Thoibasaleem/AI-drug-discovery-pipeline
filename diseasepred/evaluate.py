import json
import numpy as np

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
def load_data(pred_file, gt_file):
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    with open(gt_file, 'r') as f:
        gt_data = json.load(f)

    return pred_data["protein_to_diseases"], gt_data


# -----------------------------------------
# GET TOP-K PREDICTIONS
# -----------------------------------------
def get_top_k(predictions, k=10):
    top_k = {}

    for protein, data in predictions.items():
        diseases = data.get("diseases", [])

        # sort by score (descending)
        diseases_sorted = sorted(diseases, key=lambda x: x["score"], reverse=True)

        # take top k
        top_k[protein] = [d["disease"] for d in diseases_sorted[:k]]

    return top_k


# -----------------------------------------
# METRICS
# -----------------------------------------
def compute_metrics(pred_topk, ground_truth, k=10):
    precision_list = []
    recall_list = []
    f1_list = []
    hits = 0
    mrr_list = []

    for protein in ground_truth:
        true_set = set(ground_truth.get(protein, []))
        pred_list = pred_topk.get(protein, [])

        if not true_set:
            continue

        pred_set = set(pred_list)

        # -----------------
        # Precision & Recall
        # -----------------
        tp = len(pred_set & true_set)

        precision = tp / k if k > 0 else 0
        recall = tp / len(true_set) if len(true_set) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

        # -----------------
        # F1 Score
        # -----------------
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        f1_list.append(f1)

        # -----------------
        # Hits@K
        # -----------------
        if tp > 0:
            hits += 1

        # -----------------
        # MRR
        # -----------------
        rr = 0
        for rank, disease in enumerate(pred_list):
            if disease in true_set:
                rr = 1 / (rank + 1)
                break

        mrr_list.append(rr)

    # -----------------------------------------
    # FINAL SCORES
    # -----------------------------------------
    results = {
        "Precision@K": np.mean(precision_list),
        "Recall@K": np.mean(recall_list),
        "F1@K": np.mean(f1_list),
        "Hits@K": hits / len(ground_truth),
        "MRR": np.mean(mrr_list)
    }

    return results


# -----------------------------------------
# MAIN
# -----------------------------------------
if __name__ == "__main__":
    PRED_FILE = "protein_disease_cache.json"
    GT_FILE = "ground_truth.json"
    K = 10

    predictions, ground_truth = load_data(PRED_FILE, GT_FILE)

    pred_topk = get_top_k(predictions, K)

    results = compute_metrics(pred_topk, ground_truth, K)

    print("\n📊 Evaluation Results:\n")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")