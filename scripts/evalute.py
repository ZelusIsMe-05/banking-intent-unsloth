import pandas as pd
from tqdm import tqdm
import json
import yaml
from inference import IntentClassification

def evaluate_model(classifier, test_df):
    print(f"Running inference on {len(test_df)} samples...")
    
    predicted_list = []
    for text in tqdm(test_df["text"], desc="Inferencing"):
        label = classifier(text)
        predicted_list.append(label.strip().lower())
    
    true_labels = test_df["label_text"].str.strip().str.lower().tolist()
    
    correct = sum(1 for p, t in zip(predicted_list, true_labels) if p == t)
    return correct, len(true_labels)

if __name__ == "__main__":
    # 1. Load config
    with open("configs/inference.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    test_csv_path = config["paths"]["test_csv"]
    test_df = pd.read_csv(test_csv_path)

    print("\n" + "="*50)
    print("EVALUATING FINE-TUNED MODEL")
    print("="*50)
    
    ft_classifier = IntentClassification("configs/inference.yaml")
    ft_correct, total = evaluate_model(ft_classifier, test_df)

    ft_acc = ft_correct / total
    
    metrics = {
        "metrics": {
            "total_samples": total,
            "fine_tuned_correct": ft_correct,
            "fine_tuned_accuracy": ft_acc
        }
    }

    print("\n" + "="*25)
    print(" FINAL EVALUATION METRICS")
    print("="*25)
    print(json.dumps(metrics, indent=4))
    print("="*25 + "\n")

    with open("eval_results.json", "w") as f:
        json.dump(metrics, f, indent=4)