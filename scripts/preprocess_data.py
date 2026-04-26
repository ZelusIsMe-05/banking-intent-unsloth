from datasets import load_dataset
import pandas as pd
import os

def main():
    print("1. Downloading banking77 dataset from Hugging Face...")
    dataset = load_dataset("banking77")

    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    # Get the list of label names
    label_names = dataset["train"].features["label"].names

    print("2. Sampling data...")
    # Shuffle the data to prevent the model from learning the label order
    sampled_train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    sampled_test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Add label_text column
    sampled_train_df["label_text"] = sampled_train_df["label"].apply(lambda x: label_names[x])
    sampled_test_df["label_text"] = sampled_test_df["label"].apply(lambda x: label_names[x])

    label_df = pd.DataFrame({
        "id": list(range(len(label_names))),
        "label_text": label_names
    })

    print(f" - Number of Train samples: {len(sampled_train_df)}")
    print(f" - Number of Test samples: {len(sampled_test_df)}")

    print("3. Saving files...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    output_dir = os.path.join(project_dir, "sample_data")

    os.makedirs(output_dir, exist_ok=True)

    sampled_train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False, encoding="utf-8")
    sampled_test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False, encoding="utf-8")
    label_df.to_csv(os.path.join(output_dir, "labels.csv"), index=False, encoding="utf-8")

    print(f"Done! Data have been saved successfully")

if __name__ == "__main__":
    main()