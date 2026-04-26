echo "Starting data preprocessing..."
python scripts/preprocess_data.py

echo "Starting model fine-tuning with Unsloth..."
python scripts/train.py