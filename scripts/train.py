from unsloth import FastLanguageModel
import yaml
import torch
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

def load_config(config_path="configs/train.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    print(f"1. Initializing {config['model']['name']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config["model"]["name"],
        max_seq_length = config["model"]["max_seq_length"],
        load_in_4bit = True,
    )

    print("2. Loading datasets...")
    train_df = pd.read_csv(config["paths"]["train_data"])
    test_df = pd.read_csv(config["paths"]["test_data"])
    
    # Prompt template for Instruction Tuning
    prompt_template = """Below is an inquiry from a bank customer. Classify the intent of this message.

### Instruction:
Classify the following message into one of the banking categories.

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token
    
    # Apply template to generate the training text
    train_df["text"] = train_df.apply(lambda row: prompt_template.format(row["text"], row["label_text"]) + EOS_TOKEN, axis=1)
    test_df["text"] = test_df.apply(lambda row: prompt_template.format(row["text"], row["label_text"]) + EOS_TOKEN, axis=1)
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    print("3. Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = config["lora"]["r"],
        target_modules = config["lora"]["target_modules"],
        lora_alpha = config["lora"]["alpha"],
        lora_dropout = config["lora"]["dropout"],
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = config["training"].get("seed", 42), # Đồng bộ seed với lora
    )

    print("4. Setting up SFT Trainer...")
    training_args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = config["model"]["max_seq_length"],
        output_dir = config["paths"]["output_dir"],
        learning_rate = float(config["training"]["learning_rate"]),
        per_device_train_batch_size = config["training"]["batch_size"],
        per_device_eval_batch_size = config["training"]["batch_size"],
        num_train_epochs = config["training"]["epochs"],
        weight_decay = config["training"]["weight_decay"],
        optim = config["training"]["optimizer"],
        warmup_steps = config["training"]["warmup_steps"],
        lr_scheduler_type = config["training"].get("lr_scheduler_type", "linear"),
        seed = config["training"].get("seed", 42),
        eval_strategy = "epoch",
        save_strategy = "epoch",
        logging_steps = 10,
        load_best_model_at_end = True,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        args = training_args,
    )

    print("5. Starting training...")
    trainer.train()

    print("6. Saving final model...")
    model.save_pretrained(config["paths"]["output_dir"])
    tokenizer.save_pretrained(config["paths"]["output_dir"])
    print("Training complete and model saved!")

if __name__ == "__main__":
    main()