from unsloth import FastLanguageModel
import yaml

class IntentClassification:
    def __init__(self, model_path):
        with open(model_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        checkpoint_path = self.config["model"]["checkpoint_path"]
        max_seq_length = self.config["model"]["max_seq_length"]

        # Load trained model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = checkpoint_path,
            max_seq_length = max_seq_length,
            load_in_4bit = True,
        )
        
        # Optimize for 2x faster inference
        FastLanguageModel.for_inference(self.model)
        
        self.prompt_template = """Below is an inquiry from a bank customer. Classify the intent of this message.

### Instruction:
Classify the following message into one of the banking categories. Output ONLY the exact label name without any punctuation or additional text.

### Input:
{}

### Response:
"""
    def __call__(self, message):
        # Format the input using the same template used in training
        prompt = self.prompt_template.format(message)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        # Generate prediction
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=32, 
            max_length=None,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode and extract only the generated label
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        predicted_label = output_text.split("### Response:\n")[-1].strip()
        
        predicted_label = predicted_label.rstrip("?.!, ")
        
        return predicted_label


if __name__ == "__main__":
    print("Loading inference model... Please wait.")
    classifier = IntentClassification("configs/inference.yaml")
    
    print("\n" + "="*50)
    print("SHORT USAGE EXAMPLE")
    print("="*50 + "\n")

    msg = "I was at the mall yesterday and someone stole my wallet with all my cards inside!"
    
    label = classifier(msg)
    print(f"Message: '{msg}'\nPredicted Intent: {label}\n")

    print("\n" + "="*50)
    print("INTERACTIVE MODE (Type 'exit' to quit)")
    print("="*50)
    
    while True:
        try:
            user_input = input("\nEnter a message: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Exiting interactive mode...")
                break
                
            if not user_input.strip():
                continue
                
            label = classifier(user_input)
            print(f"Predicted Intent: {label}")
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break