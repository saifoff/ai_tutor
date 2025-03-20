import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from dotenv import load_dotenv
import wandb
from tqdm import tqdm

# Load environment variables
load_dotenv()

class EducationalTutorTrainer:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Add special tokens for tutoring
        special_tokens = {
            "additional_special_tokens": [
                "<student>", "</student>",
                "<tutor>", "</tutor>",
                "<question>", "</question>",
                "<hint>", "</hint>",
                "<solution>", "</solution>"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def prepare_dataset(self, data_path):
        """Prepare the dataset for training"""
        dataset = load_dataset("json", data_files=data_path)
        
        def format_training_example(example):
            """Format each example for training"""
            formatted_text = f"<tutor>Let's solve this step by step.</tutor>\n"
            formatted_text += f"<question>{example['question']}</question>\n"
            formatted_text += f"<hint>{example['hint']}</hint>\n"
            formatted_text += f"<solution>{example['solution']}</solution>\n"
            return {"text": formatted_text}
        
        return dataset.map(
            format_training_example,
            remove_columns=dataset["train"].column_names
        )
    
    def train(self, dataset, output_dir, num_train_epochs=3):
        """Train the model"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=100,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            report_to="wandb",
            fp16=True,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            ),
        )
        
        # Initialize wandb
        wandb.init(project="educational-tutor")
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Close wandb
        wandb.finish()

def main():
    # Initialize trainer
    trainer = EducationalTutorTrainer()
    
    # Prepare dataset
    dataset = trainer.prepare_dataset("data/training/training_data.json")
    
    # Train the model
    trainer.train(
        dataset=dataset,
        output_dir=os.getenv("MODEL_PATH", "models/educational_tutor")
    )

if __name__ == "__main__":
    main() 