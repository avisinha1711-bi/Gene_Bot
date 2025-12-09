# train_model.py
"""
Train Mistral-7B on genetics data for Gene_Bot.
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import model_config, chatbot_config
from knowledge_base import GeneticsKnowledgeBase

class MistralGeneticsTrainer:
    def __init__(self):
        """Initialize Mistral-7B trainer"""
        self.config = model_config
        self.tokenizer = None
        self.model = None
        self.knowledge_base = GeneticsKnowledgeBase()
        
    def prepare_training_data(self, data_path=None):
        """
        Prepare genetics-specific training data.
        
        Args:
            data_path: Path to training data JSON/CSV
        """
        print("Preparing training data...")
        
        if data_path and os.path.exists(data_path):
            if data_path.endswith('.json'):
                with open(data_path, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            elif data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            else:
                raise ValueError("Unsupported file format")
        else:
            # Use knowledge base data
            df = self._prepare_from_knowledge_base()
        
        # Convert to instruction format
        training_examples = []
        for _, row in df.iterrows():
            if 'question' in df.columns and 'answer' in df.columns:
                prompt = self._create_prompt(row['question'])
                training_examples.append({
                    "text": f"{prompt}{row['answer']}</s>"
                })
            elif 'text' in df.columns:
                training_examples.append({
                    "text": row['text'] + "</s>"
                })
        
        print(f"Prepared {len(training_examples)} training examples")
        return Dataset.from_pandas(pd.DataFrame(training_examples))
    
    def _prepare_from_knowledge_base(self):
        """Prepare training data from knowledge base"""
        try:
            kb_data = []
            kb_dir = chatbot_config.knowledge_base_path
            
            for filename in os.listdir(kb_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(kb_dir, filename), 'r') as f:
                        data = json.load(f)
                        
                    if isinstance(data, dict):
                        # Q&A format
                        if 'questions' in data and 'answers' in data:
                            for q, a in zip(data['questions'], data['answers']):
                                kb_data.append({'question': q, 'answer': a})
                        # Document format
                        elif 'content' in data:
                            kb_data.append({'text': data['content']})
                    elif isinstance(data, list):
                        for item in data:
                            kb_data.append({'text': str(item)})
            
            return pd.DataFrame(kb_data)
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample genetics data"""
        sample_data = {
            'question': [
                "What is DNA?",
                "Explain CRISPR gene editing",
                "What causes Down syndrome?",
                "How does mRNA vaccine work?",
                "What is genetic inheritance?"
            ],
            'answer': [
                "DNA (deoxyribonucleic acid) is the hereditary material in humans and almost all other organisms. It contains the biological instructions that make each species unique.",
                "CRISPR (Clustered Regularly Interspaced Short Palindromic Repeats) is a gene-editing technology that allows scientists to precisely modify DNA sequences in living organisms. It works by using a guide RNA to target specific DNA sequences and the Cas9 enzyme to cut the DNA at that location.",
                "Down syndrome is caused by the presence of an extra copy of chromosome 21 (trisomy 21). This additional genetic material alters the course of development and causes the characteristic features of Down syndrome.",
                "mRNA vaccines work by introducing a piece of mRNA that corresponds to a viral protein, usually a small piece of a protein found on the virus's outer membrane. Cells use this mRNA as a template to build the foreign protein, which then triggers an immune response.",
                "Genetic inheritance is the process by which genetic information is passed from parents to offspring. It follows Mendelian principles where traits are determined by pairs of genes, with one gene inherited from each parent."
            ]
        }
        return pd.DataFrame(sample_data)
    
    def _create_prompt(self, question):
        """Create instruction prompt for Mistral"""
        return f"""<s>[INST] You are GeneBot, a specialized genetics expert chatbot. 
Provide accurate, detailed, and clear explanations about genetics and molecular biology.

Question: {question}

Answer: [/INST]"""
    
    def tokenize_function(self, examples):
        """Tokenize training examples"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
    
    def load_model_and_tokenizer(self):
        """Load Mistral-7B model and tokenizer"""
        print("Loading Mistral-7B model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Add special tokens if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            load_in_4bit=self.config.load_in_4bit,
            torch_dtype=torch.float16 if self.config.load_in_4bit else torch.float32,
            device_map=self.config.device_map,
            quantization_config=self._get_quantization_config() if self.config.load_in_4bit else None,
            trust_remote_code=True
        )
        
        # Apply LoRA for efficient fine-tuning
        self._apply_lora()
        
        print("Model loaded successfully!")
    
    def _get_quantization_config(self):
        """Get quantization configuration"""
        from transformers import BitsAndBytesConfig
        
        return BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True
        )
    
    def _apply_lora(self):
        """Apply LoRA for parameter-efficient fine-tuning"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def train(self, data_path=None, output_dir=None):
        """
        Fine-tune Mistral-7B on genetics data.
        
        Args:
            data_path: Path to training data
            output_dir: Directory to save fine-tuned model
        """
        print("=" * 60)
        print("MISTRAL-7B GENETICS FINE-TUNING")
        print("=" * 60)
        
        # Load model
        self.load_model_and_tokenizer()
        
        # Prepare data
        dataset = self.prepare_training_data(data_path)
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        
        # Split dataset
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir or "./models/mistral-7b-finetuned",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=2e-4,
            fp16=True,
            push_to_hub=False,
            report_to="none"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train
        print("\nStarting training...")
        trainer.train()
        
        # Save model
        self.save_model(output_dir or "./models/mistral-7b-finetuned")
        
        return trainer
    
    def save_model(self, output_dir):
        """Save the fine-tuned model"""
        print(f"\nSaving model to {output_dir}...")
        
        # Create directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save configuration
        config_data = {
            "model_name": self.config.model_name,
            "fine_tuned_on": "genetics_data",
            "training_date": datetime.now().isoformat(),
            "parameters": {
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "max_length": 512
            }
        }
        
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Model saved to {output_dir}")
    
    def generate_response(self, prompt, **kwargs):
        """Generate response using fine-tuned model"""
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # Format prompt
        formatted_prompt = self._create_prompt(prompt)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_new_tokens', self.config.max_new_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                top_p=kwargs.get('top_p', self.config.top_p),
                repetition_penalty=kwargs.get('repetition_penalty', self.config.repetition_penalty),
                do_sample=True
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer part
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        return response


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune Mistral-7B on genetics data')
    parser.add_argument('--data', type=str, help='Path to training data (JSON/CSV)')
    parser.add_argument('--output', type=str, default='./models/mistral-7b-genetics', 
                       help='Output directory for fine-tuned model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', type=str, help='Test the model with a question')
    
    args = parser.parse_args()
    
    trainer = MistralGeneticsTrainer()
    
    if args.train:
        # Train model
        trainer.train(data_path=args.data, output_dir=args.output)
    
    elif args.test:
        # Test model
        response = trainer.generate_response(args.test)
        print(f"\nQuestion: {args.test}")
        print(f"\nAnswer: {response}")
    
    else:
        print("Use --train to fine-tune or --test to test the model")


if __name__ == "__main__":
    main()
