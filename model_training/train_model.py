"""
Train Mistral-7B on genetics data for GeneBot.
"""

import os
import json
import torch
import warnings
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, load_dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import model_config, training_config
from knowledge_base import GeneticsKnowledgeBase

warnings.filterwarnings('ignore')

class GeneticsDataset:
    """Dataset preparation for genetics training"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def prepare_dataset(self, data_path: Optional[str] = None) -> Dataset:
        """Prepare dataset from various sources"""
        if data_path and os.path.exists(data_path):
            return self._load_from_file(data_path)
        else:
            return self._create_default_dataset()
    
    def _load_from_file(self, data_path: str) -> Dataset:
        """Load dataset from file"""
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path, encoding='utf-8')
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        return self._process_dataframe(df)
    
    def _process_dataframe(self, df: pd.DataFrame) -> Dataset:
        """Process dataframe into training format"""
        examples = []
        
        # Check for different column formats
        if 'instruction' in df.columns and 'output' in df.columns:
            # Instruction-output format
            for _, row in df.iterrows():
                text = self._format_instruction_example(
                    row['instruction'], 
                    row['output']
                )
                examples.append({'text': text})
        
        elif 'question' in df.columns and 'answer' in df.columns:
            # Q&A format
            for _, row in df.iterrows():
                text = self._format_qa_example(
                    row['question'], 
                    row['answer']
                )
                examples.append({'text': text})
        
        elif 'text' in df.columns:
            # Plain text format
            for _, row in df.iterrows():
                text = row['text']
                if len(text) > 100:  # Filter out very short texts
                    examples.append({'text': text + self.tokenizer.eos_token})
        
        elif 'prompt' in df.columns and 'completion' in df.columns:
            # OpenAI format
            for _, row in df.iterrows():
                text = f"{row['prompt']}{row['completion']}{self.tokenizer.eos_token}"
                examples.append({'text': text})
        
        else:
            raise ValueError("Could not identify dataset format")
        
        print(f"Prepared {len(examples)} training examples")
        return Dataset.from_pandas(pd.DataFrame(examples))
    
    def _format_instruction_example(self, instruction: str, output: str) -> str:
        """Format instruction example for Mistral"""
        return f"""<s>[INST] {instruction} [/INST] {output}</s>"""
    
    def _format_qa_example(self, question: str, answer: str) -> str:
        """Format Q&A example for Mistral"""
        instruction = f"You are a genetics expert. Answer the following question accurately: {question}"
        return f"""<s>[INST] {instruction} [/INST] {answer}</s>"""
    
    def _create_default_dataset(self) -> Dataset:
        """Create default genetics dataset"""
        default_data = [
            {
                "instruction": "What is DNA and how does it store genetic information?",
                "output": "DNA (deoxyribonucleic acid) is a molecule that carries the genetic instructions used in growth, development, functioning and reproduction of all known organisms. It's composed of two strands forming a double helix, made of nucleotides containing four bases: adenine (A), thymine (T), cytosine (C), and guanine (G). The sequence of these bases determines genetic information through the genetic code."
            },
            {
                "instruction": "Explain CRISPR-Cas9 gene editing technology.",
                "output": "CRISPR-Cas9 is a revolutionary genome editing technology derived from bacterial immune systems. It consists of two components: 1) Cas9 enzyme - molecular scissors that cut DNA, and 2) guide RNA (gRNA) - directs Cas9 to specific DNA sequences. The system allows precise DNA modifications including gene knockout, knock-in, and base editing, enabling applications in genetic research, agriculture, and gene therapy."
            },
            {
                "instruction": "What causes sickle cell anemia at the molecular level?",
                "output": "Sickle cell anemia is caused by a single nucleotide mutation in the HBB gene (hemoglobin beta chain), where adenine is replaced by thymine (GAG → GTG). This results in substitution of glutamic acid with valine at position 6 of the beta-globin chain. The mutant hemoglobin (HbS) polymerizes under low oxygen conditions, causing red blood cells to adopt a sickle shape, leading to vaso-occlusive crises and hemolytic anemia."
            },
            {
                "instruction": "How does PCR (Polymerase Chain Reaction) work?",
                "output": "PCR amplifies specific DNA sequences through repeated temperature cycles: 1) Denaturation (94-98°C) - DNA strands separate, 2) Annealing (50-65°C) - primers bind to target sequences, 3) Extension (72°C) - DNA polymerase synthesizes new strands. This exponential amplification (2^n copies after n cycles) enables detection and analysis of minute DNA quantities in research, diagnostics, and forensics."
            },
            {
                "instruction": "What are the differences between mitosis and meiosis?",
                "output": "Mitosis produces two genetically identical diploid daughter cells for growth and repair, involving one division. Meiosis produces four genetically unique haploid gametes for sexual reproduction, involving two divisions with crossing over and independent assortment. Mitosis maintains chromosome number, while meiosis halves it. Genetic variation occurs only in meiosis through recombination and random assortment."
            }
        ]
        
        texts = []
        for item in default_data:
            text = self._format_instruction_example(item["instruction"], item["output"])
            texts.append({"text": text})
        
        return Dataset.from_pandas(pd.DataFrame(texts))

class MistralGeneticsTrainer:
    def __init__(self):
        """Initialize Mistral-7B trainer"""
        self.config = model_config
        self.train_config = training_config
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
    def prepare_training(self, data_path: Optional[str] = None):
        """Prepare all components for training"""
        print("=" * 60)
        print("MISTRAL-7B GENETICS FINE-TUNING PREPARATION")
        print("=" * 60)
        
        # Load tokenizer
        print("\n1. Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare dataset
        print("\n2. Preparing dataset...")
        dataset_prep = GeneticsDataset(self.tokenizer)
        self.dataset = dataset_prep.prepare_dataset(data_path)
        
        # Tokenize dataset
        print("\n3. Tokenizing dataset...")
        tokenized_dataset = self.dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=self.dataset.column_names
        )
        
        # Split dataset
        split_dataset = tokenized_dataset.train_test_split(
            test_size=0.1,
            shuffle=True,
            seed=42
        )
        
        print(f"\nDataset statistics:")
        print(f"  Training samples: {len(split_dataset['train'])}")
        print(f"  Validation samples: {len(split_dataset['test'])}")
        
        return split_dataset
    
    def _tokenize_function(self, examples):
        """Tokenize examples"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None
        )
    
    def load_model(self):
        """Load model with quantization"""
        print("\n4. Loading model...")
        
        bnb_config = None
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=True
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if self.config.load_in_4bit else torch.float32,
            device_map=self.config.device_map,
            trust_remote_code=True,
            use_safetensors=True
        )
        
        # Apply LoRA
        print("\n5. Applying LoRA...")
        self._apply_lora()
        
        print("\nModel loaded successfully!")
        self.model.print_trainable_parameters()
    
    def _apply_lora(self):
        """Apply LoRA configuration"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
    
    def train(self, split_dataset, output_dir: Optional[str] = None):
        """Train the model"""
        print("\n6. Starting training...")
        
        output_dir = output_dir or self.train_config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.train_config.num_train_epochs,
            per_device_train_batch_size=self.train_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.train_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
            warmup_steps=self.train_config.warmup_steps,
            logging_steps=self.train_config.logging_steps,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=self.train_config.learning_rate,
            fp16=True,
            push_to_hub=False,
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=self.train_config.save_total_limit
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
        
        # Start training
        train_result = trainer.train()
        
        # Save model
        print("\n7. Saving model...")
        self.save_model(output_dir, trainer)
        
        # Evaluate
        print("\n8. Evaluating...")
        eval_results = trainer.evaluate()
        
        print("\nTraining completed!")
        print(f"Training loss: {train_result.training_loss:.4f}")
        print(f"Evaluation loss: {eval_results['eval_loss']:.4f}")
        
        return trainer, train_result, eval_results
    
    def save_model(self, output_dir: str, trainer):
        """Save the trained model"""
        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training configuration
        config_data = {
            "base_model": self.config.model_name,
            "fine_tuned_on": "genetics_data",
            "training_date": datetime.now().isoformat(),
            "training_config": {
                "epochs": self.train_config.num_train_epochs,
                "batch_size": self.train_config.per_device_train_batch_size,
                "learning_rate": self.train_config.learning_rate,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha
            },
            "dataset_info": {
                "training_samples": len(trainer.train_dataset),
                "validation_samples": len(trainer.eval_dataset)
            }
        }
        
        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print(f"Model saved to: {output_dir}")
        print(f"Config saved to: {config_path}")
    
    def test_generation(self, test_prompt: str, **kwargs):
        """Test model generation"""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Move model to evaluation mode
        self.model.eval()
        
        # Format prompt
        formatted_prompt = f"<s>[INST] {test_prompt} [/INST]"
        
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
                max_new_tokens=kwargs.get('max_new_tokens', 200),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.95),
                repetition_penalty=kwargs.get('repetition_penalty', 1.1),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        return response

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune Mistral-7B on genetics data')
    parser.add_argument('--data', type=str, default=None, help='Path to training data')
    parser.add_argument('--output', type=str, default='./models/mistral-7b-genetics', 
                       help='Output directory for fine-tuned model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', type=str, help='Test the model with a question')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    
    args = parser.parse_args()
    
    # Update config
    if args.epochs:
        training_config.num_train_epochs = args.epochs
    if args.batch_size:
        training_config.per_device_train_batch_size = args.batch_size
        training_config.per_device_eval_batch_size = args.batch_size
    
    trainer = MistralGeneticsTrainer()
    
    if args.train:
        # Prepare and train
        split_dataset = trainer.prepare_training(args.data)
        trainer.load_model()
        trainer.train(split_dataset, args.output)
    
    elif args.test:
        # Test with a question
        trainer.load_model()
        response = trainer.test_generation(args.test)
        print(f"\n{'='*60}")
        print(f"Test Question: {args.test}")
        print(f"{'='*60}")
        print(f"\nGenerated Response:\n{response}")
        print(f"\n{'='*60}")
    
    else:
        print("\nUsage:")
        print("  --train : Train the model")
        print("  --test 'question' : Test the model")
        print("\nExamples:")
        print("  python train_model.py --train --data ./data/genetics_qa.json")
        print("  python train_model.py --test 'What is DNA?'")

if __name__ == "__main__":
    main()
