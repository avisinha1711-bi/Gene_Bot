"""
Configuration settings for the Genetics Chatbot.
"""

import os
from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    model_path: str = "./models/mistral-7b-genetics"
    load_in_4bit: bool = True
    device_map: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

@dataclass
class ChatbotConfig:
    """Chatbot configuration"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    knowledge_base_path: str = "./data/knowledge_base"
    max_history: int = 20
    rag_top_k: int = 3
    vector_db_path: str = "./data/vector_db"
    
    # API limits
    max_request_size: int = 1024 * 1024  # 1MB
    rate_limit_per_minute: int = 60

@dataclass
class TrainingConfig:
    """Training configuration"""
    output_dir: str = "./models/mistral-7b-finetuned"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 3
    eval_steps: int = 50
    save_steps: int = 100

# Create configuration instances
model_config = ModelConfig()
chatbot_config = ChatbotConfig()
training_config = TrainingConfig()

# Environment variables override
if os.environ.get("MODEL_PATH"):
    model_config.model_path = os.environ.get("MODEL_PATH")
if os.environ.get("HOST"):
    chatbot_config.host = os.environ.get("HOST")
if os.environ.get("PORT"):
    chatbot_config.port = int(os.environ.get("PORT"))
if os.environ.get("DEBUG"):
    chatbot_config.debug = os.environ.get("DEBUG").lower() == "true"
