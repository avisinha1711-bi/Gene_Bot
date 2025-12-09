# config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for Mistral-7B model"""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    model_path: str = "./models/mistral-7b-genetics"
    tokenizer_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    
    # Quantization settings (for memory efficiency)
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    device_map: str = "auto"
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.15
    
    # Training settings
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_r: int = 8
    target_modules = ["q_proj", "v_proj"]

@dataclass
class ChatbotConfig:
    """Configuration for chatbot"""
    knowledge_base_path: str = "./knowledge_base"
    use_rag: bool = True
    similarity_threshold: float = 0.7
    max_history: int = 10
    
    # API settings
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False

# Create configuration instances
model_config = ModelConfig()
chatbot_config = ChatbotConfig()
