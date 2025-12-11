"""
Optimized inference for Mistral-7B genetics chatbot.
"""

import torch
import gc
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import logging
from functools import lru_cache

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
    TextStreamer
)
from config import model_config, chatbot_config
from knowledge_base import GeneticsKnowledgeBase

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Optimize model loading and inference"""
    
    @staticmethod
    def cleanup_memory():
        """Clean up GPU memory"""
        torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def get_quantization_config():
        """Get quantization configuration"""
        return BitsAndBytesConfig(
            load_in_4bit=model_config.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, model_config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True
        )

class MistralGeneticsBot:
    def __init__(self, model_path: Optional[str] = None, use_rag: bool = True):
        """Initialize Mistral-7B genetics bot"""
        self.config = model_config
        self.chatbot_config = chatbot_config
        self.model_path = model_path or self.config.model_path
        self.use_rag = use_rag
        
        # Initialize knowledge base if using RAG
        self.knowledge_base = None
        if self.use_rag:
            try:
                self.knowledge_base = GeneticsKnowledgeBase(
                    self.chatbot_config.knowledge_base_path,
                    self.chatbot_config.vector_db_path
                )
                logger.info("Knowledge base loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load knowledge base: {e}")
                self.use_rag = False
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.streamer = None
        
        # Conversation history
        self.conversation_history = []
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the Mistral-7B model with optimization"""
        try:
            logger.info("Loading Mistral-7B model...")
            ModelOptimizer.cleanup_memory()
            
            # Configure quantization
            bnb_config = None
            if self.config.load_in_4bit:
                bnb_config = ModelOptimizer.get_quantization_config()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add special tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.chat_template is None:
                self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% else %}{{ message['content'] }}{% endif %}{% endfor %}"
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.float16 if self.config.load_in_4bit else torch.float32,
                device_map=self.config.device_map,
                trust_remote_code=True,
                use_safetensors=True,
                low_cpu_mem_usage=True
            )
            
            # Create text streamer
            self.streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16 if self.config.load_in_4bit else torch.float32,
                device_map=self.config.device_map
            )
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    @lru_cache(maxsize=128)
    def format_prompt(self, query: str, context: str = "") -> str:
        """Format prompt for Mistral instruction following (cached)"""
        if context:
            prompt = f"""<s>[INST] <<SYS>>
You are GeneBot, an expert genetics assistant with specialized knowledge in:
- Molecular biology and genetics
- Genetic disorders and inheritance patterns
- DNA sequencing and analysis
- CRISPR and gene editing technologies
- Pharmacogenomics and personalized medicine
<</SYS>>

Context Information:
{context[:2000]}  # Limit context length

Based on the above context and your expertise, answer the following question accurately and comprehensively.

Question: {query}

Provide a detailed, accurate answer citing relevant principles and examples. If information is unavailable in the context, use your general genetics knowledge.
Answer: [/INST]"""
        else:
            prompt = f"""<s>[INST] <<SYS>>
You are GeneBot, an expert genetics assistant. Provide accurate, detailed explanations about genetics and molecular biology.
<</SYS>>

Question: {query}

Answer the question comprehensively based on established genetics principles and current scientific understanding.
Answer: [/INST]"""
        
        return prompt
    
    def retrieve_context(self, query: str) -> str:
        """Retrieve relevant context from knowledge base"""
        if not self.use_rag or not self.knowledge_base:
            return ""
        
        try:
            results = self.knowledge_base.search(query, k=self.chatbot_config.rag_top_k)
            
            if not results:
                return ""
            
            # Format context with citations
            context_parts = ["Relevant Context:"]
            for i, result in enumerate(results, 1):
                source = result['metadata'].get('source', 'Unknown')
                text = result['text'][:500]  # Limit text length
                context_parts.append(f"[{i}] Source: {source}")
                context_parts.append(f"{text}")
                if len(result['text']) > 500:
                    context_parts.append("...")
                context_parts.append("")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""
    
    def generate_response(self, query: str, stream: bool = False, **generation_kwargs) -> Dict[str, Any]:
        """Generate response for a query"""
        try:
            # Retrieve context if using RAG
            context = ""
            if self.use_rag:
                context = self.retrieve_context(query)
            
            # Format prompt
            prompt = self.format_prompt(query, context)
            
            # Set generation parameters
            gen_params = {
                'max_new_tokens': generation_kwargs.get('max_new_tokens', self.config.max_new_tokens),
                'temperature': generation_kwargs.get('temperature', self.config.temperature),
                'top_p': generation_kwargs.get('top_p', self.config.top_p),
                'repetition_penalty': generation_kwargs.get('repetition_penalty', self.config.repetition_penalty),
                'do_sample': True,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'num_return_sequences': 1
            }
            
            # Add streamer if streaming
            if stream and self.streamer:
                gen_params['streamer'] = self.streamer
            
            # Generate response
            sequences = self.pipeline(
                prompt,
                **gen_params
            )
            
            # Extract response
            generated_text = sequences[0]['generated_text']
            
            # Extract just the assistant's response
            response = generated_text
            if "[/INST]" in generated_text:
                response = generated_text.split("[/INST]")[-1].strip()
            
            # Clean response
            response = self._clean_response(response)
            
            # Update conversation history
            self._update_history(query, response, context)
            
            return {
                'query': query,
                'response': response,
                'context_used': bool(context),
                'context_sources': self._extract_sources(context) if context else [],
                'timestamp': datetime.now().isoformat(),
                'model': self.model_path,
                'generation_params': gen_params
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'query': query,
                'response': f"I apologize, but I encountered an error: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the response"""
        import re
        
        # Remove special tokens
        tokens_to_remove = [
            r'<s>', r'</s>', r'\[INST\]', r'\[/INST\]',
            r'<<SYS>>', r'<</SYS>>'
        ]
        
        for token in tokens_to_remove:
            response = re.sub(token, '', response)
        
        # Clean up whitespace
        response = re.sub(r'\n\s*\n', '\n\n', response)
        response = response.strip()
        
        # Format bullet points and lists
        response = re.sub(r'^\s*[-*•]\s*', '• ', response, flags=re.MULTILINE)
        
        return response
    
    def _extract_sources(self, context: str) -> List[str]:
        """Extract source references from context"""
        import re
        sources = re.findall(r'Source:\s*(.+)', context)
        return list(set(sources))
    
    def _update_history(self, query: str, response: str, context: str = ""):
        """Update conversation history"""
        history_item = {
            'id': len(self.conversation_history) + 1,
            'query': query,
            'response': response,
            'context_used': bool(context),
            'context_length': len(context) if context else 0,
            'response_length': len(response),
            'timestamp': datetime.now().isoformat()
        }
        
        self.conversation_history.append(history_item)
        
        # Limit history size
        if len(self.conversation_history) > self.chatbot_config.max_history:
            self.conversation_history = self.conversation_history[-self.chatbot_config.max_history:]
    
    def chat(self, message: str, **kwargs) -> str:
        """Simple chat interface"""
        result = self.generate_response(message, **kwargs)
        return result['response']
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get conversation history"""
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_path': self.model_path,
            'model_type': type(self.model).__name__,
            'use_rag': self.use_rag,
            'device': str(self.model.device) if self.model else 'N/A',
            'parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            'quantization': '4-bit' if self.config.load_in_4bit else 'None'
        }
