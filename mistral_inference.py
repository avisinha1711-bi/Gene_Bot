# mistral_inference.py
"""
Optimized inference for Mistral-7B genetics chatbot.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from typing import Dict, List, Any
import json
from datetime import datetime

from config import model_config, chatbot_config
from knowledge_base import GeneticsKnowledgeBase

class MistralGeneticsBot:
    def __init__(self, model_path=None, use_rag=True):
        """Initialize Mistral-7B genetics bot"""
        self.config = model_config
        self.chatbot_config = chatbot_config
        self.model_path = model_path or self.config.model_path
        self.use_rag = use_rag
        
        # Initialize knowledge base if using RAG
        if self.use_rag:
            self.knowledge_base = GeneticsKnowledgeBase(
                self.chatbot_config.knowledge_base_path
            )
        
        # Initialize model
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # Load model
        self.load_model()
        
        # Conversation history
        self.conversation_history = []
    
    def load_model(self):
        """Load the Mistral-7B model with optimization"""
        print("Loading Mistral-7B model...")
        
        # Configure quantization
        bnb_config = None
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if self.config.load_in_4bit else torch.float32,
            device_map=self.config.device_map,
            trust_remote_code=True
        )
        
        # Create pipeline for easy inference
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16 if self.config.load_in_4bit else torch.float32,
            device_map=self.config.device_map
        )
        
        print("Model loaded successfully!")
    
    def format_prompt(self, query: str, context: str = "") -> str:
        """Format prompt for Mistral instruction following"""
        if context:
            prompt = f"""<s>[INST] You are GeneBot, an expert genetics assistant. Use the following context to answer the question accurately.

Context: {context}

Question: {query}

Provide a detailed, accurate answer based on the context and your genetics knowledge. If the context doesn't contain relevant information, use your general knowledge.

Answer: [/INST]"""
        else:
            prompt = f"""<s>[INST] You are GeneBot, an expert genetics assistant. Answer the following question accurately and in detail.

Question: {query}

Provide a comprehensive answer based on genetics and molecular biology principles.

Answer: [/INST]"""
        
        return prompt
    
    def retrieve_context(self, query: str) -> str:
        """Retrieve relevant context from knowledge base"""
        if not self.use_rag or self.knowledge_base is None:
            return ""
        
        results = self.knowledge_base.search(query, k=3)
        
        if not results:
            return ""
        
        # Format context from top results
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Reference {i}: {result['metadata'].get('source', 'Unknown')}]")
            context_parts.append(f"{result['text'][:500]}...")
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, **generation_kwargs) -> Dict[str, Any]:
        """Generate response for a query"""
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
            'eos_token_id': self.tokenizer.eos_token_id
        }
        
        # Generate response
        sequences = self.pipeline(
            prompt,
            **gen_params
        )
        
        # Extract response
        generated_text = sequences[0]['generated_text']
        
        # Extract just the assistant's response
        if "[/INST]" in generated_text:
            response = generated_text.split("[/INST]")[-1].strip()
        else:
            response = generated_text
        
        # Clean response
        response = self._clean_response(response)
        
        # Update conversation history
        self._update_history(query, response, context)
        
        return {
            'query': query,
            'response': response,
            'context_used': bool(context),
            'context': context if context else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the response"""
        # Remove any remaining special tokens
        for token in ["<s>", "</s>", "[INST]", "[/INST]"]:
            response = response.replace(token, "")
        
        # Remove duplicate newlines
        import re
        response = re.sub(r'\n\s*\n', '\n\n', response)
        
        return response.strip()
    
    def _update_history(self, query: str, response: str, context: str = ""):
        """Update conversation history"""
        history_item = {
            'query': query,
            'response': response,
            'context': context,
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
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


def create_sample_knowledge_base():
    """Create a sample knowledge base for testing"""
    kb = GeneticsKnowledgeBase()
    
    # Add sample genetics knowledge
    sample_docs = [
        {
            "text": "DNA (deoxyribonucleic acid) is a molecule composed of two polynucleotide chains that coil around each other to form a double helix. The polymer carries genetic instructions for the development, functioning, growth and reproduction of all known organisms and many viruses.",
            "source": "genetics_basics",
            "metadata": {"topic": "DNA basics", "type": "educational"}
        },
        {
            "text": "CRISPR-Cas9 is a genome editing technology that allows scientists to make precise changes to DNA sequences in living cells. It consists of two components: the Cas9 enzyme, which cuts DNA, and a guide RNA that directs Cas9 to the specific DNA sequence to be edited.",
            "source": "gene_editing",
            "metadata": {"topic": "CRISPR", "type": "technology"}
        },
        {
            "text": "Genetic inheritance follows Mendelian principles where traits are determined by genes that come in pairs, one from each parent. Dominant traits require only one copy of the gene to be expressed, while recessive traits require two copies.",
            "source": "inheritance",
            "metadata": {"topic": "Mendelian genetics", "type": "principles"}
        }
    ]
    
    for doc in sample_docs:
        kb.add_document(
            text=doc["text"],
            source=doc["source"],
            metadata=doc["metadata"]
        )
    
    print("Sample knowledge base created!")
    return kb


def main():
    """Test the Mistral genetics bot"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Mistral genetics bot')
    parser.add_argument('--question', type=str, help='Question to ask')
    parser.add_argument('--model-path', type=str, help='Path to fine-tuned model')
    parser.add_argument('--create-kb', action='store_true', help='Create sample knowledge base')
    
    args = parser.parse_args()
    
    if args.create_kb:
        create_sample_knowledge_base()
        return
    
    # Initialize bot
    bot = MistralGeneticsBot(model_path=args.model_path)
    
    if args.question:
        response = bot.chat(args.question)
        print(f"\nQuestion: {args.question}")
        print(f"\nAnswer: {response}")
    else:
        # Interactive chat
        print("GeneBot - Genetics Expert Assistant")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
                
                response = bot.chat(user_input)
                print(f"\nGeneBot: {response}")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
