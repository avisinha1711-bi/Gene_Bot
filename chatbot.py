from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging
import os
from config import MODEL_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class IntelligentChatbot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.chat_pipeline = None
        self.conversation_history = []
        self.max_history = 10
        
    def load_model(self):
        """Load the language model"""
        try:
            logger.info("Loading language model...")
            
            # Using a smaller, faster model for demo purposes
            model_name = "microsoft/DialoGPT-small"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Create text generation pipeline
            self.chat_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=MODEL_CONFIG['max_length'],
                temperature=MODEL_CONFIG['temperature'],
                top_p=MODEL_CONFIG['top_p'],
                repetition_penalty=MODEL_CONFIG['repetition_penalty']
            )
            
            # Add genetics-specific context
            self.genetics_context = """
            You are GeneBot, an expert AI assistant specializing in genetics and molecular biology.
            You provide accurate, educational information about:
            - DNA, RNA, genes, chromosomes
            - Protein synthesis, transcription, translation
            - Genetic mutations, inheritance patterns
            - Genomics, bioinformatics
            - Cellular biology and molecular genetics
            
            Always respond in a helpful, educational manner. If unsure about something, acknowledge it.
            """
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def generate_response(self, user_input, conversation_id=None):
        """Generate intelligent response using the language model"""
        try:
            if not self.chat_pipeline:
                return "I'm still initializing. Please try again in a moment."
            
            # Build context with genetics focus
            prompt = f"{self.genetics_context}\n\nUser: {user_input}\nGeneBot:"
            
            # Generate response
            responses = self.chat_pipeline(
                prompt,
                max_new_tokens=MODEL_CONFIG['max_new_tokens'],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = responses[0]['generated_text']
            
            # Extract just the GeneBot response
            if "GeneBot:" in response:
                response = response.split("GeneBot:")[-1].strip()
            
            # Clean up the response
            response = self.clean_response(response)
            
            # Update conversation history
            self.update_conversation_history(user_input, response, conversation_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble processing your request. Could you try rephrasing it?"
    
    def clean_response(self, response):
        """Clean and format the model response"""
        # Remove any trailing incomplete sentences
        if '.' in response:
            response = response.split('.')[0] + '.'
        
        # Ensure response ends with proper punctuation
        if not response.endswith(('.', '!', '?')):
            response += '.'
            
        return response.strip()
    
    def update_conversation_history(self, user_input, response, conversation_id):
        """Maintain conversation context"""
        if len(self.conversation_history) >= self.max_history:
            self.conversation_history.pop(0)
        
        self.conversation_history.append({
            'user': user_input,
            'bot': response,
            'conversation_id': conversation_id
        })

# Initialize chatbot
chatbot = IntelligentChatbot()

@app.route('/')
def serve_frontend():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """API endpoint for chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        conversation_id = data.get('conversation_id')
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        logger.info(f"Received message: {user_message}")
        
        # Generate intelligent response
        response = chatbot.generate_response(user_message, conversation_id)
        
        logger.info(f"Generated response: {response}")
        
        return jsonify({
            'response': response,
            'conversation_id': conversation_id
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': chatbot.chat_pipeline is not None
    })

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    chatbot.conversation_history.clear()
    return jsonify({'status': 'history cleared'})

if __name__ == '__main__':
    # Load model when starting the server
    model_loaded = chatbot.load_model()
    
    if model_loaded:
        print("🤖 GeneBot is starting up...")
        print("🧬 Intelligent genetics chatbot ready!")
        print("📡 Server running on http://localhost:5000")
        
        app.run(
            host=MODEL_CONFIG['host'],
            port=MODEL_CONFIG['port'],
            debug=MODEL_CONFIG['debug']
        )
    else:
        print("❌ Failed to load model. Please check the error logs.")
