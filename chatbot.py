from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import logging
import os
import sys
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
        self.model_loaded = False
        
    def load_model(self):
        """Load the language model with fallback options"""
        try:
            logger.info("Loading language model...")
            
            # Try to import transformers, but have fallback if not available
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            except ImportError as e:
                logger.error(f"Transformers not available: {str(e)}")
                return False
            
            # Use smaller model for deployment compatibility
            model_name = "microsoft/DialoGPT-small"
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                
                # Create text generation pipeline
                self.chat_pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_length=MODEL_CONFIG.get('max_length', 512),
                    temperature=MODEL_CONFIG.get('temperature', 0.7),
                    top_p=MODEL_CONFIG.get('top_p', 0.9),
                    repetition_penalty=MODEL_CONFIG.get('repetition_penalty', 1.1)
                )
                
                self.model_loaded = True
                logger.info("Model loaded successfully!")
                return True
                
            except Exception as e:
                logger.error(f"Error loading specific model: {str(e)}")
                # Fallback to simple rule-based system
                return self.setup_fallback_system()
            
        except Exception as e:
            logger.error(f"Error in model loading: {str(e)}")
            return self.setup_fallback_system()
    
    def setup_fallback_system(self):
        """Setup fallback rule-based system when model fails to load"""
        logger.info("Setting up fallback rule-based system")
        self.model_loaded = False
        
        # Genetics knowledge base for fallback
        self.genetics_knowledge = {
            "dna": "DNA (Deoxyribonucleic Acid) is the hereditary material in humans and almost all other organisms. It contains the genetic instructions for development, functioning, growth, and reproduction.",
            "rna": "RNA (Ribonucleic Acid) is a nucleic acid that plays crucial roles in coding, decoding, regulation, and expression of genes. Major types include mRNA, tRNA, and rRNA.",
            "gene": "A gene is the basic physical and functional unit of heredity. Genes are made up of DNA and act as instructions to make molecules called proteins.",
            "chromosome": "Chromosomes are thread-like structures located inside the nucleus of animal and plant cells. Each chromosome is made of protein and a single molecule of DNA.",
            "mutation": "A mutation is a change in a DNA sequence. Mutations can result from DNA copying mistakes, exposure to radiation or chemicals, or viral infection.",
            "protein": "Proteins are large, complex molecules that play many critical roles in the body. They do most of the work in cells and are required for the structure, function, and regulation of the body's tissues and organs.",
            "transcription": "Transcription is the process of making an RNA copy of a gene's DNA sequence. This copy, called mRNA, carries the genetic information needed to make proteins.",
            "translation": "Translation is the process where ribosomes in the cytoplasm or ER synthesize proteins after transcription of DNA to RNA in the cell's nucleus.",
            "genotype": "Genotype refers to the genetic constitution of an individual organism. It represents the specific allele combination for a particular gene.",
            "phenotype": "Phenotype refers to the observable physical properties of an organism including appearance, development, and behavior.",
            "allele": "An allele is one of two or more versions of a gene. An individual inherits two alleles for each gene, one from each parent.",
            "heredity": "Heredity is the passing of genetic traits from parents to their offspring through sexual or asexual reproduction.",
            "genome": "A genome is an organism's complete set of DNA, including all of its genes. The human genome contains about 3 billion base pairs.",
            "mitosis": "Mitosis is a process of cell division that results in two genetically identical daughter cells developing from a single parent cell.",
            "meiosis": "Meiosis is a special type of cell division that produces gametes (sperm and egg cells) with half the number of chromosomes of the parent cell."
        }
        
        return True
    
    def generate_fallback_response(self, user_input):
        """Generate response using rule-based system"""
        input_lower = user_input.lower()
        
        # Check for greetings
        if any(word in input_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return "Hello! I'm GeneBot, your genetics assistant. I can help you understand DNA, RNA, genes, mutations, and other genetics concepts. What would you like to know?"
        
        # Check for thanks
        if any(word in input_lower for word in ['thank', 'thanks', 'appreciate']):
            return "You're welcome! I'm happy to help with your genetics questions."
        
        # Direct concept lookup
        for concept, definition in self.genetics_knowledge.items():
            if concept in input_lower:
                return definition
        
        # Pattern-based responses
        if any(phrase in input_lower for phrase in ['what is', 'tell me about', 'explain', 'define']):
            for concept in self.genetics_knowledge:
                if concept in input_lower:
                    return self.genetics_knowledge[concept]
            return "I can explain various genetics concepts like DNA, RNA, genes, chromosomes, mutations, proteins, transcription, translation, genotype, phenotype, alleles, heredity, genome, mitosis, and meiosis. Which specific concept would you like me to explain?"
        
        # Comparison questions
        if 'difference between' in input_lower or 'different from' in input_lower or 'compare' in input_lower:
            if 'dna' in input_lower and 'rna' in input_lower:
                return "DNA vs RNA: DNA is double-stranded with deoxyribose sugar, while RNA is single-stranded with ribose sugar. DNA uses thymine (T), RNA uses uracil (U). DNA stores genetic information long-term, while RNA acts as a messenger in protein synthesis."
            if 'genotype' in input_lower and 'phenotype' in input_lower:
                return "Genotype vs Phenotype: Genotype is the genetic makeup (specific alleles), while phenotype is the observable physical characteristics. Genotype influences phenotype, but environment also affects expression."
            if 'mitosis' in input_lower and 'meiosis' in input_lower:
                return "Mitosis vs Meiosis: Mitosis produces two identical diploid cells for growth/repair. Meiosis produces four genetically diverse haploid gametes for reproduction, involving genetic recombination."
        
        # Process questions
        if any(phrase in input_lower for phrase in ['how does', 'how do', 'how is']):
            if 'transcription' in input_lower:
                return "Transcription has three steps: 1) Initiation - RNA polymerase binds to DNA promoter, 2) Elongation - mRNA is built complementary to DNA, 3) Termination - mRNA is released."
            if 'translation' in input_lower:
                return "Translation: 1) Initiation - ribosome assembles around start codon, 2) Elongation - tRNA brings amino acids, peptide bonds form, 3) Termination - stop codon releases protein."
            if 'dna replication' in input_lower:
                return "DNA replication: 1) Helicase unwinds DNA, 2) Primase adds RNA primers, 3) DNA polymerase adds nucleotides, 4) Leading strand continuous, lagging strand in fragments, 5) Ligase joins fragments."
        
        return "I'm still learning about genetics. Could you try asking about specific concepts like DNA, RNA, genes, mutations, protein synthesis, or inheritance patterns?"
    
    def generate_response(self, user_input, conversation_id=None):
        """Generate intelligent response using available system"""
        try:
            if not self.model_loaded or not self.chat_pipeline:
                return self.generate_fallback_response(user_input)
            
            # Build context with genetics focus
            genetics_context = """You are GeneBot, an expert AI assistant specializing in genetics and molecular biology. You provide accurate, educational information about DNA, RNA, genes, chromosomes, protein synthesis, genetic mutations, inheritance patterns, genomics, and cellular biology. Always respond in a helpful, educational manner."""
            
            prompt = f"{genetics_context}\n\nUser: {user_input}\nGeneBot:"
            
            # Generate response with error handling
            try:
                responses = self.chat_pipeline(
                    prompt,
                    max_new_tokens=MODEL_CONFIG.get('max_new_tokens', 150),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                response = responses[0]['generated_text']
                
                # Extract just the GeneBot response
                if "GeneBot:" in response:
                    response = response.split("GeneBot:")[-1].strip()
                
                # Clean up the response
                response = self.clean_response(response)
                
            except Exception as e:
                logger.error(f"Error in model generation: {str(e)}")
                response = self.generate_fallback_response(user_input)
            
            # Update conversation history
            self.update_conversation_history(user_input, response, conversation_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble processing your request. Could you try rephrasing it?"
    
    def clean_response(self, response):
        """Clean and format the model response"""
        if not response:
            return "I'm not sure how to respond to that. Could you ask about genetics concepts?"
        
        # Remove any trailing incomplete sentences
        if '.' in response:
            last_sentence = response.split('.')[-1].strip()
            if len(last_sentence.split()) < 3:  # If last part is too short, remove it
                response = '.'.join(response.split('.')[:-1]) + '.'
        
        # Ensure response ends with proper punctuation
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
            
        return response.strip() if response else "Please ask me about genetics!"
    
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
    """Serve the main frontend"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving frontend: {str(e)}")
        return """
        <html>
            <head><title>GeneBot - Genetics Chatbot</title></head>
            <body>
                <h1>GeneBot - Genetics Chatbot</h1>
                <p>Backend is running. Frontend template might be missing.</p>
            </body>
        </html>
        """

@app.route('/api/chat', methods=['POST', 'GET'])
def chat_endpoint():
    """API endpoint for chat messages"""
    try:
        if request.method == 'GET':
            return jsonify({'error': 'Use POST method for chat'}), 405
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
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
            'conversation_id': conversation_id,
            'model_used': 'ai' if chatbot.model_loaded else 'fallback'
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'response': 'I apologize, but I encountered an error. Please try again.'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': chatbot.model_loaded,
        'service': 'GeneBot Genetics Chatbot'
    })

@app.route('/api/info', methods=['GET'])
def info():
    """Service information"""
    return jsonify({
        'name': 'GeneBot',
        'version': '1.0',
        'description': 'Intelligent Genetics Chatbot',
        'model_loaded': chatbot.model_loaded,
        'conversation_history_count': len(chatbot.conversation_history)
    })

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    chatbot.conversation_history.clear()
    return jsonify({'status': 'history cleared'})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """Main application entry point"""
    try:
        # Load model when starting the server
        model_loaded = chatbot.load_model()
        
        if model_loaded:
            print("🚀 GeneBot is starting up...")
            if chatbot.model_loaded:
                print("🤖 AI Model loaded successfully!")
            else:
                print("📚 Using fallback rule-based system")
            print("🧬 Intelligent genetics chatbot ready!")
            print(f"📡 Server running on http://{MODEL_CONFIG.get('host', 'localhost')}:{MODEL_CONFIG.get('port', 5000)}")
            
            # Get debug mode from config with safe default
            debug_mode = MODEL_CONFIG.get('debug', False)
            if isinstance(debug_mode, str):
                debug_mode = debug_mode.lower() == 'true'
            
            app.run(
                host=MODEL_CONFIG.get('host', '0.0.0.0'),
                port=int(MODEL_CONFIG.get('port', 5000)),
                debug=debug_mode
            )
        else:
            print("❌ Failed to initialize chatbot service.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Fatal error starting GeneBot: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
