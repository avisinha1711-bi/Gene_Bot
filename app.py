from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import json
import os

app = Flask(__name__)
CORS(app)

# Simple genetics knowledge base
GENETICS_KB = {
    "dna": "DNA (Deoxyribonucleic acid) is the hereditary material in humans and almost all other organisms.",
    "gene": "A gene is the basic physical and functional unit of heredity. Genes are made up of DNA.",
    "chromosome": "Chromosomes are thread-like structures located inside the nucleus of animal and plant cells.",
    "mutation": "A mutation is a change in a DNA sequence. Mutations can result from DNA copying mistakes or environmental factors.",
    "genome": "The genome is the entire set of genetic instructions found in a cell."
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').lower()
        
        # Simple response logic - extend this with your AI
        response = "I'm a genetics bot. Ask me about DNA, genes, chromosomes, mutations, or genomes!"
        
        # Check if question matches known topics
        for topic, info in GENETICS_KB.items():
            if topic in user_message:
                response = info
                break
        
        return jsonify({
            'status': 'success',
            'response': response,
            'user_message': user_message
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'response': f'Error processing request: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'Genetics Chatbot'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
