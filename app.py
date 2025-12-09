# app.py
"""
Flask API for deploying the genetics chatbot.
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
from datetime import datetime
import os

from mistral_inference import MistralGeneticsBot
from knowledge_base import GeneticsKnowledgeBase
from config import chatbot_config

app = Flask(__name__)
CORS(app)

# Initialize chatbot
print("Initializing GeneBot...")
chatbot = MistralGeneticsBot(use_rag=True)
print("GeneBot initialized successfully!")

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Generate response
        result = chatbot.generate_response(message)
        
        return jsonify({
            'success': True,
            'response': result['response'],
            'context_used': result['context_used'],
            'timestamp': result['timestamp'],
            'history_length': len(chatbot.conversation_history)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    try:
        return jsonify({
            'success': True,
            'history': chatbot.get_conversation_history(),
            'total': len(chatbot.conversation_history)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    try:
        chatbot.clear_history()
        return jsonify({
            'success': True,
            'message': 'History cleared'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/knowledge-base/stats', methods=['GET'])
def kb_stats():
    """Get knowledge base statistics"""
    try:
        kb = GeneticsKnowledgeBase(chatbot_config.knowledge_base_path)
        stats = kb.get_statistics()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'GeneBot API',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': chatbot.model is not None
    })

@app.route('/api/query-test', methods=['POST'])
def query_test():
    """Test endpoint with sample queries"""
    try:
        sample_queries = [
            "What is DNA?",
            "Explain CRISPR gene editing",
            "What causes genetic disorders?",
            "How does inheritance work?",
            "What is RNA sequencing?"
        ]
        
        results = []
        for query in sample_queries:
            response = chatbot.chat(query, max_new_tokens=150)
            results.append({
                'query': query,
                'response': response[:200] + "..." if len(response) > 200 else response
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', chatbot_config.port))
    app.run(
        host=chatbot_config.host,
        port=port,
        debug=chatbot_config.debug
    )
