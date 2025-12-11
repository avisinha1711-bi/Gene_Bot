"""
Flask API for deploying the genetics chatbot.
"""

from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
import json
from datetime import datetime
import os
import logging
from functools import wraps
import time

from mistral_inference import MistralGeneticsBot
from knowledge_base import GeneticsKnowledgeBase
from config import chatbot_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Rate limiting
request_timestamps = {}

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ip = request.remote_addr
        current_time = time.time()
        
        # Clean old timestamps
        request_timestamps[ip] = [
            ts for ts in request_timestamps.get(ip, []) 
            if current_time - ts < 60
        ]
        
        # Check rate limit
        if len(request_timestamps.get(ip, [])) >= chatbot_config.rate_limit_per_minute:
            return jsonify({
                'error': 'Rate limit exceeded. Please try again later.'
            }), 429
        
        # Add current timestamp
        if ip not in request_timestamps:
            request_timestamps[ip] = []
        request_timestamps[ip].append(current_time)
        
        return f(*args, **kwargs)
    return decorated_function

# Initialize chatbot
chatbot = None
knowledge_base = None

def initialize_services():
    """Initialize chatbot and knowledge base services"""
    global chatbot, knowledge_base
    
    try:
        logger.info("Initializing GeneBot...")
        chatbot = MistralGeneticsBot(use_rag=True)
        knowledge_base = GeneticsKnowledgeBase(
            chatbot_config.knowledge_base_path,
            chatbot_config.vector_db_path
        )
        logger.info("Services initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

# Initialize on startup
with app.app_context():
    initialize_services()

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
@rate_limit
def chat():
    """Handle chat requests"""
    try:
        # Validate request
        if request.content_length and request.content_length > chatbot_config.max_request_size:
            return jsonify({'error': 'Request too large'}), 413
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Check if streaming is requested
        stream = data.get('stream', False)
        
        if stream:
            def generate():
                # This would need to be implemented with proper streaming support
                result = chatbot.generate_response(message, stream=True)
                yield f"data: {json.dumps({'chunk': result['response']})}\n\n"
                yield "data: [DONE]\n\n"
            
            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:
            # Generate response
            result = chatbot.generate_response(message)
            
            return jsonify({
                'success': True,
                'response': result['response'],
                'context_used': result.get('context_used', False),
                'context_sources': result.get('context_sources', []),
                'timestamp': result.get('timestamp'),
                'history_length': len(chatbot.conversation_history),
                'model_info': chatbot.get_model_info()
            })
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    try:
        limit = request.args.get('limit', default=None, type=int)
        
        return jsonify({
            'success': True,
            'history': chatbot.get_conversation_history(limit),
            'total': len(chatbot.conversation_history)
        })
    
    except Exception as e:
        logger.error(f"Error getting history: {e}")
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
            'message': 'History cleared successfully'
        })
    
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/knowledge-base/stats', methods=['GET'])
def kb_stats():
    """Get knowledge base statistics"""
    try:
        stats = knowledge_base.get_statistics()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    
    except Exception as e:
        logger.error(f"Error getting KB stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/knowledge-base/add', methods=['POST'])
def kb_add():
    """Add document to knowledge base"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        text = data.get('text', '').strip()
        source = data.get('source', 'api_upload')
        metadata = data.get('metadata', {})
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        knowledge_base.add_document(text, source, metadata)
        
        return jsonify({
            'success': True,
            'message': 'Document added successfully',
            'document_id': len(knowledge_base.documents) - 1
        })
    
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/knowledge-base/search', methods=['POST'])
def kb_search():
    """Search knowledge base"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        k = data.get('k', 3)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        results = knowledge_base.search(query, k)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'count': len(results)
        })
    
    except Exception as e:
        logger.error(f"Error searching KB: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        info = chatbot.get_model_info()
        
        return jsonify({
            'success': True,
            'model_info': info
        })
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if services are running
        model_ok = chatbot is not None and chatbot.model is not None
        kb_ok = knowledge_base is not None
        
        status = 'healthy' if model_ok and kb_ok else 'degraded'
        
        return jsonify({
            'status': status,
            'service': 'GeneBot API',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': model_ok,
            'knowledge_base_loaded': kb_ok,
            'version': '1.0.0'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/docs')
def api_docs():
    """API documentation"""
    docs = {
        'endpoints': {
            'POST /api/chat': 'Send a message to the chatbot',
            'GET /api/history': 'Get conversation history',
            'POST /api/clear': 'Clear conversation history',
            'GET /api/knowledge-base/stats': 'Get knowledge base statistics',
            'POST /api/knowledge-base/add': 'Add document to knowledge base',
            'POST /api/knowledge-base/search': 'Search knowledge base',
            'GET /api/model/info': 'Get model information',
            'GET /api/health': 'Health check',
            'GET /': 'Web interface'
        },
        'parameters': {
            'chat': {
                'message': 'string (required) - The message to send',
                'stream': 'boolean (optional) - Whether to stream the response'
            }
        }
    }
    
    return jsonify(docs)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'docs': '/api/docs'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', chatbot_config.port))
    
    # For production, use gunicorn instead
    if os.environ.get('FLASK_ENV') == 'production':
        print(f"Production mode - Use: gunicorn -w 4 -b {chatbot_config.host}:{port} app:app")
    else:
        app.run(
            host=chatbot_config.host,
            port=port,
            debug=chatbot_config.debug,
            threaded=True
        )
