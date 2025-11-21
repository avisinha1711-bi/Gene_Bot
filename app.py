from flask import Flask, render_template, request, jsonify
from chatbot import GeneticsChatbot
from note_manager import NoteManager
import os

app = Flask(__name__)
chatbot = GeneticsChatbot()
note_manager = NoteManager()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    response = chatbot.get_response(user_input)
    return jsonify({'response': response})

@app.route('/notes', methods=['GET'])
def get_notes():
    notes = note_manager.get_all_notes()
    return jsonify({'notes': notes})

@app.route('/notes', methods=['POST'])
def add_note():
    data = request.json
    title = data.get('title', '')
    content = data.get('content', '')
    category = data.get('category', 'general')
    
    if title and content:
        note_id = note_manager.add_note(title, content, category)
        return jsonify({'success': True, 'note_id': note_id})
    else:
        return jsonify({'success': False, 'error': 'Title and content are required'})

@app.route('/notes/search', methods=['POST'])
def search_notes():
    query = request.json.get('query', '')
    results = note_manager.search_notes(query)
    return jsonify({'results': results})

@app.route('/notes/<note_id>', methods=['DELETE'])
def delete_note(note_id):
    success = note_manager.delete_note(note_id)
    return jsonify({'success': success})

@app.route('/stats', methods=['GET'])
def get_stats():
    stats = note_manager.get_note_stats()
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
