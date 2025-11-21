// Global variables
let currentNotes = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadNotes();
    loadStats();
    
    // Add Enter key support for chat
    document.getElementById('user-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Add Enter key support for note title
    document.getElementById('note-title').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            document.getElementById('note-content').focus();
        }
    });
});

// Chat functionality
function sendMessage() {
    const userInput = document.getElementById('user-input');
    const message = userInput.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addMessageToChat('user', message);
    userInput.value = '';
    
    // Send to backend
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        addMessageToChat('bot', data.response);
    })
    .catch(error => {
        console.error('Error:', error);
        addMessageToChat('bot', 'Sorry, I encountered an error. Please try again.');
    });
}

function addMessageToChat(sender, message) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.innerHTML = `<strong>${sender === 'user' ? 'You' : 'GeneBot'}:</strong> ${message}`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function setSuggestion(question) {
    document.getElementById('user-input').value = question;
    document.getElementById('user-input').focus();
}

// Notes functionality
function loadNotes() {
    fetch('/notes')
        .then(response => response.json())
        .then(data => {
            currentNotes = data.notes;
            displayNotes(currentNotes);
        })
        .catch(error => {
            console.error('Error loading notes:', error);
        });
}

function displayNotes(notes) {
    const notesList = document.getElementById('notes-list');
    
    if (notes.length === 0) {
        notesList.innerHTML = '<p>No notes yet. Start by adding some notes above!</p>';
        return;
    }
    
    notesList.innerHTML = notes.map(note => `
        <div class="note-item">
            <div class="note-header">
                <div class="note-title">${escapeHtml(note.title)}</div>
                <div class="note-category">${escapeHtml(note.category)}</div>
            </div>
            <div class="note-content">${escapeHtml(note.content)}</div>
            <div class="note-footer">
                <span>${note.timestamp}</span>
                <button class="delete-note" onclick="deleteNote('${note.id}')">Delete</button>
            </div>
        </div>
    `).join('');
}

function addNote() {
    const title = document.getElementById('note-title').value.trim();
    const content = document.getElementById('note-content').value.trim();
    const category = document.getElementById('note-category').value.trim() || 'general';
    
    if (!title || !content) {
        alert('Please enter both title and content for the note.');
        return;
    }
    
    fetch('/notes', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            title: title,
            content: content,
            category: category
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Clear form
            document.getElementById('note-title').value = '';
            document.getElementById('note-content').value = '';
            document.getElementById('note-category').value = '';
            
            // Reload notes
            loadNotes();
            loadStats();
        } else {
            alert('Error saving note: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error saving note. Please try again.');
    });
}

function searchNotes() {
    const query = document.getElementById('search-query').value.trim();
    
    if (!query) {
        loadNotes();
        return;
    }
    
    fetch('/notes/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query })
    })
    .then(response => response.json())
    .then(data => {
        displayNotes(data.results);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function clearSearch() {
    document.getElementById('search-query').value = '';
    loadNotes();
}

function deleteNote(noteId) {
    if (!confirm('Are you sure you want to delete this note?')) {
        return;
    }
    
    fetch(`/notes/${noteId}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            loadNotes();
            loadStats();
        } else {
            alert('Error deleting note.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error deleting note. Please try again.');
    });
}

function loadStats() {
    fetch('/stats')
        .then(response => response.json())
        .then(data => {
            displayStats(data);
        })
        .catch(error => {
            console.error('Error loading stats:', error);
        });
}

function displayStats(stats) {
    const statsDiv = document.getElementById('stats');
    
    let statsHTML = '<h3>📊 Note Statistics</h3>';
    statsHTML += `<div class="stat-item"><span>Total Notes:</span><span>${stats.total_notes}</span></div>`;
    
    if (stats.categories.length > 0) {
        statsHTML += '<div class="stat-item"><strong>Categories:</strong><span></span></div>';
        stats.categories.forEach(category => {
            const count = stats.category_counts[category] || 0;
            statsHTML += `<div class="stat-item"><span>• ${category}:</span><span>${count}</span></div>`;
        });
    }
    
    statsDiv.innerHTML = statsHTML;
}

// Utility function to escape HTML
function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
