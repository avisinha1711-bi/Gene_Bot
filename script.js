// ===============================
// Intelligent Chatbot Frontend
// ===============================

const API_BASE_URL = 'http://localhost:5000/api';
let conversationId = generateConversationId();

// ===============================
// INITIALIZATION
// ===============================
document.addEventListener('DOMContentLoaded', function() {
    initializeChat();
    checkServerHealth();
});

async function initializeChat() {
    // Add Enter key support for chat
    document.getElementById('user-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Add input event for real-time validation
    document.getElementById('user-input').addEventListener('input', function(e) {
        validateInput(e.target.value);
    });
}

// ===============================
// CHAT FUNCTIONALITY
// ===============================
async function sendMessage() {
    const userInput = document.getElementById('user-input');
    const message = userInput.value.trim();
    
    if (!message) return;
    
    // Disable input while processing
    userInput.disabled = true;
    document.querySelector('.send-button').disabled = true;
    
    try {
        // Add user message to chat
        addMessageToChat('user', message);
        userInput.value = '';
        
        // Show typing indicator
        showTypingIndicator();
        
        // Send to intelligent backend
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                conversation_id: conversationId
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Hide typing indicator
        hideTypingIndicator();
        
        // Add AI response to chat
        if (data.response) {
            addMessageToChat('bot', data.response);
        } else {
            throw new Error('No response from server');
        }
        
    } catch (error) {
        console.error('Error sending message:', error);
        hideTypingIndicator();
        addMessageToChat('bot', 'Sorry, I encountered an error. Please try again.');
    } finally {
        // Re-enable input
        userInput.disabled = false;
        document.querySelector('.send-button').disabled = false;
        userInput.focus();
    }
}

// ===============================
// UI MANAGEMENT
// ===============================
function addMessageToChat(sender, message) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const senderName = sender === 'user' ? 'You' : 'GeneBot';
    messageDiv.innerHTML = `<strong>${senderName}:</strong> ${escapeHtml(message)}`;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Add animation
    messageDiv.style.animation = 'fadeIn 0.3s ease';
}

function showTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    typingIndicator.style.display = 'block';
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    typingIndicator.style.display = 'none';
}

function validateInput(input) {
    // Basic input validation
    if (input.length > 500) {
        document.getElementById('user-input').value = input.substring(0, 500);
    }
}

// ===============================
// UTILITY FUNCTIONS
// ===============================
function generateConversationId() {
    return 'conv_' + Date.now().toString(36) + Math.random().toString(36).substr(2);
}

function escapeHtml(unsafe) {
    if (!unsafe) return '';
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

async function checkServerHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.model_loaded) {
            console.log('✅ AI model loaded successfully');
        } else {
            console.warn('⚠️ AI model not loaded - using fallback mode');
        }
    } catch (error) {
        console.error('❌ Server not reachable:', error);
        addMessageToChat('bot', 'Note: Server connection issue. Some features may be limited.');
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl + Enter to send
    if (e.ctrlKey && e.key === 'Enter') {
        sendMessage();
    }
    
    // Escape to clear input
    if (e.key === 'Escape') {
        document.getElementById('user-input').value = '';
    }
});
