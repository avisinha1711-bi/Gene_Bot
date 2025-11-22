class ChatBot {
    constructor() {
        this.chatHistory = [];
        this.apiBase = window.location.hostname === 'localhost' 
            ? 'http://localhost:5000' 
            : 'https://your-backend-url.herokuapp.com';
    }

    async sendMessage(message) {
        try {
            const response = await fetch(`${this.apiBase}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();
            
            if (data.status === 'success') {
                return data.response;
            } else {
                return 'Sorry, I encountered an error. Please try again.';
            }
        } catch (error) {
            console.error('Error:', error);
            return 'Connection error. Please check if the server is running.';
        }
    }

    // UI interaction methods
    async handleUserInput() {
        const userInput = document.getElementById('user-input').value;
        if (!userInput.trim()) return;

        this.addMessage('user', userInput);
        document.getElementById('user-input').value = '';

        // Show typing indicator
        this.showTypingIndicator();

        const botResponse = await this.sendMessage(userInput);
        
        this.hideTypingIndicator();
        this.addMessage('bot', botResponse);
    }

    addMessage(sender, message) {
        const chatBox = document.getElementById('chat-box');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        messageDiv.textContent = message;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    showTypingIndicator() {
        const chatBox = document.getElementById('chat-box');
        const typingDiv = document.createElement('div');
        typingDiv.id = 'typing-indicator';
        typingDiv.className = 'message bot-message typing';
        typingDiv.textContent = 'Typing...';
        chatBox.appendChild(typingDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    hideTypingIndicator() {
        const typingDiv = document.getElementById('typing-indicator');
        if (typingDiv) {
            typingDiv.remove();
        }
    }
}

// Initialize chatbot when page loads
document.addEventListener('DOMContentLoaded', function() {
    window.chatBot = new ChatBot();
    
    // Enter key support
    document.getElementById('user-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            window.chatBot.handleUserInput();
        }
    });
});
