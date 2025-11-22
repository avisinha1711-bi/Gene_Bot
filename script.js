class GeneticsChatInterface {
    constructor() {
        this.conversationHistory = [];
        this.feedbackSystem = new FeedbackSystem();
    }

    async sendMessage(message) {
        // Add user message to chat
        this.addMessageToChat('user', message);
        
        // Send to backend
        const response = await this.fetchBotResponse(message);
        this.addMessageToChat('bot', response.answer);
        
        // Store conversation for learning
        this.conversationHistory.push({
            user: message,
            bot: response.answer,
            timestamp: new Date()
        });
    }

    async fetchBotResponse(message) {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                history: this.conversationHistory.slice(-5) // Last 5 messages
            })
        });
        return await response.json();
    }

    provideFeedback(messageId, isPositive) {
        this.feedbackSystem.recordFeedback(messageId, isPositive);
        
        // Send feedback to backend for learning
        fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message_id: messageId,
                feedback: isPositive ? 'positive' : 'negative'
            })
        });
    }
}

class FeedbackSystem {
    constructor() {
        this.feedbackHistory = [];
    }

    recordFeedback(messageId, isPositive) {
        this.feedbackHistory.push({
            messageId,
            isPositive,
            timestamp: new Date()
        });
    }
}
