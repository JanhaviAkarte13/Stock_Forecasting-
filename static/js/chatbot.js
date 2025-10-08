// Chatbot Functionality

class StockChatbot {
    constructor(containerId, formId, inputId) {
        this.container = document.getElementById(containerId);
        this.form = document.getElementById(formId);
        this.input = document.getElementById(inputId);
        this.messageHistory = [];
        this.isTyping = false;
        
        this.init();
    }
    
    init() {
        // Add welcome message
        this.addBotMessage("Hello! 👋 I'm your AI stock assistant. Ask me anything about stocks, markets, or trading!");
        
        // Set up event listeners
        if (this.form) {
            this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        }
        
        // Load suggestions
        this.loadSuggestions();
    }
    
    async handleSubmit(e) {
        e.preventDefault();
        
        const question = this.input.value.trim();
        if (!question || this.isTyping) return;
        
        // Add user message
        this.addUserMessage(question);
        this.input.value = '';
        
        // Show typing indicator
        const typingId = this.addTypingIndicator();
        this.isTyping = true;
        
        try {
            // Send request to backend
            const response = await fetch('/chatbot/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question: question })
            });
            
            const data = await response.json();
            
            // Remove typing indicator
            this.removeMessage(typingId);
            
            if (data.success) {
                this.addBotMessage(data.answer);
            } else {
                this.addBotMessage("❌ Sorry, I encountered an error. Please try again.");
            }
        } catch (error) {
            this.removeMessage(typingId);
            this.addBotMessage("❌ Connection error. Please check your internet connection and try again.");
        } finally {
            this.isTyping = false;
        }
    }
    
    addUserMessage(text) {
        const messageDiv = this.createMessage(text, 'user');
        this.container.appendChild(messageDiv);
        this.scrollToBottom();
        this.messageHistory.push({ role: 'user', content: text });
    }
    
    addBotMessage(text) {
        const messageDiv = this.createMessage(text, 'bot');
        this.container.appendChild(messageDiv);
        this.scrollToBottom();
        this.messageHistory.push({ role: 'bot', content: text });
    }
    
    addTypingIndicator() {
        const id = 'typing-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.id = id;
        messageDiv.className = 'mb-3 text-start';
        messageDiv.innerHTML = `
            <div class="d-inline-block p-3 rounded bg-light" style="max-width: 80%;">
                <span class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </span>
                Thinking...
            </div>
        `;
        this.container.appendChild(messageDiv);
        this.scrollToBottom();
        return id;
    }
    
    createMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `mb-3 ${sender === 'user' ? 'text-end' : 'text-start'}`;
        
        const bubble = document.createElement('div');
        bubble.className = `d-inline-block p-3 rounded ${sender === 'user' ? 'bg-primary text-white' : 'bg-light'}`;
        bubble.style.maxWidth = '80%';
        bubble.style.wordWrap = 'break-word';
        
        // Format text with line breaks
        bubble.innerHTML = text.replace(/\n/g, '<br>');
        
        // Add icon
        const icon = document.createElement('span');
        icon.className = 'me-2';
        icon.textContent = sender === 'user' ? '👤' : '🤖';
        
        if (sender === 'bot') {
            bubble.insertBefore(icon, bubble.firstChild);
        } else {
            bubble.appendChild(icon);
        }
        
        messageDiv.appendChild(bubble);
        return messageDiv;
    }
    
    removeMessage(id) {
        const element = document.getElementById(id);
        if (element) {
            element.remove();
        }
    }
    
    scrollToBottom() {
        this.container.scrollTop = this.container.scrollHeight;
    }
    
    clearChat() {
        this.container.innerHTML = '';
        this.messageHistory = [];
        this.addBotMessage("Chat cleared! How can I help you?");
    }
    
    async loadSuggestions() {
        try {
            const response = await fetch('/chatbot/suggestions');
            const data = await response.json();
            
            if (data.success && data.suggestions) {
                this.displaySuggestions(data.suggestions);
            }
        } catch (error) {
            console.error('Error loading suggestions:', error);
        }
    }
    
    displaySuggestions(suggestions) {
        const container = document.getElementById('suggestions');
        if (!container) return;
        
        container.innerHTML = '';
        
        suggestions.forEach(suggestion => {
            const badge = document.createElement('span');
            badge.className = 'badge bg-light text-dark border me-2 mb-2';
            badge.style.cursor = 'pointer';
            badge.textContent = suggestion;
            
            badge.addEventListener('click', () => {
                // Remove emojis and set input value
                const cleanText = suggestion.replace(/[📈💰📊🔍💡🎯📉🌟⚡🚀]/g, '').trim();
                this.input.value = cleanText;
                this.form.dispatchEvent(new Event('submit'));
            });
            
            container.appendChild(badge);
        });
    }
}

// CSS for typing indicator
const style = document.createElement('style');
style.textContent = `
    .typing-indicator {
        display: inline-block;
    }
    
    .typing-indicator span {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #6c757d;
        margin: 0 2px;
        animation: typing 1.4s infinite;
    }
    
    .typing-indicator span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-indicator span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
        }
        30% {
            transform: translateY(-10px);
        }
    }
`;
document.head.appendChild(style);

// Initialize chatbot when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('chatContainer')) {
        window.stockChatbot = new StockChatbot('chatContainer', 'chatForm', 'userQuestion');
    }
});