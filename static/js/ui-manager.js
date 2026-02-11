import { CONFIG } from './config.js';

export class UIManager {
    constructor() {
        this.chatPanel = document.getElementById('chat-panel');
        this.statusText = document.getElementById('status-text');
        this.chatHistory = document.getElementById('chat-history');
        this.userInput = document.getElementById('user-input');
        this.sendBtn = document.getElementById('send-btn');
        
        this.currentAiMessageDiv = null;
    }

    updateStatus(state) {
        // Remove all potential status classes
        this.chatPanel.classList.remove(...Object.keys(CONFIG.UI.STATUS_TEXT).map(s => `status-${s}`));
        this.chatPanel.classList.add(`status-${state}`);

        const text = CONFIG.UI.STATUS_TEXT[state] || CONFIG.UI.STATUS_TEXT['idle'];
        this.statusText.innerText = text;
    }

    appendUserMessage(text) {
        this.createMessageDiv('user', text);
    }

    startAiMessage() {
        if (!this.currentAiMessageDiv) {
            this.currentAiMessageDiv = this.createMessageDiv('astra', '');
        }
    }

    appendToAiMessage(text) {
        if (!this.currentAiMessageDiv) this.startAiMessage();
        this.currentAiMessageDiv.innerText += text;
        this.scrollToBottom();
    }

    finalizeAiMessage(text) {
        if (this.currentAiMessageDiv) {
            this.currentAiMessageDiv.innerText = text;
            this.currentAiMessageDiv = null;
        }
    }

    createMessageDiv(sender, text) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', sender);
        msgDiv.innerText = text;
        this.chatHistory.appendChild(msgDiv);
        this.scrollToBottom();
        return msgDiv;
    }

    scrollToBottom() {
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
    }

    onSend(callback) {
        const handler = () => {
            const text = this.userInput.value.trim();
            if (!text) return;
            callback(text);
            this.userInput.value = "";
        };

        this.sendBtn.addEventListener('click', handler);
        this.userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handler();
        });
    }
}