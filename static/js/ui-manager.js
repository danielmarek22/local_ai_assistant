import { CONFIG } from './config.js';
import { marked } from 'https://cdn.jsdelivr.net/npm/marked@13.0.2/lib/marked.esm.js';
import DOMPurify from 'https://cdn.jsdelivr.net/npm/dompurify@3.1.6/+esm';

marked.setOptions({
    gfm: true,
    breaks: true
});

export class UIManager {
    constructor() {
        this.chatPanel = document.getElementById('chat-panel');
        this.statusText = document.getElementById('status-text');
        this.chatHistory = document.getElementById('chat-history');
        this.userInput = document.getElementById('user-input');
        this.sendBtn = document.getElementById('send-btn');
        this.chatCloseBtn = document.getElementById('chat-close-btn');
        this.chatOpenBtn = document.getElementById('chat-open-btn');
        
        this.currentAiMessageDiv = null;
        this.chatHistoryStorageKey = null;
        this.defaultMessages = this.serializeChatHistory();
        this.initAutoResize();
        this.initPanelControls();
    }

    initAutoResize() {
        this.userInput.addEventListener('input', () => {
            this.userInput.style.height = 'auto'; // Reset to calculate shrink
            this.userInput.style.height = this.userInput.scrollHeight + 'px';
        });
    }

    initPanelControls() {
        this.chatCloseBtn.addEventListener('click', () => {
            this.chatPanel.classList.add('hidden');
            this.chatOpenBtn.classList.remove('hidden');
        });

        this.chatOpenBtn.addEventListener('click', () => {
            this.chatPanel.classList.remove('hidden');
            this.chatOpenBtn.classList.add('hidden');
            this.userInput.focus();
        });

        this.chatOpenBtn.classList.add('hidden');
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
        const rawText = (this.currentAiMessageDiv.dataset.rawText || '') + text;
        this.setMessageContent(this.currentAiMessageDiv, rawText);
        this.persistChatHistory();
        this.scrollToBottom();
    }

    finalizeAiMessage(text) {
        if (this.currentAiMessageDiv) {
            this.setMessageContent(this.currentAiMessageDiv, text);
            this.persistChatHistory();
            this.currentAiMessageDiv = null;
        }
    }

    createMessageDiv(sender, text) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', sender);
        this.setMessageContent(msgDiv, text);
        this.chatHistory.appendChild(msgDiv);
        this.persistChatHistory();
        this.scrollToBottom();
        return msgDiv;
    }

    setMessageContent(msgDiv, text) {
        msgDiv.dataset.rawText = text;

        if (msgDiv.classList.contains('astra')) {
            const unsafeHtml = marked.parse(text);
            const safeHtml = DOMPurify.sanitize(unsafeHtml, {
                USE_PROFILES: { html: true }
            });
            msgDiv.innerHTML = safeHtml;

            for (const link of msgDiv.querySelectorAll('a')) {
                link.setAttribute('target', '_blank');
                link.setAttribute('rel', 'noopener noreferrer');
            }
            return;
        }

        msgDiv.innerText = text;
    }

    setSessionScope(serverInstanceId) {
        const nextStorageKey = `${CONFIG.UI.STORAGE_KEYS.CHAT_HISTORY}:${serverInstanceId}`;
        if (this.chatHistoryStorageKey === nextStorageKey) return;

        this.chatHistoryStorageKey = nextStorageKey;
        this.currentAiMessageDiv = null;
        this.restoreChatHistory();
    }

    restoreChatHistory() {
        if (!this.chatHistoryStorageKey) return;

        const savedHistory = sessionStorage.getItem(this.chatHistoryStorageKey);
        if (!savedHistory) {
            this.renderMessages(this.defaultMessages);
            this.persistChatHistory();
            return;
        }

        try {
            const messages = JSON.parse(savedHistory);
            if (!Array.isArray(messages) || messages.length === 0) {
                this.renderMessages(this.defaultMessages);
                this.persistChatHistory();
                return;
            }
            this.renderMessages(messages);
        } catch (error) {
            console.warn('Failed to restore chat history from session storage:', error);
            sessionStorage.removeItem(this.chatHistoryStorageKey);
            this.renderMessages(this.defaultMessages);
            this.persistChatHistory();
        }
    }

    persistChatHistory() {
        if (!this.chatHistoryStorageKey) return;

        sessionStorage.setItem(this.chatHistoryStorageKey, JSON.stringify(this.serializeChatHistory()));
    }

    serializeChatHistory() {
        return Array.from(this.chatHistory.querySelectorAll('.message')).map((message) => {
            const sender = Array.from(message.classList).find((className) => className !== 'message') || 'astra';
            return {
                sender,
                text: message.dataset.rawText || ''
            };
        });
    }

    renderMessages(messages) {
        this.chatHistory.replaceChildren();

        for (const message of messages) {
            if (!message || typeof message.sender !== 'string' || typeof message.text !== 'string') {
                continue;
            }

            const msgDiv = document.createElement('div');
            msgDiv.classList.add('message', message.sender);
            this.setMessageContent(msgDiv, message.text);
            this.chatHistory.appendChild(msgDiv);
        }

        this.scrollToBottom();
    }

    scrollToBottom() {
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
    }

    onSend(callback) {
            const handler = (e) => {
                // If it's a keypress (Enter), prevent default newline
                if (e && e.type === 'keypress') {
                    e.preventDefault();
                }

                const text = this.userInput.value.trim();
                if (!text) return;
                
                callback(text);
                
                this.userInput.value = "";
                this.userInput.style.height = 'auto'; // Reset height after sending
            };

            this.sendBtn.addEventListener('click', () => handler());
            
            this.userInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    handler(e);
                }
            });
    }
}
