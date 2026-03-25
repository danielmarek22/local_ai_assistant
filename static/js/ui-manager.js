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
        this.composerMenu = document.getElementById('composer-menu');
        this.composerMenuBtn = document.getElementById('composer-menu-btn');
        this.composerMenuPopover = document.getElementById('composer-menu-popover');
        this.reasoningToggle = document.getElementById('reasoning-toggle');
        this.playbackVolumeInput = document.getElementById('playback-volume');
        this.playbackVolumeValue = document.getElementById('playback-volume-value');
        this.sendBtn = document.getElementById('send-btn');
        this.chatCloseBtn = document.getElementById('chat-close-btn');
        this.chatOpenBtn = document.getElementById('chat-open-btn');
        this.chatTabs = Array.from(document.querySelectorAll('.chat-tab'));
        this.tabPanels = Array.from(document.querySelectorAll('.tab-panel'));
        this.historyList = document.getElementById('history-list');
        this.historyStatus = document.getElementById('history-status');
        this.historyRefreshBtn = document.getElementById('history-refresh-btn');
        this.historyNewChatBtn = document.getElementById('history-new-chat-btn');
        
        this.currentAiMessageDiv = null;
        this.chatHistoryStorageKey = null;
        this.currentSessionId = null;
        this.reasoningEnabledForNextSend = false;
        this.volumeStorageKey = CONFIG.UI.STORAGE_KEYS.AUDIO_VOLUME;
        this.defaultMessages = this.serializeChatHistory();
        this.initAutoResize();
        this.initPanelControls();
        this.initTabs();
        this.initHistoryControls();
        this.initComposerControls();
        this.initConfigControls();
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

    initTabs() {
        for (const tab of this.chatTabs) {
            tab.addEventListener('click', () => {
                this.setActiveTab(tab.dataset.tabTarget);
            });
        }
    }

    setActiveTab(tabId) {
        for (const tab of this.chatTabs) {
            const isActive = tab.dataset.tabTarget === tabId;
            tab.classList.toggle('active', isActive);
            tab.setAttribute('aria-selected', String(isActive));
        }

        for (const panel of this.tabPanels) {
            const isActive = panel.id === tabId;
            panel.classList.toggle('active', isActive);
            panel.setAttribute('aria-hidden', String(!isActive));
        }

        if (tabId === 'chat-view') {
            this.userInput.focus();
            this.scrollToBottom();
        }
    }

    initHistoryControls() {
        this.historyRefreshBtn.addEventListener('click', () => {
            if (this.onHistoryRefreshHandler) {
                this.onHistoryRefreshHandler();
            }
        });

        this.historyNewChatBtn.addEventListener('click', () => {
            if (this.onHistoryNewChatHandler) {
                this.onHistoryNewChatHandler();
            }
        });

        this.historyList.addEventListener('click', (event) => {
            const actionButton = event.target.closest('[data-history-action]');
            if (!actionButton) return;

            const { historyAction, sessionId } = actionButton.dataset;
            if (!sessionId) return;

            if (historyAction === 'open' && this.onHistoryOpenHandler) {
                this.onHistoryOpenHandler(sessionId);
            }

            if (historyAction === 'delete' && this.onHistoryDeleteHandler) {
                this.onHistoryDeleteHandler(sessionId);
            }
        });
    }

    initComposerControls() {
        this.composerMenuBtn.addEventListener('click', (event) => {
            event.stopPropagation();
            this.toggleComposerMenu();
        });

        this.reasoningToggle.addEventListener('click', () => {
            this.reasoningEnabledForNextSend = !this.reasoningEnabledForNextSend;
            this.syncReasoningToggle();
            this.closeComposerMenu();
        });

        document.addEventListener('click', (event) => {
            if (this.composerMenu.contains(event.target)) {
                return;
            }

            this.closeComposerMenu();
        });
    }

    initConfigControls() {
        if (!this.playbackVolumeInput || !this.playbackVolumeValue) {
            return;
        }

        this.setPlaybackVolume(this.readStoredPlaybackVolume(), { persist: false, notify: false });

        this.playbackVolumeInput.addEventListener('input', () => {
            const nextVolume = Number(this.playbackVolumeInput.value) / 100;
            this.setPlaybackVolume(nextVolume, { persist: true, notify: true });
        });
    }

    readStoredPlaybackVolume() {
        try {
            const rawValue = localStorage.getItem(this.volumeStorageKey);
            if (rawValue === null) {
                return CONFIG.AUDIO.DEFAULT_VOLUME;
            }

            const parsedValue = Number(rawValue);
            if (Number.isFinite(parsedValue) && parsedValue >= 0 && parsedValue <= 1) {
                return parsedValue;
            }

            localStorage.removeItem(this.volumeStorageKey);
        } catch (error) {
            console.warn('Failed to restore playback volume:', error);
        }

        return CONFIG.AUDIO.DEFAULT_VOLUME;
    }

    setPlaybackVolume(volume, { persist = true, notify = true } = {}) {
        const nextVolume = Number.isFinite(volume)
            ? Math.min(1, Math.max(0, volume))
            : CONFIG.AUDIO.DEFAULT_VOLUME;
        const percentage = Math.round(nextVolume * 100);

        if (this.playbackVolumeInput) {
            this.playbackVolumeInput.value = String(percentage);
        }
        if (this.playbackVolumeValue) {
            this.playbackVolumeValue.textContent = `${percentage}%`;
        }

        if (persist) {
            try {
                localStorage.setItem(this.volumeStorageKey, String(nextVolume));
            } catch (error) {
                console.warn('Failed to persist playback volume:', error);
            }
        }

        if (notify && this.onVolumeChangeHandler) {
            this.onVolumeChangeHandler(nextVolume);
        }
    }

    getPlaybackVolume() {
        if (!this.playbackVolumeInput) {
            return CONFIG.AUDIO.DEFAULT_VOLUME;
        }

        return Number(this.playbackVolumeInput.value) / 100;
    }

    syncReasoningToggle() {
        this.composerMenuBtn.classList.toggle('active', this.reasoningEnabledForNextSend);
        this.reasoningToggle.classList.toggle('active', this.reasoningEnabledForNextSend);
        this.reasoningToggle.setAttribute('aria-checked', String(this.reasoningEnabledForNextSend));
    }

    toggleComposerMenu() {
        const shouldOpen = this.composerMenuPopover.classList.contains('hidden');
        this.composerMenuPopover.classList.toggle('hidden', !shouldOpen);
        this.composerMenuBtn.setAttribute('aria-expanded', String(shouldOpen));
    }

    closeComposerMenu() {
        this.composerMenuPopover.classList.add('hidden');
        this.composerMenuBtn.setAttribute('aria-expanded', 'false');
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
        if (!this.currentAiMessageDiv && text) {
            this.currentAiMessageDiv = this.createMessageDiv('astra', '');
        }

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

    setSessionScope(serverInstanceId, sessionId) {
        const nextStorageKey = `${CONFIG.UI.STORAGE_KEYS.CHAT_HISTORY}:${serverInstanceId}:${sessionId}`;
        if (this.chatHistoryStorageKey === nextStorageKey) return;

        this.currentSessionId = sessionId;
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

    renderSessionMessages(sessionData) {
        const messages = sessionData.messages.map((message) => ({
            sender: message.role === 'assistant' ? 'astra' : message.role,
            text: message.content,
        }));

        this.currentAiMessageDiv = null;
        this.renderMessages(messages);
        this.persistChatHistory();
    }

    resetChatToDefault() {
        this.currentAiMessageDiv = null;
        this.renderMessages(this.defaultMessages);
        this.persistChatHistory();
    }

    setHistoryLoading(isLoading) {
        this.historyRefreshBtn.disabled = isLoading;
        this.historyRefreshBtn.textContent = isLoading ? 'Refreshing...' : 'Refresh';
        this.historyNewChatBtn.disabled = isLoading;
    }

    setHistoryStatus(message = '', tone = 'info') {
        this.historyStatus.textContent = message;
        this.historyStatus.classList.toggle('hidden', !message);
        this.historyStatus.classList.toggle('error', tone === 'error');
    }

    renderHistorySessions(sessions, activeSessionId) {
        this.historyList.replaceChildren();

        if (!sessions.length) {
            const emptyState = document.createElement('div');
            emptyState.className = 'history-empty';
            emptyState.textContent = 'No saved conversations yet.';
            this.historyList.appendChild(emptyState);
            return;
        }

        for (const session of sessions) {
            const item = document.createElement('article');
            item.className = 'history-item';
            if (session.session_id === activeSessionId) {
                item.classList.add('active');
            }

            const preview = session.preview || 'No preview available.';
            const title = this.buildSessionTitle(session);
            const updatedAt = this.formatTimestamp(session.updated_at);
            const itemMeta = `${session.message_count} messages • ${updatedAt}`;

            item.innerHTML = `
                <div class="history-item-header">
                    <div>
                        <div class="history-item-title">${this.escapeHtml(title)}</div>
                        <div class="history-item-meta">${this.escapeHtml(itemMeta)}</div>
                    </div>
                </div>
                <div class="history-item-preview">${this.escapeHtml(preview)}</div>
                <div class="history-item-actions">
                    <button class="history-action-btn" type="button" data-history-action="open" data-session-id="${this.escapeHtml(session.session_id)}">Open</button>
                    <button class="history-action-btn delete" type="button" data-history-action="delete" data-session-id="${this.escapeHtml(session.session_id)}">Delete</button>
                </div>
            `;

            this.historyList.appendChild(item);
        }
    }

    buildSessionTitle(session) {
        if (session.preview) {
            return session.preview.slice(0, 48);
        }

        return `Session ${session.session_id}`;
    }

    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        if (Number.isNaN(date.getTime())) {
            return timestamp;
        }

        return date.toLocaleString();
    }

    escapeHtml(value) {
        return String(value)
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#39;');
    }

    onHistoryRefresh(callback) {
        this.onHistoryRefreshHandler = callback;
    }

    onHistoryOpen(callback) {
        this.onHistoryOpenHandler = callback;
    }

    onHistoryDelete(callback) {
        this.onHistoryDeleteHandler = callback;
    }

    onHistoryNewChat(callback) {
        this.onHistoryNewChatHandler = callback;
    }

    onVolumeChange(callback) {
        this.onVolumeChangeHandler = callback;
        callback(this.getPlaybackVolume());
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
                
                const sendOptions = {
                    reasoning: this.reasoningEnabledForNextSend,
                };

                callback(text, sendOptions);

                this.reasoningEnabledForNextSend = false;
                this.syncReasoningToggle();
                this.closeComposerMenu();
                
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
