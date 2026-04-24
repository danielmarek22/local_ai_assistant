import { CONFIG } from './config.js';
import { marked } from 'https://cdn.jsdelivr.net/npm/marked@13.0.2/lib/marked.esm.js';
import DOMPurify from 'https://cdn.jsdelivr.net/npm/dompurify@3.1.6/+esm';
import { extractBase64Payload, extractImageFilesFromDataTransfer, insertTextAtCursor, isImageFile } from './attachment-utils.mjs';

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
        this.imageInput = document.getElementById('image-input');
        this.attachmentPreview = document.getElementById('attachment-preview');
        this.composerMenu = document.getElementById('composer-menu');
        this.composerMenuBtn = document.getElementById('composer-menu-btn');
        this.composerMenuPopover = document.getElementById('composer-menu-popover');
        this.imageAttachBtn = document.getElementById('image-attach-btn');
        this.reasoningToggle = document.getElementById('reasoning-toggle');
        this.playbackVolumeInput = document.getElementById('playback-volume');
        this.playbackVolumeValue = document.getElementById('playback-volume-value');
        this.reflectNowBtn = document.getElementById('reflect-now-btn');
        this.reflectStatus = document.getElementById('reflect-status');
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
        this.currentThinkingMessageDiv = null;
        this.chatHistoryStorageKey = null;
        this.currentSessionId = null;
        this.reasoningEnabledForNextSend = false;
        this.pendingAttachments = [];
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
            this.userInput.style.height = 'auto';
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

        this.imageAttachBtn?.addEventListener('click', () => {
            this.closeComposerMenu();
            this.imageInput?.click();
        });

        this.imageInput?.addEventListener('change', async (event) => {
            const files = Array.from(event.target.files || []);
            if (!files.length) return;

            await this.addPendingAttachments(files);
            event.target.value = '';
        });

        this.userInput.addEventListener('paste', (event) => {
            void this.handlePasteEvent(event);
        });

        this.attachmentPreview?.addEventListener('click', (event) => {
            const removeButton = event.target.closest('[data-attachment-remove]');
            if (!removeButton) return;

            this.removePendingAttachment(removeButton.dataset.attachmentRemove);
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

    async handlePasteEvent(event) {
        const imageFiles = extractImageFilesFromDataTransfer(event.clipboardData);
        if (!imageFiles.length) {
            return;
        }

        const pastedText = event.clipboardData?.getData('text/plain') || '';
        event.preventDefault();

        if (pastedText) {
            insertTextAtCursor(this.userInput, pastedText);
            this.userInput.dispatchEvent(new Event('input', { bubbles: true }));
        }

        await this.addPendingAttachments(imageFiles);
    }

    async addPendingAttachments(files) {
        const nextAttachments = [];

        for (const file of files) {
            if (!isImageFile(file)) {
                continue;
            }

            try {
                nextAttachments.push(await this.readImageFile(file));
            } catch (error) {
                console.warn(`Failed to read image ${file.name}:`, error);
            }
        }

        if (!nextAttachments.length) return;

        this.pendingAttachments.push(...nextAttachments);
        this.renderPendingAttachments();
    }

    readImageFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = () => {
                const result = typeof reader.result === 'string' ? reader.result : '';
                const base64Data = extractBase64Payload(result);
                if (!base64Data) {
                    reject(new Error('Image did not produce base64 data'));
                    return;
                }

                resolve({
                    id: this.createAttachmentId(),
                    name: file.name,
                    mimeType: file.type,
                    size: file.size,
                    data: base64Data,
                });
            };

            reader.onerror = () => {
                reject(reader.error || new Error('FileReader failed'));
            };

            reader.readAsDataURL(file);
        });
    }

    createAttachmentId() {
        if (window.crypto?.randomUUID) {
            return window.crypto.randomUUID();
        }

        return `attachment-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    removePendingAttachment(attachmentId) {
        if (!attachmentId) return;

        this.pendingAttachments = this.pendingAttachments.filter((attachment) => attachment.id !== attachmentId);
        this.renderPendingAttachments();
    }

    clearPendingAttachments() {
        this.pendingAttachments = [];
        this.renderPendingAttachments();
        if (this.imageInput) {
            this.imageInput.value = '';
        }
    }

    renderPendingAttachments() {
        if (!this.attachmentPreview) return;

        this.attachmentPreview.replaceChildren();
        this.attachmentPreview.classList.toggle('hidden', this.pendingAttachments.length === 0);

        for (const attachment of this.pendingAttachments) {
            const chip = this.createAttachmentNode(attachment, { removable: true });
            this.attachmentPreview.appendChild(chip);
        }
    }

    initConfigControls() {
        if (this.playbackVolumeInput && this.playbackVolumeValue) {
            this.setPlaybackVolume(this.readStoredPlaybackVolume(), { persist: false, notify: false });

            this.playbackVolumeInput.addEventListener('input', () => {
                const nextVolume = Number(this.playbackVolumeInput.value) / 100;
                this.setPlaybackVolume(nextVolume, { persist: true, notify: true });
            });
        }

        this.reflectNowBtn?.addEventListener('click', () => {
            if (this.onReflectHandler) {
                this.onReflectHandler();
            }
        });

        this.setReflectRunning(false);
        this.setReflectStatus('');
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
        this.chatPanel.classList.remove(...Object.keys(CONFIG.UI.STATUS_TEXT).map((status) => `status-${status}`));
        this.chatPanel.classList.add(`status-${state}`);

        const text = CONFIG.UI.STATUS_TEXT[state] || CONFIG.UI.STATUS_TEXT.idle;
        this.statusText.innerText = text;
    }

    appendUserMessage(text, attachments = []) {
        this.currentThinkingMessageDiv = null;
        this.currentAiMessageDiv = null;
        this.createMessageDiv('user', text, attachments);
    }

    addNoticeToLastUserMessage(text, tone = 'warning') {
        const noticeText = typeof text === 'string' ? text.trim() : '';
        if (!noticeText) return;

        const lastUserMessage = this.findLastUserMessage();
        if (!lastUserMessage) return;

        let notice = lastUserMessage.querySelector('.message-notice');
        if (!notice) {
            notice = document.createElement('div');
            notice.className = 'message-notice';
            lastUserMessage.prepend(notice);
        }

        notice.textContent = noticeText;
        notice.classList.toggle('warning', tone === 'warning');
        notice.classList.toggle('info', tone !== 'warning');

        this.persistChatHistory();
        this.scrollToBottom();
    }

    findLastUserMessage() {
        const messages = this.chatHistory.querySelectorAll('.message.user');
        if (!messages.length) return null;
        return messages[messages.length - 1];
    }

    startAiMessage() {
        if (!this.currentAiMessageDiv) {
            this.currentAiMessageDiv = this.createMessageDiv('astra', '');
        }
    }

    startThinkingMessage() {
        if (!this.currentThinkingMessageDiv) {
            this.currentThinkingMessageDiv = this.createMessageDiv('thinking', '');
        }
    }

    appendToAiMessage(text) {
        if (!this.currentAiMessageDiv) this.startAiMessage();
        const rawText = (this.currentAiMessageDiv.dataset.rawText || '') + text;
        this.setMessageContent(this.currentAiMessageDiv, rawText);
        this.persistChatHistory();
        this.scrollToBottom();
    }

    appendToThinkingMessage(text) {
        if (!text) return;

        if (!this.currentThinkingMessageDiv) this.startThinkingMessage();
        const rawText = (this.currentThinkingMessageDiv.dataset.rawText || '') + text;
        this.setMessageContent(this.currentThinkingMessageDiv, rawText);
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

    finalizeThinkingMessage() {
        this.currentThinkingMessageDiv = null;
    }

    createMessageDiv(sender, text, attachments = []) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', sender);
        this.setMessageContent(msgDiv, text, attachments);
        this.chatHistory.appendChild(msgDiv);
        this.persistChatHistory();
        this.scrollToBottom();
        return msgDiv;
    }

    setMessageContent(msgDiv, text, attachments = []) {
        const normalizedAttachments = this.normalizeAttachments(attachments);
        msgDiv.dataset.rawText = text;

        if (normalizedAttachments.length) {
            msgDiv.dataset.attachments = JSON.stringify(normalizedAttachments.map((attachment) => ({
                id: attachment.id,
                name: attachment.name,
                mimeType: attachment.mimeType,
                data: attachment.data,
                url: attachment.url,
                size: attachment.size,
            })));
        } else {
            delete msgDiv.dataset.attachments;
        }

        if (msgDiv.classList.contains('astra') || msgDiv.classList.contains('thinking')) {
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

        msgDiv.replaceChildren();

        if (text) {
            const body = document.createElement('div');
            body.className = 'message-body';
            body.innerText = text;
            msgDiv.appendChild(body);
        }

        if (normalizedAttachments.length) {
            const gallery = document.createElement('div');
            gallery.className = 'message-attachments';

            for (const attachment of normalizedAttachments) {
                gallery.appendChild(this.createAttachmentNode(attachment));
            }

            msgDiv.appendChild(gallery);
        }
    }

    normalizeAttachments(attachments) {
        if (!Array.isArray(attachments)) {
            return [];
        }

        return attachments
            .map((attachment) => this.normalizeAttachment(attachment))
            .filter(Boolean);
    }

    normalizeAttachment(attachment) {
        if (!attachment || typeof attachment !== 'object') {
            return null;
        }

        const data = typeof attachment.data === 'string' ? attachment.data.trim() : '';
        const url = typeof attachment.url === 'string' ? attachment.url.trim() : '';
        if (!data && !url) {
            return null;
        }

        const mimeType = typeof attachment.mimeType === 'string'
            ? attachment.mimeType
            : typeof attachment.mime_type === 'string'
                ? attachment.mime_type
                : 'image/png';
        if (!mimeType.startsWith('image/')) {
            return null;
        }

        const size = Number.isFinite(attachment.size)
            ? attachment.size
            : Number.isFinite(attachment.size_bytes)
                ? attachment.size_bytes
                : null;

        return {
            id: typeof attachment.id === 'string' && attachment.id ? attachment.id : this.createAttachmentId(),
            name: typeof attachment.name === 'string' && attachment.name.trim() ? attachment.name.trim() : 'image',
            mimeType,
            data: data || null,
            url: url || null,
            size,
        };
    }

    createAttachmentNode(attachment, { removable = false } = {}) {
        const node = document.createElement('figure');
        node.className = removable ? 'attachment-chip' : 'message-attachment';

        const image = document.createElement('img');
        image.className = removable ? 'attachment-chip-image' : 'message-attachment-image';
        image.src = this.buildAttachmentSrc(attachment);
        image.alt = attachment.name;
        node.appendChild(image);

        const caption = document.createElement('figcaption');
        caption.className = removable ? 'attachment-chip-meta' : 'message-attachment-meta';
        caption.textContent = this.formatAttachmentLabel(attachment.name, removable ? 18 : 28);
        node.appendChild(caption);

        if (removable) {
            const removeButton = document.createElement('button');
            removeButton.type = 'button';
            removeButton.className = 'attachment-chip-remove';
            removeButton.dataset.attachmentRemove = attachment.id;
            removeButton.setAttribute('aria-label', `Remove ${attachment.name}`);
            removeButton.textContent = 'x';
            node.appendChild(removeButton);
        }

        return node;
    }

    buildAttachmentSrc(attachment) {
        if (attachment.url) {
            return attachment.url;
        }

        return `data:${attachment.mimeType};base64,${attachment.data}`;
    }

    formatAttachmentLabel(name, limit = 24) {
        if (name.length <= limit) {
            return name;
        }

        return `${name.slice(0, Math.max(0, limit - 1))}…`;
    }

    setSessionScope(serverInstanceId, sessionId) {
        const nextStorageKey = `${CONFIG.UI.STORAGE_KEYS.CHAT_HISTORY}:${serverInstanceId}:${sessionId}`;
        if (this.chatHistoryStorageKey === nextStorageKey) return;

        this.currentSessionId = sessionId;
        this.chatHistoryStorageKey = nextStorageKey;
        this.currentAiMessageDiv = null;
        this.currentThinkingMessageDiv = null;
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

        try {
            sessionStorage.setItem(this.chatHistoryStorageKey, JSON.stringify(this.serializeChatHistory()));
        } catch (error) {
            console.warn('Failed to persist chat history:', error);
        }
    }

    serializeChatHistory() {
        return Array.from(this.chatHistory.querySelectorAll('.message')).map((message) => {
            const sender = Array.from(message.classList).find((className) => className !== 'message') || 'astra';
            return {
                sender,
                text: message.dataset.rawText || '',
                attachments: this.readStoredAttachments(message.dataset.attachments),
            };
        });
    }

    readStoredAttachments(rawValue) {
        if (!rawValue) {
            return [];
        }

        try {
            const parsed = JSON.parse(rawValue);
            return this.normalizeAttachments(parsed).map((attachment) => ({
                name: attachment.name,
                mimeType: attachment.mimeType,
                data: attachment.data,
                url: attachment.url,
                size: attachment.size,
            }));
        } catch (error) {
            console.warn('Failed to parse stored attachments:', error);
            return [];
        }
    }

    renderMessages(messages) {
        this.chatHistory.replaceChildren();

        for (const message of messages) {
            if (!message || typeof message.sender !== 'string' || typeof message.text !== 'string') {
                continue;
            }

            const msgDiv = document.createElement('div');
            msgDiv.classList.add('message', message.sender);
            this.setMessageContent(msgDiv, message.text, message.attachments || []);
            this.chatHistory.appendChild(msgDiv);
        }

        this.scrollToBottom();
    }

    renderSessionMessages(sessionData) {
        const messages = sessionData.messages.map((message) => ({
            sender: message.role === 'assistant' ? 'astra' : message.role,
            text: message.content,
            attachments: message.attachments || [],
        }));

        this.currentAiMessageDiv = null;
        this.currentThinkingMessageDiv = null;
        this.renderMessages(messages);
        this.persistChatHistory();
    }

    resetChatToDefault() {
        this.currentAiMessageDiv = null;
        this.currentThinkingMessageDiv = null;
        this.clearPendingAttachments();
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

    setReflectRunning(isRunning) {
        if (!this.reflectNowBtn) return;

        this.reflectNowBtn.disabled = isRunning;
        this.reflectNowBtn.textContent = isRunning ? 'Dreaming...' : 'Run Reflection';
    }

    setReflectStatus(message = '', tone = 'info') {
        if (!this.reflectStatus) return;

        const hasMessage = Boolean(message);
        this.reflectStatus.textContent = message;
        this.reflectStatus.classList.toggle('hidden', !hasMessage);
        this.reflectStatus.classList.toggle('success', hasMessage && tone === 'success');
        this.reflectStatus.classList.toggle('error', hasMessage && tone === 'error');
    }

    onReflect(callback) {
        this.onReflectHandler = callback;
    }

    scrollToBottom() {
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
    }

    onSend(callback) {
        const handler = (event) => {
            if (event && event.type === 'keypress') {
                event.preventDefault();
            }

            const text = this.userInput.value.trim();
            const attachments = this.pendingAttachments.map((attachment) => ({
                name: attachment.name,
                mimeType: attachment.mimeType,
                data: attachment.data,
                url: attachment.url,
                size: attachment.size,
            }));
            if (!text && attachments.length === 0) return;

            const sendOptions = {
                reasoning: this.reasoningEnabledForNextSend,
                attachments,
            };

            callback(text, sendOptions);

            this.reasoningEnabledForNextSend = false;
            this.syncReasoningToggle();
            this.closeComposerMenu();
            this.clearPendingAttachments();

            this.userInput.value = '';
            this.userInput.style.height = 'auto';
            this.userInput.focus();
        };

        this.sendBtn.addEventListener('click', () => handler());

        this.userInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                handler(event);
            }
        });
    }
}
