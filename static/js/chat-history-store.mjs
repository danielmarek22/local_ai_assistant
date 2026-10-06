import { CONFIG } from './config.js';

// Owns the session-storage scope and snapshot format, not rendered message nodes.
export class ChatHistoryStore {
    constructor(normalizeAttachments, storage = () => sessionStorage) {
        this.normalizeAttachments = normalizeAttachments;
        this.storage = storage;
        this.storageKey = null;
    }

    setScope(serverInstanceId, sessionId) {
        const nextKey = `${CONFIG.UI.STORAGE_KEYS.CHAT_HISTORY}:${serverInstanceId}:${sessionId}`;
        if (this.storageKey === nextKey) return false;
        this.storageKey = nextKey;
        return true;
    }

    restore(defaultMessages, render, serialize) {
        if (!this.storageKey) return;

        const savedHistory = this.storage().getItem(this.storageKey);
        if (!savedHistory) {
            render(defaultMessages);
            this.persist(serialize);
            return;
        }

        try {
            const messages = JSON.parse(savedHistory);
            if (!Array.isArray(messages) || messages.length === 0) {
                render(defaultMessages);
                this.persist(serialize);
                return;
            }
            render(messages);
        } catch (error) {
            console.warn('Failed to restore chat history from session storage:', error);
            this.storage().removeItem(this.storageKey);
            render(defaultMessages);
            this.persist(serialize);
        }
    }

    persist(serialize) {
        if (!this.storageKey) return;

        try {
            this.storage().setItem(this.storageKey, JSON.stringify(serialize()));
        } catch (error) {
            console.warn('Failed to persist chat history:', error);
        }
    }

    serialize(container) {
        return Array.from(container.querySelectorAll('.message')).map((message) => {
            const sender = Array.from(message.classList).find((className) => className !== 'message') || 'astra';
            return {
                sender,
                text: message.dataset.rawText || '',
                attachments: this.readStoredAttachments(message.dataset.attachments),
                senderId: message.dataset.senderId || '',
                senderDisplayName: message.dataset.senderDisplayName || '',
                senderType: message.dataset.senderType || '',
                inputSource: message.dataset.inputSource || '',
                messageId: Number(message.dataset.messageId) || null,
                awaitingMessageId: message.dataset.awaitingMessageId === 'true',
                retryableFailure: message.dataset.retryError ? {
                    message: message.dataset.retryError,
                    attempts: Number(message.dataset.retryAttempts) || 1,
                } : null,
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
}
