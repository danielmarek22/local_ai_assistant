import { marked } from '../vendor/marked/13.0.2/lib/marked.esm.js';
import DOMPurify from '../vendor/dompurify/3.1.6/dist/purify.es.mjs';

marked.setOptions({ gfm: true, breaks: true });

// Formats individual messages; does not own streaming state or persist history.
export class MessageRenderer {
    constructor(getContext = () => ({})) {
        this.getContext = getContext;
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
            this.prependSenderHeader(msgDiv);
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
        this.prependSenderHeader(msgDiv);
    }

    setMessageMetadata(msgDiv, metadata = {}) {
        const senderType = metadata.senderType || '';
        const inputSource = metadata.inputSource || '';
        msgDiv.dataset.senderId = metadata.senderId || '';
        msgDiv.dataset.senderDisplayName = metadata.senderDisplayName || '';
        msgDiv.dataset.senderType = senderType;
        msgDiv.dataset.inputSource = inputSource;
        if (metadata.messageId) msgDiv.dataset.messageId = String(metadata.messageId);
        if (metadata.awaitingMessageId && !metadata.messageId) {
            msgDiv.dataset.awaitingMessageId = 'true';
        }
        if (metadata.retryableFailure?.message) {
            msgDiv.dataset.retryError = metadata.retryableFailure.message;
            msgDiv.dataset.retryAttempts = String(metadata.retryableFailure.attempts || 1);
        }
        const controlledTypes = ['human', 'external_agent', 'local_assistant', 'system', 'tool', 'integration_runtime'];
        const controlledSources = ['local_text', 'local_voice', 'manual_relay', 'assistant_generation', 'system_runtime', 'tool_runtime', 'integration_runtime'];
        if (controlledTypes.includes(senderType)) msgDiv.classList.add(`sender-${senderType.replaceAll('_', '-')}`);
        if (controlledSources.includes(inputSource)) msgDiv.classList.add(`source-${inputSource.replaceAll('_', '-')}`);
        if (this.getContext().conversationMode === 'manual_group') msgDiv.classList.add('group-message');
    }

    prependSenderHeader(msgDiv) {
        if (this.getContext().conversationMode !== 'manual_group') return;
        const existing = msgDiv.querySelector('.message-sender-header');
        if (existing) existing.remove();
        const header = document.createElement('div');
        header.className = 'message-sender-header';
        header.textContent = msgDiv.dataset.senderDisplayName || this.fallbackSenderLabel(msgDiv);
        msgDiv.prepend(header);
    }

    fallbackSenderLabel(msgDiv) {
        if (msgDiv.classList.contains('astra')) return this.getContext().localAssistantDisplayName;
        if (msgDiv.classList.contains('thinking')) return `${this.getContext().localAssistantDisplayName} thinking`;
        if (msgDiv.classList.contains('system')) return 'System';
        if (msgDiv.classList.contains('tool')) return 'Tool';
        return this.getContext().localHumanDisplayName;
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

    createAttachmentId() {
        if (window.crypto?.randomUUID) {
            return window.crypto.randomUUID();
        }

        return `attachment-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }
}
