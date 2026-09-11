import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

// Resolve browser import URLs to local vendored modules without constructing the UI.
const sourceUrl = new URL('../static/js/ui-manager.js', import.meta.url);
const source = (await readFile(sourceUrl, 'utf8')).replace(
    /from '([^']+)'/g,
    (_, specifier) => `from '${specifier.startsWith('/static/')
        ? new URL(`..${specifier}`, import.meta.url).href
        : new URL(specifier, sourceUrl).href}'`,
);
const { UIManager } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

function message() {
    return { dataset: {}, classList: new Set(['message', 'user']) };
}

test('pending identity survives serialization and refresh, and replay is idempotent', () => {
    const ui = Object.create(UIManager.prototype);
    const original = message();
    ui.setMessageMetadata(original, { awaitingMessageId: true });
    ui.chatHistory = { querySelectorAll: () => [original] };
    ui.readStoredAttachments = () => [];
    const saved = ui.serializeChatHistory()[0];
    assert.equal(saved.awaitingMessageId, true);
    const restored = message();
    ui.setMessageMetadata(restored, saved);
    const next = message();
    ui.setMessageMetadata(next, { awaitingMessageId: true });
    const messages = [restored, next];
    ui.chatHistory = {
        querySelectorAll: () => messages,
        querySelector: () => messages.find(item => item.dataset.awaitingMessageId === 'true'),
    };
    let saves = 0;
    ui.persistChatHistory = () => saves++;
    ui.acknowledgeUserMessage(42);
    ui.acknowledgeUserMessage(42);
    assert.equal(restored.dataset.messageId, '42');
    assert.equal(restored.dataset.awaitingMessageId, undefined);
    assert.equal(next.dataset.messageId, undefined);
    assert.equal(saves, 1);
});

test('text, voice and relay mark pending identity before initial persistence', () => {
    const ui = Object.create(UIManager.prototype);
    ui.createMessageDiv = (sender, text, attachments, metadata) => {
        assert.equal(metadata.awaitingMessageId, true);
        return message();
    };
    ui.appendUserMessage('Text');
    ui.appendVoiceUserMessage('Voice');
    ui.appendRelayMessage('Relay', 'Person', 'human');
});
