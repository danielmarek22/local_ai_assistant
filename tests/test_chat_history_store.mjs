import test from 'node:test';
import assert from 'node:assert/strict';
import { ChatHistoryStore } from '../static/js/chat-history-store.mjs';
import { CONFIG } from '../static/js/config.js';

function fixture() {
    const saved = new Map();
    const store = new ChatHistoryStore(values => values, () => ({
        getItem: key => saved.get(key) ?? null,
        setItem: (key, value) => saved.set(key, value),
        removeItem: key => saved.delete(key),
    }));
    return { saved, store };
}

test('history scopes preserve the existing key and recover missing or invalid snapshots', t => {
    t.mock.method(console, 'warn', () => {});
    for (const raw of [null, 'invalid json', '{}', '[]']) {
        const { saved, store } = fixture();
        const defaults = [{ sender: 'system', text: 'Ready' }];
        assert.equal(store.setScope('server', 'session'), true);
        assert.equal(store.setScope('server', 'session'), false);
        const key = `${CONFIG.UI.STORAGE_KEYS.CHAT_HISTORY}:server:session`;
        if (raw !== null) saved.set(key, raw);
        let rendered;
        store.restore(defaults, messages => { rendered = messages; }, () => rendered);
        assert.deepEqual(rendered, defaults);
        assert.deepEqual(JSON.parse(saved.get(key)), defaults);
        store.setScope('server', 'other');
        store.persist(() => [{ sender: 'user', text: 'Other' }]);
        assert.deepEqual(JSON.parse(saved.get(key)), defaults);
        assert.equal(saved.size, 2);
    }
});

test('snapshot round trip preserves sender, attachments, pending IDs and retry information', () => {
    const { store } = fixture();
    const attachments = [{ name: 'image', mimeType: 'image/png', data: 'AA==', url: null, size: 1 }];
    const elements = [{
        classList: new Set(['message', 'user', 'sender-human']),
        dataset: {
            rawText: 'Hello', attachments: JSON.stringify(attachments), senderId: 'human-a',
            senderDisplayName: 'Alice', senderType: 'human', inputSource: 'manual_relay',
            messageId: '42', retryError: 'Try again', retryAttempts: '2',
        },
    }, { classList: new Set(['message', 'user']), dataset: { rawText: 'Pending', awaitingMessageId: 'true' } }];
    const snapshot = store.serialize({ querySelectorAll: () => elements });
    assert.deepEqual(snapshot[0], {
        sender: 'user', text: 'Hello', attachments, senderId: 'human-a', senderDisplayName: 'Alice',
        senderType: 'human', inputSource: 'manual_relay', messageId: 42, awaitingMessageId: false,
        retryableFailure: { message: 'Try again', attempts: 2 },
    });
    assert.equal(snapshot[1].awaitingMessageId, true);
    assert.equal(snapshot[1].messageId, null);
    store.setScope('server', 'session');
    store.persist(() => snapshot);
    store.restore([], messages => assert.deepEqual(messages, snapshot), () => assert.fail('valid history must not be rewritten'));
});
