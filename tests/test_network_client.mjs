import test from 'node:test';
import assert from 'node:assert/strict';

import { NetworkClient } from '../static/js/network-client.js';


class FakeWebSocket {
    static OPEN = 1;

    constructor(url) {
        this.url = url;
        this.readyState = FakeWebSocket.OPEN;
        this.sent = [];
    }

    send(value) { this.sent.push(value); }
    close() { this.readyState = 3; }
}


globalThis.WebSocket = FakeWebSocket;
globalThis.window = { location: { href: 'http://localhost:8000/' } };


test('session kind is sent on connection and restored from session init', () => {
    let initialized = null;
    const client = new NetworkClient({ onSessionInit: (payload) => { initialized = payload; } });
    client.connect({ sessionMode: 'new', sessionKind: 'manual_group' });
    assert.equal(new URL(client.ws.url).searchParams.get('session_kind'), 'manual_group');

    client.ws.onmessage({ data: JSON.stringify({
        type: 'session_init',
        server_instance_id: 'server-1',
        session_id: 'session-1',
        session_kind: 'manual_group',
        local_human_display_name: 'Local Person',
        local_assistant_display_name: 'Astra Custom',
    }) });
    assert.equal(initialized.sessionKind, 'manual_group');
    assert.equal(initialized.localHumanDisplayName, 'Local Person');
    assert.equal(initialized.localAssistantDisplayName, 'Astra Custom');
});


test('relay payload contains only the accepted v1 fields', () => {
    const client = new NetworkClient({});
    client.ws = new FakeWebSocket('ws://localhost/ws');

    assert.equal(client.sendRelayMessage('Claude', 'external_agent', 'Hello'), true);
    assert.deepEqual(JSON.parse(client.ws.sent[0]), {
        type: 'relay_message',
        sender_display_name: 'Claude',
        sender_type: 'external_agent',
        text: 'Hello',
    });
});
