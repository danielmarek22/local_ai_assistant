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


test('oversized websocket closure reports the server reason before reconnecting', () => {
    let notice = null;
    let reconnectScheduled = false;
    const client = new NetworkClient({
        onUserNotice: (payload) => { notice = payload; },
    });
    client.scheduleReconnect = () => { reconnectScheduled = true; };
    client.connect();

    client.ws.onclose({ code: 1009, reason: 'Binary frame exceeds the limit' });

    assert.deepEqual(notice, {
        scope: 'last_user_message',
        tone: 'warning',
        message: 'Binary frame exceeds the limit',
    });
    assert.equal(reconnectScheduled, true);
});


test('undeclared server frames are ignored', () => {
    let state = null;
    const client = new NetworkClient({ onState: (value) => { state = value; } });
    client.connect();

    client.ws.onmessage({ data: JSON.stringify({
        type: 'assistant_state_v2',
        state: 'thinking',
    }) });

    assert.equal(state, null);
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


test('outfit catalog and change events are dispatched', () => {
    let initialized = null;
    let changed = null;
    const client = new NetworkClient({
        onSessionInit: (payload) => { initialized = payload; },
        onOutfit: (payload) => { changed = payload; },
    });
    client.connect();

    client.ws.onmessage({ data: JSON.stringify({
        type: 'session_init',
        server_instance_id: 'server-1',
        session_id: 'session-1',
        outfit_catalog: { pajamas: '/static/avatars/pajamas.vrm' },
        current_outfit: 'pajamas',
    }) });
    assert.deepEqual(initialized.outfitCatalog, {
        pajamas: '/static/avatars/pajamas.vrm',
    });
    assert.equal(initialized.currentOutfit, 'pajamas');

    client.ws.onmessage({ data: JSON.stringify({
        type: 'assistant_outfit',
        outfit: 'pajamas',
        url: '/static/avatars/pajamas.vrm',
    }) });
    assert.deepEqual(changed, {
        outfit: 'pajamas',
        url: '/static/avatars/pajamas.vrm',
    });
});


test('state and animation handlers receive turn IDs', () => {
    let stateEvent = null;
    let animationEvent = null;
    const client = new NetworkClient({
        onState: (state, turnId) => { stateEvent = { state, turnId }; },
        onAnimation: (animation, turnId) => { animationEvent = { animation, turnId }; },
    });
    client.connect();

    client.ws.onmessage({ data: JSON.stringify({
        type: 'assistant_state',
        state: 'thinking',
        turn_id: 'turn-7',
    }) });
    client.ws.onmessage({ data: JSON.stringify({
        type: 'assistant_animation',
        animation: 'wave',
        turn_id: 'turn-7',
    }) });

    assert.deepEqual(stateEvent, { state: 'thinking', turnId: 'turn-7' });
    assert.deepEqual(animationEvent, { animation: 'wave', turnId: 'turn-7' });
});
