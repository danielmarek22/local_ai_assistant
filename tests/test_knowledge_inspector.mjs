import test from 'node:test';
import assert from 'node:assert/strict';

import {
    KnowledgeInspector,
    SAVED_MEMORIES_EXPLANATION,
    pageWindow,
} from '../static/js/knowledge-inspector.js';
import { NetworkClient, buildKnowledgeBeliefsUrl } from '../static/js/network-client.js';


test('belief query uses exact URL encoding', () => {
    const url = buildKnowledgeBeliefsUrl({
        subject_id: 'person/a & b',
        predicate: 'current state',
        limit: 25,
        offset: 0,
        visibility: '',
    });
    const parsed = new URL(url, 'http://localhost');
    assert.equal(parsed.searchParams.get('subject_id'), 'person/a & b');
    assert.equal(parsed.searchParams.get('predicate'), 'current state');
    assert.equal(parsed.searchParams.get('limit'), '25');
    assert.equal(parsed.searchParams.get('offset'), '0');
    assert.equal(parsed.searchParams.has('visibility'), false);
});


test('knowledge paths URL-encode belief and session IDs', async () => {
    const urls = [];
    globalThis.fetch = async (url) => {
        urls.push(String(url));
        return { ok: true, json: async () => ({}) };
    };
    const client = new NetworkClient({});
    await client.getBelief('belief/a b');
    await client.getEffectiveBeliefs('session/a & b');
    await client.getBeliefContext('session/a & b');
    assert.equal(urls[0], '/api/knowledge/beliefs/belief%2Fa%20b');
    assert.equal(new URL(urls[1], 'http://localhost').searchParams.get('session_id'), 'session/a & b');
    assert.equal(new URL(urls[2], 'http://localhost').searchParams.get('session_id'), 'session/a & b');
});


test('saved memories endpoint is requested directly', async () => {
    const urls = [];
    globalThis.fetch = async (url) => {
        urls.push(String(url));
        return { ok: true, json: async () => ({ records: [], total: 0 }) };
    };
    const client = new NetworkClient({});
    const payload = await client.getSavedMemories();
    assert.equal(urls[0], '/api/knowledge/memories');
    assert.deepEqual(payload, { records: [], total: 0 });
});


test('pagination window reports bounded positions and navigation', () => {
    assert.deepEqual(pageWindow(0, 50, 0), {
        start: 0, end: 0, hasPrevious: false, hasNext: false,
    });
    assert.deepEqual(pageWindow(121, 50, 50), {
        start: 51, end: 100, hasPrevious: true, hasNext: true,
    });
    assert.deepEqual(pageWindow(121, 50, 100), {
        start: 101, end: 121, hasPrevious: true, hasNext: false,
    });
});


test('fetch errors retain HTTP status for safe component handling', async () => {
    globalThis.fetch = async () => ({ ok: false, status: 404 });
    const client = new NetworkClient({});
    await assert.rejects(
        client.getBelief('gone'),
        (error) => error.status === 404 && error.message === 'Request failed (404)',
    );
});


test('inspection session selection never switches the chat session', async () => {
    let switched = false;
    const inspector = Object.create(KnowledgeInspector.prototype);
    inspector.client = { switchSession: () => { switched = true; } };
    inspector.closeDetail = () => {};
    inspector.refresh = async () => {};
    await inspector.selectInspectionSession('historical/session');
    assert.equal(inspector.selectedSessionId, 'historical/session');
    assert.equal(switched, false);
});


test('saved-memory refresh does not load or switch sessions', async () => {
    let loadedMemories = 0;
    let loadedSessions = 0;
    let switched = false;
    const inspector = Object.create(KnowledgeInspector.prototype);
    inspector.view = 'memories';
    inspector.client = { switchSession: () => { switched = true; } };
    inspector.loadMemories = async () => { loadedMemories += 1; };
    inspector.loadSessions = async () => { loadedSessions += 1; };
    await inspector.refresh(true);
    assert.equal(loadedMemories, 1);
    assert.equal(loadedSessions, 0);
    assert.equal(switched, false);
});


test('authoritative session changes follow active inspection but preserve manual selection', async () => {
    const inspector = Object.create(KnowledgeInspector.prototype);
    inspector.loadedSessions = false;
    inspector.activeSessionId = 'active-a';
    inspector.selectedSessionId = 'active-a';
    await inspector.setActiveSession('active-b');
    assert.equal(inspector.selectedSessionId, 'active-b');

    inspector.selectedSessionId = 'historical';
    await inspector.setActiveSession('active-c');
    assert.equal(inspector.selectedSessionId, 'historical');
});


test('hostile stored values are assigned as literal table text', () => {
    class FakeElement {
        constructor(tag) {
            this.tag = tag;
            this.children = [];
            this.textContent = '';
            this.className = '';
        }
        set innerHTML(_value) { throw new Error('innerHTML must not be used'); }
        append(...children) { this.children.push(...children); }
        appendChild(child) { this.children.push(child); return child; }
        replaceChildren(...children) { this.children = children; }
        addEventListener() {}
    }
    globalThis.document = { createElement: (tag) => new FakeElement(tag) };
    const inspector = Object.create(KnowledgeInspector.prototype);
    inspector.openDetail = async () => {};
    const container = new FakeElement('div');
    const hostile = '<img src=x onerror="alert(1)">';
    inspector.renderBeliefTable(container, [{
        belief_id: 'belief<script>',
        record_status: 'active',
        subject: { display_name_at_evidence_time: hostile, id: 'subject<&>' },
        predicate: 'predicate</td><script>',
        value_json: JSON.stringify({ payload: hostile }),
        epistemic_status: 'SELF_REPORT',
        source: { display_name_at_evidence_time: 'source<script>', sender_id: 'source<&>' },
        visibility: 'AGENT_CURRENT',
        expires_at: null,
        revision: 1,
        updated_at: '2026-08-26T00:00:00+00:00',
    }]);
    const texts = [];
    const visit = (node) => {
        if (node.textContent) texts.push(node.textContent);
        node.children.forEach(visit);
    };
    visit(container);
    assert.ok(texts.includes(`${hostile} (subject<&>)`));
    assert.ok(texts.includes('predicate</td><script>'));
    assert.ok(texts.includes(JSON.stringify({ payload: hostile })));
});


test('saved memory loading, empty, success, and failure states are safe', async () => {
    const buildInspector = (getSavedMemories) => {
        const states = [];
        const statuses = [];
        const rendered = [];
        const inspector = Object.create(KnowledgeInspector.prototype);
        inspector.client = { getSavedMemories };
        inspector.memoryCount = { textContent: '' };
        inspector.memoryTable = { replaceChildren: () => { rendered.push([]); } };
        inspector.setLoading = (value) => states.push(value);
        inspector.setStatus = (message, tone = 'info') => statuses.push([message, tone]);
        inspector.renderSavedMemories = (records) => rendered.push(records);
        return { inspector, states, statuses, rendered };
    };

    const success = buildInspector(async () => ({
        records: [{ id: 'm1', content: 'literal' }], total: 1,
    }));
    await success.inspector.loadMemories();
    assert.deepEqual(success.states, [true, false]);
    assert.equal(success.inspector.memoryCount.textContent, '1');
    assert.deepEqual(success.rendered.at(-1), [{ id: 'm1', content: 'literal' }]);
    assert.deepEqual(success.statuses.at(-1), ['', 'info']);

    const empty = buildInspector(async () => ({ records: [], total: 0 }));
    await empty.inspector.loadMemories();
    assert.equal(empty.inspector.memoryCount.textContent, '0');
    assert.deepEqual(empty.statuses.at(-1), ['No saved memories.', 'info']);

    const failed = buildInspector(async () => { throw new Error('secret backend detail'); });
    await failed.inspector.loadMemories();
    assert.equal(failed.inspector.memoryCount.textContent, 'Unavailable');
    assert.deepEqual(failed.statuses.at(-1), ['Failed to load saved memories.', 'error']);
    assert.equal(failed.statuses.flat().includes('secret backend detail'), false);
});


test('hostile saved memory content remains literal text', () => {
    class FakeElement {
        constructor(tag) {
            this.tag = tag;
            this.children = [];
            this.textContent = '';
            this.className = '';
        }
        set innerHTML(_value) { throw new Error('innerHTML must not be used'); }
        append(...children) { this.children.push(...children); }
        appendChild(child) { this.children.push(child); return child; }
        replaceChildren(...children) { this.children = children; }
    }
    globalThis.document = { createElement: (tag) => new FakeElement(tag) };
    const inspector = Object.create(KnowledgeInspector.prototype);
    inspector.memoryTable = new FakeElement('div');
    const hostile = '<svg onload="alert(1)"></svg>';
    inspector.renderSavedMemories([{
        id: 'memory<&>',
        category: 'category<script>',
        content: hostile,
        importance: 2,
        created_at: '2026-08-26T09:00:00',
        last_accessed_at: '2026-08-26T10:00:00',
    }]);
    const texts = [];
    const visit = (node) => {
        if (node.textContent) texts.push(node.textContent);
        node.children.forEach(visit);
    };
    visit(inspector.memoryTable);
    assert.ok(texts.includes(hostile));
    assert.ok(texts.includes('category<script>'));
});


test('global saved-memory explanation states scope, deletion, and provenance limits', () => {
    assert.match(SAVED_MEMORIES_EXPLANATION, /global structured records/);
    assert.match(SAVED_MEMORIES_EXPLANATION, /survive individual chat-session deletion/);
    assert.match(SAVED_MEMORIES_EXPLANATION, /does not store source session or message provenance/);
});
