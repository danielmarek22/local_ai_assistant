import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
    CLIENT_FRAME_TYPES,
    SERVER_FRAME_TYPES,
} from '../static/js/network-client.js';


test('frontend websocket discriminators match the protocol manifest', async () => {
    const manifestUrl = new URL('../static/websocket-protocol.json', import.meta.url);
    const manifest = JSON.parse(await readFile(manifestUrl, 'utf8'));

    assert.equal(manifest.version, 1);
    assert.deepEqual([...CLIENT_FRAME_TYPES].sort(), manifest.client_frame_types);
    assert.deepEqual([...SERVER_FRAME_TYPES].sort(), manifest.server_frame_types);
});
