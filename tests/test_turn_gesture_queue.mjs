import test from 'node:test';
import assert from 'node:assert/strict';

import { TurnGestureQueue } from '../static/js/turn-gesture-queue.js';


test('a newer turn discards deferred gestures from older turns', () => {
    const queue = new TurnGestureQueue(3);
    queue.push('wave', 'turn-1');
    queue.push('nod', 'turn-1');
    queue.setTurn('turn-2');
    queue.push('cheer', 'turn-2');

    assert.equal(queue.peek(), 'cheer');
    assert.equal(queue.shift(), 'cheer');
    assert.equal(queue.peek(), null);
});


test('the gesture backlog is bounded to the newest entries', () => {
    const queue = new TurnGestureQueue(2);
    queue.push('one', 'turn-1');
    queue.push('two', 'turn-1');
    queue.push('three', 'turn-1');

    assert.equal(queue.shift(), 'two');
    assert.equal(queue.shift(), 'three');
});
