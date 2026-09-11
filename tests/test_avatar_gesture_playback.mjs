import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import * as THREE from '../static/vendor/three/0.169.0/build/three.module.min.js';
import { TurnGestureQueue } from '../static/js/turn-gesture-queue.js';
import { CONFIG } from '../static/js/config.js';

// Exercise the real manager methods/callback without constructing WebGL or loaders.
const source = await readFile(new URL('../static/js/avatar-manager.js', import.meta.url), 'utf8');
const AvatarManager = new Function('THREE', 'TurnGestureQueue', 'CONFIG',
    source.replace(/^import .*;\s*$/gm, '').replace('export class AvatarManager', 'return class AvatarManager'),
)(THREE, TurnGestureQueue, CONFIG);
const marker = "this.mixer.addEventListener('finished', (event) => {";
const start = source.indexOf(marker) + marker.length;
const finished = new Function('event', source.slice(start, source.indexOf('                });', start)));

function manager(visible = true) {
    const instance = Object.create(AvatarManager.prototype);
    Object.assign(instance, {
        mixer: new THREE.AnimationMixer(new THREE.Object3D()),
        gestureQueue: new TurnGestureQueue(8), gestureCatalog: {wave: 'wave', nod: 'nod'},
        gestureAnimations: {wave: 'wave', nod: 'nod'}, stateAnimations: {idle: ['idle'], responding: ['idle']},
        animations: {}, currentState: 'idle', currentAction: null,
        isPageVisible: visible, isGesturePlaying: false, activeGestureName: null,
    });
    for (const name of ['wave', 'nod', 'idle']) {
        instance.animations[name] = instance.mixer.clipAction(new THREE.AnimationClip(name, 1, []));
    }
    instance.mixer.addEventListener('finished', finished.bind(instance));
    instance.playRandomVariant('idle');
    return instance;
}

for (const names of [['wave', 'wave'], ['wave', 'nod']]) {
    test(`hidden-tab backlog completes ${names.join(' then ')}`, () => {
        const avatar = manager(false);
        for (const name of names) avatar.queueGesture(name, 'turn');
        globalThis.document = {visibilityState: 'visible'};
        avatar.handleVisibilityChange();
        avatar.mixer.update(1.1);
        avatar.mixer.update(1.1);
        avatar.mixer.update(1.1);
        assert.equal(avatar.isGesturePlaying, false);
        assert.equal(avatar.activeGestureName, null);
        assert.equal(avatar.currentAction, avatar.animations.idle);
        assert.equal(avatar.currentAction.paused, false);
        delete globalThis.document;
    });
}

test('ordinary selection of the current looping action does not restart it', () => {
    const avatar = manager();
    avatar.mixer.update(0.4);
    avatar.fadeToAction('idle');
    assert.equal(avatar.currentAction.time, 0.4);
});

test('a completed one-shot can be replayed explicitly', () => {
    const avatar = manager();
    let completions = 0;
    avatar.mixer.addEventListener('finished', () => completions++);
    avatar.playAnimationOnce('wave', 0);
    avatar.mixer.update(1.1);
    avatar.playAnimationOnce('wave', 0);
    assert.equal(avatar.animations.wave.paused, false);
    assert.equal(avatar.animations.wave.time, 0);
    avatar.mixer.update(1.1);
    assert.equal(completions, 2);
});
