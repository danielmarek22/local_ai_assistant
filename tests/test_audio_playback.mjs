import test, { beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { AudioManager } from '../static/js/audio-manager.js';

const flush = () => new Promise(resolve => setImmediate(resolve));
let plays;
let playError;
let allowResume;
let resumeCalls;
let loads;

beforeEach(() => {
    plays = [];
    playError = null;
    allowResume = true;
    resumeCalls = 0;
    loads = [];
    globalThis.document = new EventTarget();
    document.visibilityState = 'visible';
    globalThis.window = new EventTarget();
    window.setTimeout = setTimeout;
    window.clearTimeout = clearTimeout;
    globalThis.Audio = class {
        constructor() {
            this.paused = true;
            this.ended = false;
            this.currentSrc = '';
        }
        play() {
            plays.push(this.src);
            if (playError) return Promise.reject(playError);
            this.currentSrc = this.src;
            this.paused = false;
            this.ended = false;
            this.onplaying?.();
            return Promise.resolve();
        }
        load() { loads.push(this.src); }
    };
    window.AudioContext = class {
        constructor() {
            this.state = 'suspended';
            this.destination = {};
        }
        createAnalyser() { return { frequencyBinCount: 128, connect() {} }; }
        createMediaElementSource() { return { connect() {} }; }
        resume() {
            resumeCalls++;
            if (!allowResume) return Promise.reject(new DOMException('Blocked', 'NotAllowedError'));
            this.state = 'running';
            return Promise.resolve();
        }
    };
});

afterEach(() => {
    delete globalThis.document;
    delete globalThis.window;
    delete globalThis.Audio;
});

test('a voice-mode interaction unlocks a newly created playback context', async () => {
    const audio = new AudioManager();
    document.dispatchEvent(new Event('pointerdown'));
    await flush();
    assert.equal(audio.audioContext.state, 'running');
    assert.equal(resumeCalls, 1);
});

test('a suspended context is resumed before the next speech clip starts', async () => {
    const audio = new AudioManager();
    audio.init();
    await flush();
    audio.audioContext.state = 'suspended';
    document.visibilityState = 'hidden';
    audio.queueAudio('background.wav');
    await flush();
    assert.equal(resumeCalls, 2);
    assert.equal(audio.audioContext.state, 'running');
    assert.deepEqual(plays, ['background.wav']);
});

test('autoplay rejection retains speech order and retries after activation', async () => {
    const audio = new AudioManager();
    playError = new DOMException('Blocked', 'NotAllowedError');
    audio.queueAudio('first.wav');
    audio.queueAudio('second.wav');
    await flush();
    audio.queueAudio('third.wav');
    await flush();
    assert.deepEqual(plays, ['first.wav']);
    assert.deepEqual(audio.audioQueue, ['first.wav', 'second.wav', 'third.wav']);
    assert.equal(audio.isPlaying, false);

    playError = null;
    document.dispatchEvent(new Event('keydown'));
    await flush();
    assert.deepEqual(plays, ['first.wav', 'first.wav']);
    audio.audioEl.onended();
    await flush();
    audio.audioEl.onended();
    await flush();
    assert.deepEqual(plays, ['first.wav', 'first.wav', 'second.wav', 'third.wav']);
    assert.deepEqual(audio.audioQueue, []);
});

test('failed context recovery preserves the clip without playing it silently', async () => {
    const audio = new AudioManager();
    allowResume = false;
    audio.init();
    audio.queueAudio('speech.wav');
    await flush();
    assert.deepEqual(plays, []);
    assert.deepEqual(audio.audioQueue, ['speech.wav']);

    allowResume = true;
    window.dispatchEvent(new Event('focus'));
    await flush();
    assert.deepEqual(plays, ['speech.wav']);
    assert.equal(audio.playbackBlocked, false);
});

test('returning to the tab retries a blocked clip without duplicating playback', async () => {
    const audio = new AudioManager();
    playError = new DOMException('Interrupted', 'AbortError');
    audio.queueAudio('speech.wav');
    await flush();
    playError = null;
    document.dispatchEvent(new Event('visibilitychange'));
    window.dispatchEvent(new Event('focus'));
    window.dispatchEvent(new Event('pageshow'));
    await flush();
    assert.deepEqual(plays, ['speech.wav', 'speech.wav']);
});

test('an unsupported clip does not prevent later speech from playing', async () => {
    const audio = new AudioManager();
    let attempts = 0;
    audio.audioEl.play = () => {
        plays.push(audio.audioEl.src);
        if (++attempts === 1) return Promise.reject(new DOMException('Bad audio', 'NotSupportedError'));
        return Promise.resolve();
    };
    audio.queueAudio('bad.wav');
    audio.queueAudio('good.wav');
    await flush();
    assert.deepEqual(plays, ['bad.wav', 'good.wav']);
    assert.equal(audio.playbackBlocked, false);
});

test('a pending resume starts loading and times out without losing or wedging speech', async () => {
    const audio = new AudioManager();
    let expire;
    let completeOldResume;
    let resumeAttempts = 0;
    window.setTimeout = callback => { expire = callback; return 1; };
    window.clearTimeout = () => {};
    window.AudioContext.prototype.resume = function () {
        resumeAttempts++;
        return new Promise(resolve => { completeOldResume = resolve; });
    };
    audio.init();
    audio.queueAudio('first.wav');
    audio.queueAudio('second.wav');
    assert.deepEqual(loads, ['first.wav']);
    assert.deepEqual(plays, []);
    assert.equal(resumeAttempts, 1);
    expire();
    await flush();
    assert.equal(audio.isPlaying, false);
    assert.equal(audio.playbackBlocked, true);
    assert.deepEqual(audio.audioQueue, ['first.wav', 'second.wav']);

    audio.audioContext.resume = function () {
        this.state = 'running';
        return Promise.resolve();
    };
    window.dispatchEvent(new Event('focus'));
    await flush();
    completeOldResume();
    await flush();
    assert.deepEqual(loads, ['first.wav']);
    assert.deepEqual(plays, ['first.wav']);
    audio.audioEl.onended();
    await flush();
    assert.deepEqual(loads, ['first.wav', 'second.wav']);
    assert.deepEqual(plays, ['first.wav', 'second.wav']);
});
