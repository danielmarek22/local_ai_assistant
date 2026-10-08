import test from 'node:test';
import assert from 'node:assert/strict';
import { MicrophoneRecorder } from '../static/js/microphone-recorder.mjs';

function recordingEnvironment(t) {
    const streams = [];
    const recorders = [];
    const timers = new Set();
    let energy = 0;
    const analyser = {
        frequencyBinCount: 2,
        getByteFrequencyData: data => data.fill(energy),
    };
    class Recorder {
        static isTypeSupported() { return true; }
        constructor(stream, options) {
            this.stream = stream;
            this.mimeType = options.mimeType;
            this.state = 'inactive';
            recorders.push(this);
        }
        start() { this.state = 'recording'; }
        stop() {
            assert.equal(this.state, 'recording');
            this.state = 'inactive';
            queueMicrotask(() => {
                this.ondataavailable({ data: new Blob(['voice']) });
                this.onstop();
            });
        }
    }
    const mediaDevices = {
        async getUserMedia() {
            const track = { stops: 0, stop() { this.stops++; } };
            const stream = { getTracks: () => [track], track };
            streams.push(stream);
            return stream;
        },
    };
    const globals = {
        navigator: { mediaDevices },
        MediaRecorder: Recorder,
        window: { AudioContext: class {
            createMediaStreamSource() { return { connect() {} }; }
            createAnalyser() { return analyser; }
        } },
        setInterval: fn => { timers.add(fn); return fn; },
        clearInterval: fn => timers.delete(fn),
    };
    for (const [key, value] of Object.entries(globals)) {
        const previous = Object.getOwnPropertyDescriptor(globalThis, key);
        Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
        t.after(() => previous
            ? Object.defineProperty(globalThis, key, previous)
            : delete globalThis[key]);
    }
    return { streams, recorders, timers, mediaDevices, setEnergy: value => { energy = value; } };
}

test('manual recording delivers a complete blob and releases its stream', async t => {
    const env = recordingEnvironment(t);
    const audio = [];
    const states = [];
    const mic = new MicrophoneRecorder({
        onAudio: (blob, options) => audio.push({ blob, options }),
        onStateChange: (...state) => states.push(state),
    });
    await mic.startManualRecording();
    await mic.startManualRecording();
    assert.equal(env.streams.length, 1);
    assert.equal(mic.isRecording, true);
    mic.stopRecording();
    await Promise.resolve();
    assert.equal(mic.isRecording, false);
    assert.equal(env.streams[0].track.stops, 1);
    assert.equal(await audio[0].blob.text(), 'voice');
    assert.equal(audio[0].blob.type, 'audio/webm;codecs=opus');
    assert.deepEqual(audio[0].options, { includeScreenContext: true });
    assert.deepEqual(states.map(state => state.slice(0, 2)), [['recording', true], ['recording', false]]);
});

test('automatic recording segments speech and stops timers, tracks and late delivery on mode change', async t => {
    const env = recordingEnvironment(t);
    const audio = [];
    let now = 1000;
    t.mock.method(Date, 'now', () => now);
    const mic = new MicrophoneRecorder({ onAudio: (...args) => audio.push(args) });
    await mic.applyVoiceMode('automatic');
    await mic.applyVoiceMode('automatic');
    assert.equal(env.streams.length, 1);
    assert.equal(env.timers.size, 1);
    const tick = [...env.timers][0];
    env.setEnergy(30);
    tick();
    assert.equal(env.recorders.length, 0);
    env.setEnergy(31);
    tick();
    env.setEnergy(0);
    now += 799;
    tick();
    assert.equal(env.recorders[0].state, 'recording');
    now++;
    tick();
    await Promise.resolve();
    assert.equal(audio.length, 1);
    assert.equal(await audio[0][0].text(), 'voice');
    assert.deepEqual(audio[0][1], { includeScreenContext: true });
    env.setEnergy(40);
    tick();
    await mic.applyVoiceMode('push_to_talk');
    await mic.applyVoiceMode('push_to_talk');
    assert.equal(env.timers.size, 0);
    assert.equal(env.streams[0].track.stops, 1);
    assert.equal(audio.length, 1);
});

test('denied microphone permission leaves recording inactive and emits no audio', async t => {
    const env = recordingEnvironment(t);
    t.mock.method(console, 'warn', () => {});
    env.mediaDevices.getUserMedia = async () => { throw new Error('permission denied'); };
    const mic = new MicrophoneRecorder({ onAudio: () => assert.fail('unexpected audio') });
    await mic.startManualRecording();
    await mic.applyVoiceMode('automatic');
    assert.equal(mic.isRecording, false);
    assert.equal(env.timers.size, 0);
    assert.equal(env.recorders.length, 0);
});
