import test from 'node:test';
import assert from 'node:assert/strict';
import { CONFIG } from '../static/js/config.js';
import { PreferenceStore } from '../static/js/preferences.mjs';

const keys = CONFIG.UI.STORAGE_KEYS;
function storage(entries = []) {
    const values = new Map(entries);
    return {
        getItem: key => values.get(key) ?? null,
        setItem: (key, value) => values.set(key, value),
        removeItem: key => values.delete(key),
    };
}

test('agent mode migration preserves precedence and removes the legacy key only after saving', () => {
    for (const [current, legacy, expected, removed] of [
        [null, 'false', true, true],
        [null, 'true', false, true],
        ['false', 'false', false, false],
        ['true', 'true', true, false],
        [null, null, Boolean(CONFIG.UI.AGENT_MODE_DEFAULT), false],
    ]) {
        const saved = storage([[keys.AGENT_MODE, current], [keys.INSTANT_MODE, legacy]]);
        const preferences = new PreferenceStore(() => saved);
        assert.equal(preferences.readStoredAgentMode(), expected);
        assert.equal(saved.getItem(keys.INSTANT_MODE), removed ? null : legacy);
        if (removed) assert.equal(saved.getItem(keys.AGENT_MODE), String(expected));
    }
});

test('stored values preserve defaults, valid values and invalid-value cleanup', () => {
    for (const [key, method, value, expected, removed] of [
        ['AUDIO_VOLUME', 'readStoredPlaybackVolume', null, CONFIG.AUDIO.DEFAULT_VOLUME, false],
        ['AUDIO_VOLUME', 'readStoredPlaybackVolume', '0', 0, false],
        ['AUDIO_VOLUME', 'readStoredPlaybackVolume', '1', 1, false],
        ['AUDIO_VOLUME', 'readStoredPlaybackVolume', '1.1', CONFIG.AUDIO.DEFAULT_VOLUME, true],
        ['AUDIO_VOLUME', 'readStoredPlaybackVolume', 'oops', CONFIG.AUDIO.DEFAULT_VOLUME, true],
        ['HIDE_THINKING', 'readStoredHideThinking', null, Boolean(CONFIG.UI.HIDE_THINKING_DEFAULT), false],
        ['HIDE_THINKING', 'readStoredHideThinking', 'true', true, false],
        ['HIDE_THINKING', 'readStoredHideThinking', 'false', false, false],
        ...CONFIG.UI.VOICE_MODES.map(value => ['VOICE_MODE', 'readStoredVoiceMode', value, value, false]),
        ['VOICE_MODE', 'readStoredVoiceMode', 'unknown', CONFIG.UI.VOICE_MODE_DEFAULT, true],
        ...CONFIG.UI.SCREEN_CAPTURE_POLICIES.map(value => ['SCREEN_CAPTURE_POLICY', 'readStoredScreenCapturePolicy', value, value, false]),
        ['SCREEN_CAPTURE_POLICY', 'readStoredScreenCapturePolicy', 'unknown', CONFIG.UI.SCREEN_CAPTURE_POLICY_DEFAULT, true],
    ]) {
        const saved = storage([[keys[key], value]]);
        assert.equal(new PreferenceStore(() => saved)[method](), expected, `${key}: ${value}`);
        assert.equal(saved.getItem(keys[key]), removed ? null : value);
    }
});

test('unavailable storage and failed migration writes preserve fallback behavior', t => {
    t.mock.method(console, 'warn', () => {});
    const blocked = new PreferenceStore(() => { throw new Error('storage blocked'); });
    for (const [method, expected] of [
        ['readStoredPlaybackVolume', CONFIG.AUDIO.DEFAULT_VOLUME],
        ['readStoredAgentMode', Boolean(CONFIG.UI.AGENT_MODE_DEFAULT)],
        ['readStoredHideThinking', Boolean(CONFIG.UI.HIDE_THINKING_DEFAULT)],
        ['readStoredVoiceMode', CONFIG.UI.VOICE_MODE_DEFAULT],
        ['readStoredScreenCapturePolicy', CONFIG.UI.SCREEN_CAPTURE_POLICY_DEFAULT],
    ]) assert.equal(blocked[method](), expected);
    assert.doesNotThrow(() => blocked.save('AUDIO_VOLUME', 0.5, 'playback volume'));
    const saved = storage([[keys.INSTANT_MODE, 'false']]);
    saved.setItem = () => { throw new Error('quota exceeded'); };
    assert.equal(new PreferenceStore(() => saved).readStoredAgentMode(), Boolean(CONFIG.UI.AGENT_MODE_DEFAULT));
    assert.equal(saved.getItem(keys.INSTANT_MODE), 'false');
});
