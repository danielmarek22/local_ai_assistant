import { CONFIG } from './config.js';

// Owns persisted preferences only; UI state and side effects stay with UIManager.
export class PreferenceStore {
    constructor(storage = () => localStorage) {
        // Access storage inside each guarded operation: even obtaining it may throw.
        this.storage = storage;
    }

    save(key, value, label) {
        try {
            this.storage().setItem(CONFIG.UI.STORAGE_KEYS[key], String(value));
        } catch (error) {
            console.warn(`Failed to persist ${label}:`, error);
        }
    }

    readStoredPlaybackVolume() {
        try {
            const rawValue = this.storage().getItem(CONFIG.UI.STORAGE_KEYS.AUDIO_VOLUME);
            if (rawValue === null) {
                return CONFIG.AUDIO.DEFAULT_VOLUME;
            }

            const parsedValue = Number(rawValue);
            if (Number.isFinite(parsedValue) && parsedValue >= 0 && parsedValue <= 1) {
                return parsedValue;
            }

            this.storage().removeItem(CONFIG.UI.STORAGE_KEYS.AUDIO_VOLUME);
        } catch (error) {
            console.warn('Failed to restore playback volume:', error);
        }

        return CONFIG.AUDIO.DEFAULT_VOLUME;
    }

    readStoredAgentMode() {
        try {
            const storedValue = this.storage().getItem(CONFIG.UI.STORAGE_KEYS.AGENT_MODE);
            if (storedValue !== null) {
                return storedValue === 'true';
            }

            const legacyValue = this.storage().getItem(CONFIG.UI.STORAGE_KEYS.INSTANT_MODE);
            if (legacyValue !== null) {
                const migratedAgentMode = legacyValue === 'false';
                this.storage().setItem(CONFIG.UI.STORAGE_KEYS.AGENT_MODE, String(migratedAgentMode));
                this.storage().removeItem(CONFIG.UI.STORAGE_KEYS.INSTANT_MODE);
                return migratedAgentMode;
            }

            return Boolean(CONFIG.UI.AGENT_MODE_DEFAULT);
        } catch (error) {
            console.warn('Failed to restore agent mode:', error);
            return Boolean(CONFIG.UI.AGENT_MODE_DEFAULT);
        }
    }

    readStoredHideThinking() {
        try {
            const storedValue = this.storage().getItem(CONFIG.UI.STORAGE_KEYS.HIDE_THINKING);
            if (storedValue === null) {
                return Boolean(CONFIG.UI.HIDE_THINKING_DEFAULT);
            }
            return storedValue === 'true';
        } catch (error) {
            console.warn('Failed to restore hide thinking setting:', error);
            return Boolean(CONFIG.UI.HIDE_THINKING_DEFAULT);
        }
    }

    normalizeVoiceMode(mode) {
        return CONFIG.UI.VOICE_MODES.includes(mode)
            ? mode
            : CONFIG.UI.VOICE_MODE_DEFAULT;
    }

    readStoredVoiceMode() {
        try {
            const rawValue = this.storage().getItem(CONFIG.UI.STORAGE_KEYS.VOICE_MODE);
            const mode = this.normalizeVoiceMode(rawValue);
            if (rawValue !== null && rawValue !== mode) {
                this.storage().removeItem(CONFIG.UI.STORAGE_KEYS.VOICE_MODE);
            }
            return mode;
        } catch (error) {
            console.warn('Failed to restore voice mode:', error);
            return CONFIG.UI.VOICE_MODE_DEFAULT;
        }
    }

    normalizeScreenCapturePolicy(policy) {
        return CONFIG.UI.SCREEN_CAPTURE_POLICIES.includes(policy)
            ? policy
            : CONFIG.UI.SCREEN_CAPTURE_POLICY_DEFAULT;
    }

    readStoredScreenCapturePolicy() {
        try {
            const rawValue = this.storage().getItem(CONFIG.UI.STORAGE_KEYS.SCREEN_CAPTURE_POLICY);
            const policy = this.normalizeScreenCapturePolicy(rawValue);
            if (rawValue !== null && rawValue !== policy) {
                this.storage().removeItem(CONFIG.UI.STORAGE_KEYS.SCREEN_CAPTURE_POLICY);
            }
            return policy;
        } catch (error) {
            console.warn('Failed to restore screen capture policy:', error);
            return CONFIG.UI.SCREEN_CAPTURE_POLICY_DEFAULT;
        }
    }
}
