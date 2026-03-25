import { CONFIG } from './config.js';

export class AudioManager {
    constructor() {
        this.audioQueue = [];
        this.isPlaying = false;
        this.isSpeechActive = false;
        this.speechEndTimer = null;
        this.audioContext = null;
        this.analyser = null;
        this.dataArray = null;
        this.onSpeechStart = null;
        this.onSpeechEnd = null;
        
        this.audioEl = new Audio();
        this.audioEl.crossOrigin = "anonymous";
        this.volume = CONFIG.AUDIO.DEFAULT_VOLUME;
        this.audioEl.volume = this.volume;

        this.audioEl.onplaying = () => {
            this.updateSpeechActivity();
        };
        this.audioEl.onpause = () => {
            this.updateSpeechActivity();
        };
        
        this.audioEl.onended = () => {
            this.isPlaying = false;
            this.playNext();
            this.updateSpeechActivity();
        };
    }

    init() {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256; 
            this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
            
            const track = this.audioContext.createMediaElementSource(this.audioEl);
            track.connect(this.analyser);
            this.analyser.connect(this.audioContext.destination);
            console.log("Audio Context Initialized");
        } else if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }
    }

    queueAudio(url) {
        this.audioQueue.push(url);
        this.playNext();
    }

    setVolume(volume) {
        const nextVolume = Number.isFinite(volume)
            ? Math.min(1, Math.max(0, volume))
            : CONFIG.AUDIO.DEFAULT_VOLUME;
        this.volume = nextVolume;
        this.audioEl.volume = nextVolume;
    }

    getVolume() {
        return this.volume;
    }

    setPlaybackHandlers({ onSpeechStart, onSpeechEnd } = {}) {
        this.onSpeechStart = onSpeechStart || null;
        this.onSpeechEnd = onSpeechEnd || null;
    }

    hasActiveSpeech() {
        return this.isSpeechActive;
    }

    getCurrentSpeechActivity() {
        return !this.audioEl.paused && !this.audioEl.ended && Boolean(this.audioEl.currentSrc);
    }

    updateSpeechActivity() {
        const nextSpeechActive = this.getCurrentSpeechActivity();
        if (nextSpeechActive === this.isSpeechActive) return;

        if (nextSpeechActive) {
            this.clearSpeechEndTimer();
            this.isSpeechActive = true;
            this.onSpeechStart?.();
            return;
        }

        this.scheduleSpeechEnd();
    }

    clearSpeechEndTimer() {
        if (!this.speechEndTimer) return;

        clearTimeout(this.speechEndTimer);
        this.speechEndTimer = null;
    }

    scheduleSpeechEnd() {
        this.clearSpeechEndTimer();
        this.speechEndTimer = window.setTimeout(() => {
            this.speechEndTimer = null;

            if (this.getCurrentSpeechActivity()) return;
            if (!this.isSpeechActive) return;

            this.isSpeechActive = false;
            this.onSpeechEnd?.();
        }, CONFIG.AUDIO.SPEECH_END_HOLD_MS);
    }

    playNext() {
        if (this.isPlaying || this.audioQueue.length === 0) return;

        this.isPlaying = true;
        const audioUrl = this.audioQueue.shift();
        this.audioEl.src = audioUrl;
        
        this.audioEl.play().catch(e => {
            console.error("Audio play failed:", e);
            this.isPlaying = false;
            this.updateSpeechActivity();
            this.playNext();
        });
    }

    getLipSyncValue() {
        if (!this.analyser || this.audioEl.paused) return 0;

        this.analyser.getByteFrequencyData(this.dataArray);
        
        let sum = 0;
        for (let i = 0; i < this.dataArray.length; i++) {
            sum += this.dataArray[i];
        }
        const average = sum / this.dataArray.length;
        
        // Map volume to mouth open
        return Math.min(1.0, average / CONFIG.AUDIO.LIP_SYNC_SENSITIVITY);
    }
}
