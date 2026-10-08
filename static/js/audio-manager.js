import { CONFIG } from './config.js';

export class AudioManager {
    constructor() {
        this.audioQueue = [];
        this.isPlaying = false;
        this.playbackBlocked = false;
        this.isSpeechActive = false;
        this.speechEndTimer = null;
        this.audioContext = null;
        this.contextResumePromise = null;
        this.loadedAudioUrl = null;
        this.analyser = null;
        this.dataArray = null;
        this.onSpeechStart = null;
        this.onSpeechEnd = null;
        
        this.audioEl = new Audio();
        this.audioEl.crossOrigin = "anonymous";
        this.audioEl.preload = 'auto';
        this.volume = CONFIG.AUDIO.DEFAULT_VOLUME;
        this.audioEl.volume = this.volume;

        this.audioEl.onplaying = () => {
            console.debug('Assistant audio playing:', this.audioEl.currentSrc);
            this.updateSpeechActivity();
        };
        this.audioEl.onpause = () => {
            this.updateSpeechActivity();
        };
        
        this.audioEl.onended = () => {
            this.isPlaying = false;
            this.loadedAudioUrl = null;
            this.playNext();
            this.updateSpeechActivity();
        };

        // Unlock speech during a real interaction, including enabling voice mode.
        // A transcript arriving over WebSocket is not a user activation event.
        document.addEventListener('pointerdown', () => this.init());
        document.addEventListener('keydown', () => this.init());
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState !== 'hidden') this.retryPlayback();
        });
        window.addEventListener('focus', () => this.retryPlayback());
        window.addEventListener('pageshow', () => this.retryPlayback());
    }

    init() {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256; 
            this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
            
            const track = this.audioContext.createMediaElementSource(this.audioEl);
            
            // --- SIGNAL ROUTING ---
            
            // 1. Send raw audio to the lipsync analyser first so animations stay perfectly accurate
            track.connect(this.analyser);

            // 2. Send the clean signal directly to the global speakers
            this.analyser.connect(this.audioContext.destination);

            console.log("Audio Context Initialized with Clean Audio");
        }
        this.retryPlayback();
    }

    async resumeAudioContext() {
        if (!this.audioContext || this.audioContext.state === 'running') return true;
        if (this.contextResumePromise) return this.contextResumePromise;

        // Autoplay blocking may leave resume() pending instead of rejecting it.
        // Keep the queue recoverable even when the browser never settles it.
        let resumeTimer;
        this.contextResumePromise = new Promise((resolve) => {
            resumeTimer = window.setTimeout(() => {
                console.warn('Audio context resume timed out; speech is waiting for activation');
                resolve(false);
            }, CONFIG.AUDIO.CONTEXT_RESUME_TIMEOUT_MS);
            try {
                Promise.resolve(this.audioContext.resume()).then(() => {
                    resolve(this.audioContext.state === 'running');
                }, (error) => {
                    console.warn('Audio context could not resume:', error);
                    resolve(false);
                });
            } catch (error) {
                console.warn('Audio context could not resume:', error);
                resolve(false);
            }
        }).finally(() => {
            window.clearTimeout(resumeTimer);
            this.contextResumePromise = null;
        });
        return this.contextResumePromise;
    }

    retryPlayback() {
        this.playbackBlocked = false;
        // Call resume synchronously from interaction handlers, even with no clips
        // queued, so voice replies can play after the user switches tabs.
        void this.resumeAudioContext();
        this.playNext();
    }

    makeDistortionCurve(amount) {
        let k = typeof amount === 'number' ? amount : 50;
        let n_samples = 44100;
        let curve = new Float32Array(n_samples);
        let deg = Math.PI / 180;
        let i = 0;
        let x;
        for ( ; i < n_samples; ++i ) {
            x = i * 2 / n_samples - 1;
            curve[i] = ( 3 + k ) * x * 20 * deg / ( Math.PI + k * Math.abs(x) );
        }
        return curve;
    }

    queueAudio(url) {
        console.debug('Assistant audio received:', url, 'visibility:', document.visibilityState);
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
        if (this.isPlaying || this.playbackBlocked || this.audioQueue.length === 0) return;

        this.isPlaying = true;
        const audioUrl = this.audioQueue.shift();
        // Loading is independent of speaker permission. Start the request before
        // waiting for resume(), and retain the loaded resource on blocked retries.
        if (this.loadedAudioUrl !== audioUrl) {
            this.loadedAudioUrl = audioUrl;
            this.audioEl.src = audioUrl;
            this.audioEl.load();
            console.debug('Assistant audio loading:', audioUrl);
        }
        // Do not consume a clip silently through a suspended Web Audio graph.
        this.resumeAudioContext().then((running) => {
            if (!running) {
                throw new DOMException('Audio context is not running', 'NotAllowedError');
            }
            return this.audioEl.play();
        }).catch(e => {
            console.error("Audio play failed:", e);
            this.isPlaying = false;
            if (e.name === 'NotAllowedError' || e.name === 'AbortError') {
                // Keep the first blocked clip and its order. Retry on activation
                // rather than draining the entire speech queue on one rejection.
                this.audioQueue.unshift(audioUrl);
                this.playbackBlocked = true;
            } else {
                this.loadedAudioUrl = null;
            }
            this.updateSpeechActivity();
            this.playNext();
        });
    }

    getVisemeData() {
        if (!this.analyser || this.audioEl.paused) {
            return { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0, intensity: 0 };
        }

        this.analyser.getByteFrequencyData(this.dataArray);

        const binCount = this.dataArray.length;
        const sampleRate = this.audioContext.sampleRate;
        const binHz = sampleRate / this.analyser.fftSize;

        const bandEnergy = (minHz, maxHz) => {
            const minBin = Math.floor(minHz / binHz);
            const maxBin = Math.min(Math.ceil(maxHz / binHz), binCount - 1);
            let sum = 0;
            for (let i = minBin; i <= maxBin; i++) sum += this.dataArray[i];
            return sum / ((maxBin - minBin + 1) * 255);
        };

        const intensity = bandEnergy(80, 4000);
        // Replace the hard gate with a smooth ramp
        const gate = Math.min(1, Math.max(0, (intensity - 0.05) / 0.1));

        return {
            aa:  Math.min(1, bandEnergy(700,  1200) * 0.8) * gate,
            ih:  Math.min(1, bandEnergy(300,   700) * 0.6) * gate,
            ou:  Math.min(1, bandEnergy(300,   800) * 0.5) * gate,
            ee:  Math.min(1, bandEnergy(2000, 3500) * 1.0) * gate,
            oh:  Math.min(1, bandEnergy(500,   900) * 0.6) * gate,
            intensity,
        };
    }
}
