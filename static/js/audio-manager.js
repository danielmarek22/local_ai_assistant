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
            
            // --- SUBTLE SCI-FI EFFECTS CHAIN ---
            
            // 1. Slight Distortion (WaveShaper)
            const distortion = this.audioContext.createWaveShaper();
            // A value of 5 gives it a slight digital "crunch" without ruining clarity
            distortion.curve = this.makeDistortionCurve(5); 
            distortion.oversample = '4x';

            // 2. Metallic Ring (Comb Filter via a micro-delay)
            const delay = this.audioContext.createDelay();
            delay.delayTime.value = 0.015; // 15ms creates a resonant metallic pitch

            const delayGain = this.audioContext.createGain();
            delayGain.gain.value = 0.25; // Blend the robot ring in at only 25% volume

            // --- SIGNAL ROUTING ---
            
            // Send raw audio to the lipsync analyser first so animations stay perfectly accurate
            track.connect(this.analyser);

            // Split the signal out of the analyser into two paths
            this.analyser.connect(distortion); // Path A: Straight to distortion
            this.analyser.connect(delay);      // Path B: To the delay node

            // Bring Path B back into Path A
            delay.connect(delayGain);
            delayGain.connect(distortion);

            // Send the final mixed, slightly robotic signal to the global speakers
            distortion.connect(this.audioContext.destination);

            console.log("Audio Context Initialized with Subtle Comm-Link Effects");
        } else if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }
    }

    // Add this helper method anywhere inside your AudioManager class!
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
