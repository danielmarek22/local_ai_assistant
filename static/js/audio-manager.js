import { CONFIG } from './config.js';

export class AudioManager {
    constructor() {
        this.audioQueue = [];
        this.isPlaying = false;
        this.audioContext = null;
        this.analyser = null;
        this.dataArray = null;
        
        this.audioEl = new Audio();
        this.audioEl.crossOrigin = "anonymous";
        
        this.audioEl.onended = () => {
            this.isPlaying = false;
            this.playNext();
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

    playNext() {
        if (this.isPlaying || this.audioQueue.length === 0) return;

        this.isPlaying = true;
        const audioUrl = this.audioQueue.shift();
        this.audioEl.src = audioUrl;
        
        this.audioEl.play().catch(e => {
            console.error("Audio play failed:", e);
            this.isPlaying = false;
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