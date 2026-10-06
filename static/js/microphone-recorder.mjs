// Owns microphone resources and recording state; has no DOM or preference access.
export class MicrophoneRecorder {
    constructor({ onAudio = null, onStateChange = () => {} } = {}) {
        this.onAudio = onAudio;
        this.onStateChange = onStateChange;
        this._mediaRecorder = null;
        this._audioChunks = [];
        this._isRecording = false;
        this._isAlwaysListening = false;
        this._alwaysListeningStream = null;
        this._audioContext = null;
        this._alwaysListeningAnalyser = null;
        this._speechDetectedTime = 0;
        this._alwaysListeningSpeechCheck = null;
        this._isSendingAlwaysListeningChunk = false;
        this._alwaysListeningVoiceContextSent = false;
    }

    get isRecording() {
        return this._isRecording;
    }

    async toggleAlwaysListening() {
        if (this._isAlwaysListening) {
            // Disable always listening
            await this.stopAlwaysListening();
        } else {
            // Enable always listening
            await this.startAlwaysListening();
        }
    }

    async startAlwaysListening() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this._alwaysListeningStream = stream;
            this._isAlwaysListening = true;

            this.onStateChange('always-listening', true, 'Automatic voice detection is active');

            // Setup Web Audio API for voice activity detection
            if (!this._audioContext) {
                this._audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            const audioContext = this._audioContext;
            const source = audioContext.createMediaStreamSource(stream);
            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 2048;
            source.connect(analyser);

            this._alwaysListeningAnalyser = analyser;

            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : '';

            this._mediaRecorder = null;
            this._audioChunks = [];
            this._speechDetectedTime = 0;
            this._isSendingAlwaysListeningChunk = false;
            this._alwaysListeningVoiceContextSent = false;

            const startSpeechSegment = () => {
                if (this._mediaRecorder?.state === 'recording') return;

                const chunks = [];
                const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
                this._mediaRecorder = recorder;
                recorder.ondataavailable = (e) => {
                    if (e.data.size > 0) {
                        chunks.push(e.data);
                    }
                };
                recorder.onstop = () => {
                    this._isSendingAlwaysListeningChunk = false;
                    if (this._mediaRecorder === recorder) {
                        this._mediaRecorder = null;
                    }

                    if (!this._isAlwaysListening || !chunks.length || !this.onAudio) return;

                    const blob = new Blob(chunks, {
                        type: recorder.mimeType || 'audio/webm',
                    });
                    this.onAudio(blob, {
                        includeScreenContext: !this._alwaysListeningVoiceContextSent,
                    });
                    this._alwaysListeningVoiceContextSent = true;
                };
                this._isSendingAlwaysListeningChunk = true;
                recorder.start();
            };

            const stopSpeechSegment = () => {
                if (this._mediaRecorder?.state === 'recording') {
                    this._mediaRecorder.stop();
                }
            };

            // Check for speech activity every 100ms and send each complete utterance
            // after a short silence, so STT receives a decodable WebM blob.
            this._alwaysListeningSpeechCheck = setInterval(() => {
                if (!this._isAlwaysListening) return;

                const hasSpeech = this.detectSpeechActivity(analyser);

                if (hasSpeech) {
                    this._speechDetectedTime = Date.now();
                    startSpeechSegment();
                }

                const timeSinceSpeech = Date.now() - this._speechDetectedTime;
                if (timeSinceSpeech >= 800) {
                    stopSpeechSegment();
                    this._alwaysListeningVoiceContextSent = false;
                }
            }, 100);
        } catch (err) {
            console.warn('Microphone access denied:', err);
            this._isAlwaysListening = false;
        }
    }

    detectSpeechActivity(analyser) {
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(dataArray);

        // Calculate average energy across frequency bins
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
            sum += dataArray[i];
        }
        const average = sum / dataArray.length;

        // Consider speech detected if average frequency energy > 30
        // (threshold can be adjusted based on testing)
        return average > 30;
    }

    async stopAlwaysListening() {
        if (this._alwaysListeningSpeechCheck) {
            clearInterval(this._alwaysListeningSpeechCheck);
            this._alwaysListeningSpeechCheck = null;
        }

        if (this._mediaRecorder?.state === 'recording') {
            this._mediaRecorder.stop();
        }
        this._mediaRecorder = null;

        if (this._alwaysListeningStream) {
            this._alwaysListeningStream.getTracks().forEach((t) => t.stop());
            this._alwaysListeningStream = null;
        }

        this._alwaysListeningAnalyser = null;

        this._isAlwaysListening = false;
        this._audioChunks = [];
        this._alwaysListeningVoiceContextSent = false;
        this.onStateChange('always-listening', false, 'Voice input');
    }

    stopRecording() {
        if (this._mediaRecorder && this._isRecording) {
            this._mediaRecorder.stop();
        }
    }

    async startManualRecording() {
        if (this._isRecording) return;

        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (err) {
            console.warn('Microphone access denied:', err);
            return;
        }

        this._audioChunks = [];
        this._isRecording = true;
        this.onStateChange('recording', true, 'Recording... release hotkey to send');

        // Prefer webm/opus; fall back to whatever the browser supports.
        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
            ? 'audio/webm;codecs=opus'
            : '';

        this._mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
        this._mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) this._audioChunks.push(e.data);
        };

        this._mediaRecorder.onstop = () => {
            stream.getTracks().forEach((t) => t.stop());
            this._isRecording = false;
            this.onStateChange('recording', false, 'Voice input');

            const blob = new Blob(this._audioChunks, {
                type: this._mediaRecorder.mimeType || 'audio/webm',
            });
            this._audioChunks = [];

            if (this.onAudio) {
                this.onAudio(blob, { includeScreenContext: true });
            }
        };

        this._mediaRecorder.start();
    }

    async applyVoiceMode(mode) {
        if (!navigator.mediaDevices?.getUserMedia) return;

        if (mode === 'automatic') {
            if (this._isRecording) {
                this.stopRecording();
            }
            if (!this._isAlwaysListening) {
                await this.startAlwaysListening();
            }
            return;
        }

        if (this._isAlwaysListening) {
            await this.stopAlwaysListening();
        }
    }
}
