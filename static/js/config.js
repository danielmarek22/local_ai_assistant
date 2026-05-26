export const CONFIG = {
    SYSTEM: {
        // Points to your Python server
        WS_URL: "ws://localhost:8000/ws",
        RECONNECT_INTERVAL_MS: 3000
    },
    AVATAR: {
        MODEL_PATH: '/static/avatar.vrm',
        EXPRESSIONS: ['happy', 'angry', 'sad', 'relaxed', 'surprised', 'neutral'],
        
        // --- UPDATED: Animation Arrays ---
        ANIMATIONS: {
            // Add multiple paths to the arrays
            idle:       [
                '/static/animations/Idle/Idle_1.fbx', 
                '/static/animations/Idle/Idle_2.fbx',
                '/static/animations/Idle/Idle_3.fbx',
                '/static/animations/Idle/Idle_4.fbx',
                '/static/animations/Idle/Idle_5.fbx',
                '/static/animations/Idle/Idle_6.fbx'
            ],
            thinking:   [
                '/static/animations/Thinking/Thinking_1.fbx',
                '/static/animations/Thinking/Thinking_2.fbx',
                // '/static/animations/Thinking/Thinking_3.fbx',
                // '/static/animations/Thinking/Thinking_4.fbx'
            ],
            dreaming:   [
                '/static/animations/Dreaming/Dreaming_1.fbx',
                '/static/animations/Dreaming/Dreaming_2.fbx'
            ],
            searching:  [
                '/static/animations/Typing_Or_Searching.fbx'
            ],
            responding: [
                '/static/animations/Talking/Talking_1.fbx', 
                '/static/animations/Talking/Talking_2.fbx',
                '/static/animations/Talking/Talking_3.fbx',
                '/static/animations/Talking/Talking_4.fbx',
                '/static/animations/Talking/Talking_5.fbx',
                '/static/animations/Talking/Talking_6.fbx',
            ]
        }
    },
    AUDIO: {
        DEFAULT_VOLUME: 1,
        // Lower = mouth opens more easily
        LIP_SYNC_SENSITIVITY: 60, 
        // 0.0 = no smoothing, 1.0 = no movement
        LIP_SYNC_SMOOTHING: 0.3,
        SPEECH_END_HOLD_MS: 250
    },
    UI: {
        STORAGE_KEYS: {
            CHAT_HISTORY: 'astra-chat-history',
            CURRENT_SESSION: 'astra-current-session',
            AUDIO_VOLUME: 'astra-audio-volume',
            MIC_HOTKEY: 'astra-mic-hotkey',
            SCREEN_CAPTURE_POLICY: 'astra-screen-capture-policy'
        },
        SCREEN_CAPTURE_POLICY_DEFAULT: 'voice',
        SCREEN_CAPTURE_POLICIES: ['voice', 'watchdog'],
        STATUS_TEXT: {
            idle:       "Astra is Idle",
            thinking:   "Astra is Thinking...",
            dreaming:   "Astra is Dreaming...",
            searching:  "Searching Knowledge Base...",
            responding: "Astra is Responding"
        },
        MIC_HOTKEY_DEFAULT: ' '  // spacebar
    }
};
