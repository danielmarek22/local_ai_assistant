export const CONFIG = {
    SYSTEM: {
        // Points to your Python server
        WS_URL: "ws://localhost:8000/ws",
        RECONNECT_INTERVAL_MS: 3000
    },
    AVATAR: {
        MODEL_PATH: '/static/avatar.vrm',
        
        // --- UPDATED: Animation Arrays ---
        ANIMATIONS: {
            // Add multiple paths to the arrays
            idle:       [
                '/static/animations/Idle_1.fbx', 
                '/static/animations/Idle_2.fbx',
                // '/static/animations/Idle_3.fbx'
            ],
            thinking:   [
                '/static/animations/Thinking_1.fbx' // Arrays work even with just one item!
            ],
            searching:  [
                '/static/animations/Typing_Or_Searching.fbx'
            ],
            responding: [
                '/static/animations/Talking_1.fbx', 
                // '/static/animations/Talking_2.fbx',
                // '/static/animations/Explaining.fbx'
            ]
        }
    },
    AUDIO: {
        // Lower = mouth opens more easily
        LIP_SYNC_SENSITIVITY: 60, 
        // 0.0 = no smoothing, 1.0 = no movement
        LIP_SYNC_SMOOTHING: 0.3   
    },
    UI: {
        STATUS_TEXT: {
            idle:       "Astra is Idle",
            thinking:   "Astra is Thinking...",
            searching:  "Searching Knowledge Base...",
            responding: "Astra is Responding"
        }
    }
};