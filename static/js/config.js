export const CONFIG = {
    SYSTEM: {
        // Points to your Python server
        WS_URL: "ws://localhost:8000/ws",
        RECONNECT_INTERVAL_MS: 3000
    },
    AVATAR: {
        MODEL_PATH: '/static/avatar.vrm',
        // Movement smoothing (lower is slower/smoother)
        POSE_LERP_FACTOR: 0.05,
        // Blink probabilities per frame
        BLINK_CHANCE_IDLE: 0.005,
        BLINK_CHANCE_ACTIVE: 0.002, // thinking/searching
        
        // Define new states here without touching logic code
        POSES: {
            idle:       { neckX: 0,     neckY: 0,    spineY: 0 },
            thinking:   { neckX: -0.25, neckY: 0.35, spineY: 0.05 },
            searching:  { neckX: -0.15, neckY: -0.4, spineY: -0.05 },
            responding: { neckX: 0.05,  neckY: 0,    spineY: 0 }
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