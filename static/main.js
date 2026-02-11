import { AudioManager } from './js/audio-manager.js';
import { AvatarManager } from './js/avatar-manager.js';
import { UIManager } from './js/ui-manager.js';
import { NetworkClient } from './js/network-client.js';

// 1. Initialize Sub-systems
const audioManager = new AudioManager();
const uiManager = new UIManager();

// 2. Initialize Avatar
const avatarManager = new AvatarManager(
    'canvas-container', 
    () => audioManager.getLipSyncValue() 
);

// 3. Define Network Handlers
const handlers = {
    onState: (state) => {
        uiManager.updateStatus(state);
        avatarManager.setState(state);
        if (state === "responding") {
            uiManager.startAiMessage();
        }
    },
    onChunk: (content) => {
        if (avatarManager.currentState !== 'responding') {
            uiManager.updateStatus('responding');
            avatarManager.setState('responding');
        }
        uiManager.appendToAiMessage(content);
    },
    onAudio: (url) => {
        audioManager.queueAudio(url);
    },
    onEnd: (finalContent) => {
        uiManager.finalizeAiMessage(finalContent);
        uiManager.updateStatus('idle');
        avatarManager.setState('idle');
    }
};

// 4. Connect Network
// (URL is now handled inside NetworkClient via Config)
const client = new NetworkClient(handlers);
client.connect();

// 5. Handle User Input
uiManager.onSend((text) => {
    // Browsers require user interaction to start AudioContext
    audioManager.init();
    
    uiManager.appendUserMessage(text);
    client.sendMessage(text);
});