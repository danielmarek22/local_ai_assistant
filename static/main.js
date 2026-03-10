import { AudioManager } from './js/audio-manager.js';
import { AvatarManager } from './js/avatar-manager.js';
import { UIManager } from './js/ui-manager.js';
import { NetworkClient } from './js/network-client.js';
import { CONFIG } from './js/config.js';

// 1. Initialize Sub-systems
const audioManager = new AudioManager();
const uiManager = new UIManager();
const sessionStorageKey = CONFIG.UI.STORAGE_KEYS.CURRENT_SESSION;
let currentServerInstanceId = null;
let currentSessionId = null;
let assistantState = 'idle';

function readStoredSessionContext() {
    const rawValue = sessionStorage.getItem(sessionStorageKey);
    if (!rawValue) return null;

    try {
        return JSON.parse(rawValue);
    } catch (error) {
        console.warn('Failed to parse current session context:', error);
        sessionStorage.removeItem(sessionStorageKey);
        return null;
    }
}

function persistSessionContext() {
    if (!currentServerInstanceId || !currentSessionId) return;

    sessionStorage.setItem(sessionStorageKey, JSON.stringify({
        serverInstanceId: currentServerInstanceId,
        sessionId: currentSessionId,
    }));
}

function clearSessionContext() {
    currentServerInstanceId = null;
    currentSessionId = null;
    sessionStorage.removeItem(sessionStorageKey);
}

// 2. Initialize Avatar
const avatarManager = new AvatarManager(
    'canvas-container', 
    () => audioManager.getLipSyncValue() 
);

function syncAssistantPresentation() {
    const visualState = audioManager.hasActiveSpeech() ? 'responding' : assistantState;
    uiManager.updateStatus(visualState);
    avatarManager.setState(visualState);
}

audioManager.setPlaybackHandlers({
    onSpeechStart: () => {
        syncAssistantPresentation();
    },
    onSpeechEnd: () => {
        syncAssistantPresentation();
    }
});

// 3. Define Network Handlers
const handlers = {
    onSessionInit: ({ serverInstanceId, sessionId }) => {
        currentServerInstanceId = serverInstanceId;
        currentSessionId = sessionId;
        uiManager.setSessionScope(serverInstanceId, sessionId);
        persistSessionContext();
    },
    onState: (state) => {
        assistantState = state;
        syncAssistantPresentation();
        if (state === "responding") {
            uiManager.startAiMessage();
        }
    },
    onChunk: (content) => {
        assistantState = 'responding';
        syncAssistantPresentation();
        uiManager.appendToAiMessage(content);
    },
    onAudio: (url) => {
        audioManager.queueAudio(url);
    },
    onEnd: (finalContent) => {
        uiManager.finalizeAiMessage(finalContent);
        assistantState = 'idle';
        syncAssistantPresentation();
    }
};

// 4. Connect Network
// (URL is now handled inside NetworkClient via Config)
const client = new NetworkClient(handlers);
const storedSessionContext = readStoredSessionContext();

if (storedSessionContext?.sessionId && storedSessionContext?.serverInstanceId) {
    client.connect({
        sessionId: storedSessionContext.sessionId,
        serverInstanceId: storedSessionContext.serverInstanceId,
        sessionMode: 'resume',
    });
} else {
    client.connect({ sessionMode: 'new' });
}

// 5. Handle User Input
uiManager.onSend((text) => {
    // Browsers require user interaction to start AudioContext
    audioManager.init();
    
    uiManager.appendUserMessage(text);
    client.sendMessage(text);
});

async function refreshHistory() {
    uiManager.setHistoryLoading(true);
    uiManager.setHistoryStatus('');

    try {
        const payload = await client.listSessions();
        uiManager.renderHistorySessions(payload.sessions || [], currentSessionId);
    } catch (error) {
        console.error(error);
        uiManager.setHistoryStatus('Failed to load saved conversations.', 'error');
    } finally {
        uiManager.setHistoryLoading(false);
    }
}

uiManager.onHistoryRefresh(() => {
    refreshHistory();
});

uiManager.onHistoryOpen(async (sessionId) => {
    uiManager.setHistoryStatus('');

    try {
        const sessionData = await client.getSession(sessionId);
        currentSessionId = sessionId;
        if (currentServerInstanceId) {
            uiManager.setSessionScope(currentServerInstanceId, currentSessionId);
        }
        uiManager.renderSessionMessages(sessionData);
        persistSessionContext();
        client.switchSession({
            sessionId,
            sessionMode: 'open',
        });
        uiManager.setActiveTab('chat-view');
        refreshHistory();
    } catch (error) {
        console.error(error);
        uiManager.setHistoryStatus('Failed to open that conversation.', 'error');
    }
});

uiManager.onHistoryDelete(async (sessionId) => {
    uiManager.setHistoryStatus('');

    try {
        await client.deleteSession(sessionId);

        if (sessionId === currentSessionId) {
            clearSessionContext();
            client.switchSession({ sessionMode: 'new' });
        }

        await refreshHistory();
    } catch (error) {
        console.error(error);
        uiManager.setHistoryStatus('Failed to delete that conversation.', 'error');
    }
});

uiManager.onHistoryNewChat(async () => {
    uiManager.setHistoryStatus('');
    clearSessionContext();
    uiManager.resetChatToDefault();
    client.switchSession({ sessionMode: 'new' });
    uiManager.setActiveTab('chat-view');
    refreshHistory();
});

refreshHistory();
