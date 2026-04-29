import { AudioManager } from './js/audio-manager.js';
import { AvatarManager } from './js/avatar-manager.js';
import { UIManager } from './js/ui-manager.js';
import { NetworkClient } from './js/network-client.js';
import { CONFIG } from './js/config.js';

const audioManager = new AudioManager();
const uiManager = new UIManager();
const sessionStorageKey = CONFIG.UI.STORAGE_KEYS.CURRENT_SESSION;
let currentServerInstanceId = null;
let currentSessionId = null;
let assistantState = 'idle';
let assistantExpression = 'neutral';
let reflectionInFlight = false;

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

const avatarManager = new AvatarManager(
    'canvas-container',
    () => audioManager.getVisemeData()
);

function syncAssistantPresentation() {
    const baseState = reflectionInFlight ? 'dreaming' : assistantState;
    const visualState = audioManager.hasActiveSpeech() ? 'responding' : baseState;
    uiManager.updateStatus(visualState);
    avatarManager.setState(visualState);
    avatarManager.setExpression(assistantExpression);
}

function setReflectionInFlight(isInFlight) {
    reflectionInFlight = isInFlight;
    uiManager.setReflectRunning(isInFlight);
    syncAssistantPresentation();
}

audioManager.setPlaybackHandlers({
    onSpeechStart: () => {
        syncAssistantPresentation();
    },
    onSpeechEnd: () => {
        syncAssistantPresentation();
    }
});

uiManager.onVolumeChange((volume) => {
    audioManager.setVolume(volume);
});

const handlers = {
    onSessionInit: ({ serverInstanceId, sessionId, gestureCatalog }) => {
        currentServerInstanceId = serverInstanceId;
        currentSessionId = sessionId;
        uiManager.setSessionScope(serverInstanceId, sessionId);
        avatarManager.setGestureCatalog(gestureCatalog || {});
        persistSessionContext();
    },
    onState: (state) => {
        assistantState = state;
        syncAssistantPresentation();
    },
    onExpression: (expression) => {
        assistantExpression = expression;
        syncAssistantPresentation();
    },
    onAnimation: (animation) => {
        avatarManager.queueGesture(animation);
    },
    onThinkingChunk: (content) => {
        uiManager.appendToThinkingMessage(content);
    },
    onChunk: (content) => {
        uiManager.appendToAiMessage(content);
    },
    onAudio: (url) => {
        audioManager.queueAudio(url);
    },
    onUserNotice: (payload) => {
        if (payload?.scope === 'last_user_message' && typeof payload?.message === 'string') {
            uiManager.addNoticeToLastUserMessage(payload.message, payload?.tone || 'warning');
        }
    },
    onEnd: (finalContent) => {
        uiManager.finalizeThinkingMessage();
        uiManager.finalizeAiMessage(finalContent);
        assistantState = 'idle';
        syncAssistantPresentation();
    },

    // ── STT handlers ─────────────────────────────────────────────
    // The server echoes the transcript back before it starts processing
    // the turn, so we render the user bubble here rather than on send.
    onSttTranscript: ({ text }) => {
        audioManager.init();
        uiManager.appendVoiceUserMessage(text);
    },
    // Silence detected — nothing to do in the UI, but the hook is here
    // if you want to add a visual cue later (e.g. a brief mic flash).
    onSttSilence: () => {},
};

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

uiManager.onSend((text, options) => {
    audioManager.init();

    uiManager.appendUserMessage(text, options.attachments || []);
    client.sendMessage(text, options);
});

// Wire mic button → binary WS frame.
// The user bubble is rendered by onSttTranscript above, not here,
// because we don't have the transcript text yet at send time.
uiManager.onMicPress((audioBlob) => {
    console.log('onMicPress fired, blob size:', audioBlob.size, 'type:', audioBlob.type);
    client.sendAudio(audioBlob);
});


uiManager.onReflect(async () => {
    if (reflectionInFlight) {
        return;
    }

    uiManager.setReflectStatus('');
    setReflectionInFlight(true);

    try {
        const result = await client.reflectMemories(0);
        uiManager.setReflectStatus(
            `Dream complete: ${result.deleted_count} deleted, ${result.created_count} created.`,
            'success',
        );
    } catch (error) {
        console.error(error);
        uiManager.setReflectStatus(error?.message || 'Memory reflection failed.', 'error');
    } finally {
        await avatarManager.playDreamingOutro();
        setReflectionInFlight(false);
    }
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
