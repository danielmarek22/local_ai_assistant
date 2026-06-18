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
let screenStream = null;
let webcamStream = null;
let visionCaptureTimer = null;

const screenShareToggle = document.getElementById('screen-share-toggle');
const webcamToggle = document.getElementById('webcam-toggle');
const screenVideo = document.getElementById('screen-capture-video');
const screenCanvas = document.getElementById('screen-capture-canvas');
const webcamVideo = document.getElementById('webcam-capture-video');
const webcamCanvas = document.getElementById('webcam-capture-canvas');

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

const pendingVoiceAttachmentBatches = [];

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

uiManager.onScreenCapturePolicyChange(() => {
    syncVisionCaptureLoop();
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
        const attachments = pendingVoiceAttachmentBatches.shift() || [];
        uiManager.appendVoiceUserMessage(text, attachments);
    },
    // Silence detected — nothing to do in the UI, but the hook is here
    // if you want to add a visual cue later (e.g. a brief mic flash).
    onSttSilence: () => {
        pendingVoiceAttachmentBatches.shift();
    },
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

function isStreamActive(stream) {
    return Boolean(stream?.getTracks().some((track) => track.readyState === 'live'));
}

function syncVisionToggle(button, isEnabled, enabledLabel, disabledLabel) {
    if (!button) return;
    button.textContent = isEnabled ? enabledLabel : disabledLabel;
    button.setAttribute('aria-pressed', String(isEnabled));
    button.classList.toggle('active', isEnabled);
}

function isScreenWatchdogPolicy() {
    return uiManager.getScreenCapturePolicy() === 'watchdog';
}

function shouldRunVisionCaptureLoop() {
    return isStreamActive(webcamStream) || (isStreamActive(screenStream) && isScreenWatchdogPolicy());
}

function stopVisionCaptureLoop() {
    if (!visionCaptureTimer) return;

    window.clearInterval(visionCaptureTimer);
    visionCaptureTimer = null;
}

function syncVisionCaptureLoop() {
    if (shouldRunVisionCaptureLoop()) {
        ensureVisionCaptureLoop();
    } else {
        stopVisionCaptureLoop();
    }
}

function ensureVisionCaptureLoop() {
    if (visionCaptureTimer) return;

    visionCaptureTimer = window.setInterval(() => {
        if (isStreamActive(screenStream) && isScreenWatchdogPolicy()) {
            sendVideoFrame({
                video: screenVideo,
                canvas: screenCanvas,
                type: 'screen_frame',
                name: 'screen.jpg',
                maxLongEdge: 960,
            });
        }

        if (isStreamActive(webcamStream)) {
            sendVideoFrame({
                video: webcamVideo,
                canvas: webcamCanvas,
                type: 'webcam_frame',
                name: 'webcam.jpg',
                maxLongEdge: 640,
            });
        }

        syncVisionCaptureLoop();
    }, 1500);
}

function captureVideoFrame(video, canvas, maxLongEdge) {
    if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        return null;
    }

    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;
    if (!sourceWidth || !sourceHeight) {
        return null;
    }

    const scale = Math.min(1, maxLongEdge / Math.max(sourceWidth, sourceHeight));
    canvas.width = Math.max(1, Math.round(sourceWidth * scale));
    canvas.height = Math.max(1, Math.round(sourceHeight * scale));

    const context = canvas.getContext('2d');
    if (!context) return null;

    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.5);
    const commaIndex = dataUrl.indexOf(',');
    return commaIndex >= 0 ? dataUrl.slice(commaIndex + 1) : null;
}

function sendVideoFrame({ video, canvas, type, name, maxLongEdge }) {
    const base64Data = captureVideoFrame(video, canvas, maxLongEdge);
    if (!base64Data) return;

    client.sendVisionFrame(type, {
        mime_type: 'image/jpeg',
        base64_data: base64Data,
        name,
    });
}

/**
 * Synchronously capture fresh frames from active streams and bundle as attachment objects.
 * @returns {Array<{name: string, mimeType: string, data: string}>} Array of attachment objects
 */
function captureAndBundleFrames({ includeScreen = true, includeWebcam = true } = {}) {
    const attachments = [];

    try {
        if (includeScreen && isStreamActive(screenStream)) {
            const base64Data = captureVideoFrame(screenVideo, screenCanvas, 960);
            if (base64Data) {
                attachments.push({
                    name: 'sync_screen.jpg',
                    mimeType: 'image/jpeg',
                    data: base64Data,
                });
            }
        }

        if (includeWebcam && isStreamActive(webcamStream)) {
            const base64Data = captureVideoFrame(webcamVideo, webcamCanvas, 640);
            if (base64Data) {
                attachments.push({
                    name: 'sync_webcam.jpg',
                    mimeType: 'image/jpeg',
                    data: base64Data,
                });
            }
        }
    } catch (error) {
        console.warn('Failed to capture and bundle frames:', error);
    }

    return attachments;
}

function stopStream(stream) {
    if (!stream) return;
    stream.getTracks().forEach((track) => track.stop());
}

async function enableScreenShare() {
    if (!navigator.mediaDevices?.getDisplayMedia || !screenVideo) return;

    try {
        screenStream = await navigator.mediaDevices.getDisplayMedia({ video: true });
        screenVideo.srcObject = screenStream;
        await screenVideo.play();
        screenStream.getVideoTracks().forEach((track) => {
            track.addEventListener('ended', disableScreenShare, { once: true });
        });
        syncVisionToggle(screenShareToggle, true, 'Disable Screen Share', 'Enable Screen Share');
        syncVisionCaptureLoop();
    } catch (error) {
        console.warn('Screen share permission was not granted:', error);
        disableScreenShare();
    }
}

function disableScreenShare() {
    stopStream(screenStream);
    screenStream = null;
    if (screenVideo) screenVideo.srcObject = null;
    syncVisionToggle(screenShareToggle, false, 'Disable Screen Share', 'Enable Screen Share');
    syncVisionCaptureLoop();
}

async function enableWebcam() {
    if (!navigator.mediaDevices?.getUserMedia || !webcamVideo) return;

    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcamVideo.srcObject = webcamStream;
        await webcamVideo.play();
        webcamStream.getVideoTracks().forEach((track) => {
            track.addEventListener('ended', disableWebcam, { once: true });
        });
        syncVisionToggle(webcamToggle, true, 'Disable Webcam', 'Enable Webcam');
        syncVisionCaptureLoop();
    } catch (error) {
        console.warn('Webcam permission was not granted:', error);
        disableWebcam();
    }
}

function disableWebcam() {
    stopStream(webcamStream);
    webcamStream = null;
    if (webcamVideo) webcamVideo.srcObject = null;
    syncVisionToggle(webcamToggle, false, 'Disable Webcam', 'Enable Webcam');
    syncVisionCaptureLoop();
}

screenShareToggle?.addEventListener('click', () => {
    if (isStreamActive(screenStream)) {
        disableScreenShare();
    } else {
        void enableScreenShare();
    }
});

webcamToggle?.addEventListener('click', () => {
    if (isStreamActive(webcamStream)) {
        disableWebcam();
    } else {
        void enableWebcam();
    }
});

window.addEventListener('beforeunload', () => {
    disableScreenShare();
    disableWebcam();
});

uiManager.onSend((text, options) => {
    audioManager.init();

    // Keep screen capture policy tied to voice/watchdog routing. Webcam
    // context keeps the existing automatic attachment behavior.
    const syncFrames = captureAndBundleFrames({
        includeScreen: false,
        includeWebcam: true,
    });
    const allAttachments = [
        ...syncFrames,
        ...(options.attachments || [])
    ];
    options.attachments = allAttachments;

    uiManager.appendUserMessage(text, options.attachments || []);
    client.sendMessage(text, options);
});

// Wire mic button → binary WS frame.
// The user bubble is rendered by onSttTranscript above, not here,
// because we don't have the transcript text yet at send time.
uiManager.onMicPress((audioBlob, options = {}) => {
    console.log('onMicPress fired, blob size:', audioBlob.size, 'type:', audioBlob.type);

    // Capture fresh frames and send them before audio (as separate JSON messages)
    const includeScreenContext = options.includeScreenContext !== false;
    const syncFrames = captureAndBundleFrames({
        includeScreen: isScreenWatchdogPolicy() ? false : includeScreenContext,
        includeWebcam: true,
    });

    syncFrames.forEach(attachment => {
        try {
            client.sendVisionFrame(
                'user_attached_frame',
                {
                    mime_type: attachment.mimeType,
                    base64_data: attachment.data,
                    name: attachment.name,
                }
            );
        } catch (error) {
            console.warn('Failed to send bundled frame before audio:', error);
        }
    });
    pendingVoiceAttachmentBatches.push(syncFrames);

    client.sendUserConfig({
        instantMode: uiManager.isInstantModeEnabled(),
        reasoning: uiManager.isReasoningAlwaysEnabled() ? true : null,
    });

    // Now send the audio blob as binary
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
