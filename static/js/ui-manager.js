import { CONFIG } from './config.js';
import { marked } from 'https://cdn.jsdelivr.net/npm/marked@13.0.2/lib/marked.esm.js';
import DOMPurify from 'https://cdn.jsdelivr.net/npm/dompurify@3.1.6/+esm';
import { extractBase64Payload, extractImageFilesFromDataTransfer, insertTextAtCursor, isImageFile } from './attachment-utils.mjs';

marked.setOptions({
    gfm: true,
    breaks: true
});

export class UIManager {
    constructor() {
        this.chatPanel = document.getElementById('chat-panel');
        this.statusText = document.getElementById('status-text');
        this.chatHistory = document.getElementById('chat-history');
        this.userInput = document.getElementById('user-input');
        this.imageInput = document.getElementById('image-input');
        this.attachmentPreview = document.getElementById('attachment-preview');
        this.composerMenu = document.getElementById('composer-menu');
        this.composerMenuBtn = document.getElementById('composer-menu-btn');
        this.composerMenuPopover = document.getElementById('composer-menu-popover');
        this.imageAttachBtn = document.getElementById('image-attach-btn');
        this.reasoningToggle = document.getElementById('reasoning-toggle');
        this.playbackVolumeInput = document.getElementById('playback-volume');
        this.playbackVolumeValue = document.getElementById('playback-volume-value');
        this.micHotkeyBtn = document.getElementById('mic-hotkey-btn');
        this.micHotkeyCurrent = document.getElementById('mic-hotkey-current');
        this.agentModeToggle = document.getElementById('agent-mode-toggle');
        this.hideThinkingToggle = document.getElementById('hide-thinking-toggle');
        this.voiceModeButtons = Array.from(document.querySelectorAll('[data-voice-mode]'));
        this.screenPolicyButtons = Array.from(document.querySelectorAll('[data-screen-policy]'));
        this.reflectNowBtn = document.getElementById('reflect-now-btn');
        this.reflectStatus = document.getElementById('reflect-status');
        this.sendBtn = document.getElementById('send-btn');
        this.conversationModeButtons = Array.from(document.querySelectorAll('[data-conversation-mode]'));
        this.relayComposer = document.getElementById('relay-composer');
        this.relaySenderName = document.getElementById('relay-sender-name');
        this.relaySenderType = document.getElementById('relay-sender-type');
        this.relayText = document.getElementById('relay-text');
        this.relayBtn = document.getElementById('relay-btn');
        this.chatCloseBtn = document.getElementById('chat-close-btn');
        this.chatOpenBtn = document.getElementById('chat-open-btn');
        this.micBtn = document.getElementById('mic-btn');
        this.chatTabs = Array.from(document.querySelectorAll('.chat-tab'));
        this.tabPanels = Array.from(document.querySelectorAll('.tab-panel'));
        this.historyList = document.getElementById('history-list');
        this.historyStatus = document.getElementById('history-status');
        this.historyRefreshBtn = document.getElementById('history-refresh-btn');
        this.historyNewChatBtn = document.getElementById('history-new-chat-btn');
        this.toolApprovalOverlay = document.getElementById('tool-approval-overlay');
        this.toolApprovalTitle = document.getElementById('tool-approval-title');
        this.toolApprovalTool = document.getElementById('tool-approval-tool');
        this.toolApprovalReason = document.getElementById('tool-approval-reason');
        this.toolApprovalDetailLabel = document.getElementById('tool-approval-detail-label');
        this.toolApprovalCommand = document.getElementById('tool-approval-command');
        this.toolApprovalApproveBtn = document.getElementById('tool-approval-approve');
        this.toolApprovalDenyBtn = document.getElementById('tool-approval-deny');

        this.currentAiMessageDiv = null;
        this.currentThinkingMessageDiv = null;
        this.chatHistoryStorageKey = null;
        this.currentSessionId = null;
        this.reasoningEnabledForNextSend = false;
        this.pendingAttachments = [];
        this.conversationMode = 'direct';
        this.localHumanDisplayName = 'You';
        this.localAssistantDisplayName = 'Astra';
        this.onConversationModeChangeHandler = null;
        this.onRelayHandler = null;
        this.volumeStorageKey = CONFIG.UI.STORAGE_KEYS.AUDIO_VOLUME;
        this.agentModeStorageKey = CONFIG.UI.STORAGE_KEYS.AGENT_MODE;
        this.hideThinkingStorageKey = CONFIG.UI.STORAGE_KEYS.HIDE_THINKING;
        this.agentModeEnabled = this.readStoredAgentMode();
        this.hideThinkingEnabled = this.readStoredHideThinking();
        this.instantModeEnabled = !this.agentModeEnabled;
        this.reasoningAlwaysStorageKey = CONFIG.UI.STORAGE_KEYS.REASONING_ALWAYS_ON;
        this.reasoningAlwaysEnabled = this.agentModeEnabled;
        this.voiceModeStorageKey = CONFIG.UI.STORAGE_KEYS.VOICE_MODE;
        this.voiceMode = this.readStoredVoiceMode();
        this.screenCapturePolicyStorageKey = CONFIG.UI.STORAGE_KEYS.SCREEN_CAPTURE_POLICY;
        this.screenCapturePolicy = this.readStoredScreenCapturePolicy();
        this.defaultMessages = this.serializeChatHistory();
        this.pendingToolApprovals = [];
        this.activeToolApproval = null;

        // STT recording state
        this._mediaRecorder = null;
        this._audioChunks = [];
        this._isRecording = false;
        this._isAlwaysListening = false;
        this._onMicPressHandler = null;
        this._micHotkey = this.loadMicHotkey();
        this._alwaysListeningStream = null;
        this._audioContext = null;
        this._alwaysListeningAnalyser = null;
        this._speechDetectedTime = 0;
        this._alwaysListeningSpeechCheck = null;
        this._isSendingAlwaysListeningChunk = false;
        this._alwaysListeningVoiceContextSent = false;

        this.initAutoResize();
        this.initPanelControls();
        this.initTabs();
        this.initHistoryControls();
        this.initComposerControls();
        this.initConversationControls();
        this.initConfigControls();
        this.initMicControls();
        this.initToolApprovalControls();
    }

    // ============================================================
    // Mic / STT
    // ============================================================

    initMicControls() {
        if (!navigator.mediaDevices?.getUserMedia) {
            // Browser doesn't support mic access — hide the button silently.
            this.micBtn?.classList.add('hidden');
            return;
        }

        // Load saved hotkey or use default
        this._micHotkey = this.loadMicHotkey();

        // Hotkey controls (spacebar by default) for manual recording
        document.addEventListener('keydown', (event) => {
            if (event.key !== this._micHotkey) return;
            // Don't trigger if user is typing in the input field
            if (document.activeElement === this.userInput) return;
            // Push-to-talk hotkey is disabled while automatic voice detection is active.
            if (this.voiceMode !== 'push_to_talk') return;
            if (this._isRecording) return;

            event.preventDefault();
            this.startManualRecording();
        });

        document.addEventListener('keyup', (event) => {
            if (event.key !== this._micHotkey) return;
            if (this.voiceMode === 'push_to_talk' && this._isRecording) {
                event.preventDefault();
                this.stopRecording();
            }
        });

        void this.applyVoiceMode(this.voiceMode);
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

            this.micBtn?.classList.add('always-listening');
            this.micBtn?.setAttribute('aria-label', 'Automatic voice detection is active');

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

                    if (!this._isAlwaysListening || !chunks.length || !this._onMicPressHandler) return;

                    const blob = new Blob(chunks, {
                        type: recorder.mimeType || 'audio/webm',
                    });
                    this._onMicPressHandler(blob, {
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
        this.micBtn?.classList.remove('always-listening');
        this.micBtn?.setAttribute('aria-label', 'Voice input');
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
        this.micBtn?.classList.add('recording');
        this.micBtn?.setAttribute('aria-label', 'Recording... release hotkey to send');

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
            this.micBtn?.classList.remove('recording');
            this.micBtn?.setAttribute('aria-label', 'Voice input');

            const blob = new Blob(this._audioChunks, {
                type: this._mediaRecorder.mimeType || 'audio/webm',
            });
            this._audioChunks = [];

            if (this._onMicPressHandler) {
                this._onMicPressHandler(blob, { includeScreenContext: true });
            }
        };

        this._mediaRecorder.start();
    }

    loadMicHotkey() {
        const stored = localStorage.getItem(CONFIG.UI.STORAGE_KEYS.MIC_HOTKEY);
        return stored || CONFIG.UI.MIC_HOTKEY_DEFAULT;
    }

    saveMicHotkey(hotkey) {
        this._micHotkey = hotkey;
        localStorage.setItem(CONFIG.UI.STORAGE_KEYS.MIC_HOTKEY, hotkey);
    }

    onMicPress(callback) {
        this._onMicPressHandler = callback;
    }

    onMicHotkeyChange(callback) {
        this._onMicHotkeyChangeHandler = callback;
    }

    initToolApprovalControls() {
        this.toolApprovalApproveBtn?.addEventListener('click', () => {
            this.resolveActiveToolApproval(true);
        });

        this.toolApprovalDenyBtn?.addEventListener('click', () => {
            this.resolveActiveToolApproval(false);
        });

        document.addEventListener('keydown', (event) => {
            if (event.key !== 'Escape') return;
            if (this.toolApprovalOverlay?.classList.contains('hidden')) return;

            event.preventDefault();
            this.resolveActiveToolApproval(false);
        });
    }

    showToolApprovalRequest(request, callback) {
        if (!request?.approvalId) return;

        this.pendingToolApprovals.push({
            approvalId: request.approvalId,
            tool: request.tool || 'unknown',
            title: request.title || 'Approve action?',
            reason: request.reason || 'This action requires human approval.',
            detailLabel: request.detailLabel || 'Details',
            detail: request.detail || '',
            callback,
        });

        if (!this.activeToolApproval) {
            this.renderNextToolApproval();
        }
    }

    renderNextToolApproval() {
        const nextRequest = this.pendingToolApprovals.shift();
        if (!nextRequest || !this.toolApprovalOverlay) return;

        this.activeToolApproval = nextRequest;

        if (this.toolApprovalTitle) {
            this.toolApprovalTitle.textContent = nextRequest.title;
        }

        if (this.toolApprovalTool) {
            this.toolApprovalTool.textContent = nextRequest.tool;
        }

        if (this.toolApprovalReason) {
            this.toolApprovalReason.textContent = nextRequest.reason;
        }

        if (this.toolApprovalDetailLabel) {
            this.toolApprovalDetailLabel.textContent = nextRequest.detailLabel;
        }

        if (this.toolApprovalCommand) {
            this.toolApprovalCommand.textContent = nextRequest.detail;
        }

        this.toolApprovalOverlay.classList.remove('hidden');
        this.toolApprovalDenyBtn?.focus();
    }

    resolveActiveToolApproval(approved) {
        const activeRequest = this.activeToolApproval;
        if (!activeRequest) return;

        this.activeToolApproval = null;
        this.toolApprovalOverlay?.classList.add('hidden');

        try {
            activeRequest.callback(activeRequest.approvalId, Boolean(approved));
        } catch (error) {
            console.warn('Failed to send tool approval decision:', error);
        }

        this.renderNextToolApproval();
    }

    async applyVoiceMode(mode = this.voiceMode) {
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

    formatKeyDisplay(key) {
        const keyMap = {
            ' ': 'Space',
            'Enter': 'Enter',
            'Tab': 'Tab',
            'Shift': 'Shift',
            'Control': 'Ctrl',
            'Alt': 'Alt',
            'Meta': 'Cmd',
            'ArrowUp': '↑',
            'ArrowDown': '↓',
            'ArrowLeft': '←',
            'ArrowRight': '→',
        };
        return keyMap[key] || (key.length === 1 ? key.toUpperCase() : key);
    }

    updateHotkeyDisplay() {
        if (this.micHotkeyCurrent) {
            this.micHotkeyCurrent.textContent = this.formatKeyDisplay(this._micHotkey);
        }
    }

    startHotkeyCapture() {
        if (!this.micHotkeyBtn) return;
        
        this.micHotkeyBtn.textContent = 'Listening...';
        this.micHotkeyBtn.disabled = true;
        this.micHotkeyBtn.classList.add('capturing');

        const handleKeyDown = (event) => {
            event.preventDefault();
            event.stopPropagation();

            const newHotkey = event.key;
            this.saveMicHotkey(newHotkey);
            this.updateHotkeyDisplay();

            this.micHotkeyBtn.textContent = 'Press any key...';
            this.micHotkeyBtn.disabled = false;
            this.micHotkeyBtn.classList.remove('capturing');

            document.removeEventListener('keydown', handleKeyDown);
        };

        document.addEventListener('keydown', handleKeyDown);

        // Timeout after 5 seconds
        setTimeout(() => {
            document.removeEventListener('keydown', handleKeyDown);
            if (this.micHotkeyBtn) {
                this.micHotkeyBtn.textContent = 'Press any key...';
                this.micHotkeyBtn.disabled = false;
                this.micHotkeyBtn.classList.remove('capturing');
            }
        }, 5000);
    }
    

    // Called by main.js when stt_transcript arrives from the server,
    // so the user bubble appears before the assistant starts responding.
    appendVoiceUserMessage(text, attachments = []) {
        this.currentThinkingMessageDiv = null;
        this.currentAiMessageDiv = null;
        const msgDiv = this.createMessageDiv('user', text, attachments, {
            senderDisplayName: this.localHumanDisplayName,
            senderType: 'human',
            inputSource: 'local_voice',
        });
        msgDiv.classList.add('from-stt');
    }

    // ============================================================
    // Existing methods — unchanged below this line
    // ============================================================

    initAutoResize() {
        this.userInput.addEventListener('input', () => {
            this.userInput.style.height = 'auto';
            this.userInput.style.height = this.userInput.scrollHeight + 'px';
        });
    }

    initPanelControls() {
        this.chatCloseBtn.addEventListener('click', () => {
            this.chatPanel.classList.add('hidden');
            this.chatOpenBtn.classList.remove('hidden');
        });

        this.chatOpenBtn.addEventListener('click', () => {
            this.chatPanel.classList.remove('hidden');
            this.chatOpenBtn.classList.add('hidden');
            this.userInput.focus();
        });

        this.chatOpenBtn.classList.add('hidden');
    }

    initTabs() {
        for (const tab of this.chatTabs) {
            tab.addEventListener('click', () => {
                this.setActiveTab(tab.dataset.tabTarget);
            });
        }
    }

    setActiveTab(tabId) {
        for (const tab of this.chatTabs) {
            const isActive = tab.dataset.tabTarget === tabId;
            tab.classList.toggle('active', isActive);
            tab.setAttribute('aria-selected', String(isActive));
        }

        for (const panel of this.tabPanels) {
            const isActive = panel.id === tabId;
            panel.classList.toggle('active', isActive);
            panel.setAttribute('aria-hidden', String(!isActive));
        }

        if (tabId === 'chat-view') {
            this.userInput.focus();
            this.scrollToBottom();
        }
    }

    initHistoryControls() {
        this.historyRefreshBtn.addEventListener('click', () => {
            if (this.onHistoryRefreshHandler) {
                this.onHistoryRefreshHandler();
            }
        });

        this.historyNewChatBtn.addEventListener('click', () => {
            if (this.onHistoryNewChatHandler) {
                this.onHistoryNewChatHandler();
            }
        });

        this.historyList.addEventListener('click', (event) => {
            const actionButton = event.target.closest('[data-history-action]');
            if (!actionButton) return;

            const { historyAction, sessionId } = actionButton.dataset;
            if (!sessionId) return;

            if (historyAction === 'open' && this.onHistoryOpenHandler) {
                this.onHistoryOpenHandler(sessionId);
            }

            if (historyAction === 'delete' && this.onHistoryDeleteHandler) {
                this.onHistoryDeleteHandler(sessionId);
            }
        });
    }

    initComposerControls() {
        this.composerMenuBtn.addEventListener('click', (event) => {
            event.stopPropagation();
            this.toggleComposerMenu();
        });

        this.imageAttachBtn?.addEventListener('click', () => {
            this.closeComposerMenu();
            this.imageInput?.click();
        });

        this.imageInput?.addEventListener('change', async (event) => {
            const files = Array.from(event.target.files || []);
            if (!files.length) return;

            await this.addPendingAttachments(files);
            event.target.value = '';
        });

        this.userInput.addEventListener('paste', (event) => {
            void this.handlePasteEvent(event);
        });

        this.attachmentPreview?.addEventListener('click', (event) => {
            const removeButton = event.target.closest('[data-attachment-remove]');
            if (!removeButton) return;

            this.removePendingAttachment(removeButton.dataset.attachmentRemove);
        });

        this.reasoningToggle.addEventListener('click', () => {
            if (this.reasoningAlwaysEnabled) return;

            this.reasoningEnabledForNextSend = !this.reasoningEnabledForNextSend;
            this.syncReasoningToggle();
            this.closeComposerMenu();
        });

        document.addEventListener('click', (event) => {
            if (this.composerMenu.contains(event.target)) {
                return;
            }

            this.closeComposerMenu();
        });
    }

    initConversationControls() {
        this.restoreLastRelaySender();
        for (const button of this.conversationModeButtons) {
            button.addEventListener('click', () => {
                const nextMode = button.dataset.conversationMode;
                if (!['direct', 'manual_group'].includes(nextMode) || nextMode === this.conversationMode) return;
                if (this.onConversationModeChangeHandler) {
                    this.onConversationModeChangeHandler(nextMode);
                }
            });
        }
        this.relayBtn?.addEventListener('click', () => {
            if (this.conversationMode !== 'manual_group' || !this.onRelayHandler) return;
            const senderDisplayName = this.relaySenderName.value.trim();
            const senderType = this.relaySenderType.value;
            const text = this.relayText.value.trim();
            if (!senderDisplayName || !text || !['human', 'external_agent'].includes(senderType)) return;
            this.persistLastRelaySender(senderDisplayName, senderType);
            this.onRelayHandler({ senderDisplayName, senderType, text });
            this.relayText.value = '';
            this.relayText.focus();
        });
    }

    setConversationMode(mode, {
        localHumanDisplayName = this.localHumanDisplayName,
        localAssistantDisplayName = this.localAssistantDisplayName,
    } = {}) {
        this.conversationMode = mode === 'manual_group' ? 'manual_group' : 'direct';
        this.localHumanDisplayName = localHumanDisplayName || 'You';
        this.localAssistantDisplayName = localAssistantDisplayName || 'Astra';
        for (const button of this.conversationModeButtons) {
            const active = button.dataset.conversationMode === this.conversationMode;
            button.classList.toggle('active', active);
            button.setAttribute('aria-checked', String(active));
        }
        this.relayComposer?.classList.toggle('hidden', this.conversationMode !== 'manual_group');
        this.chatHistory.classList.toggle('group-transcript', this.conversationMode === 'manual_group');
    }

    getConversationMode() { return this.conversationMode; }
    onConversationModeChange(callback) { this.onConversationModeChangeHandler = callback; }
    onRelay(callback) { this.onRelayHandler = callback; }

    restoreLastRelaySender() {
        try {
            const value = JSON.parse(localStorage.getItem(CONFIG.UI.STORAGE_KEYS.LAST_RELAY_SENDER) || 'null');
            if (value && typeof value.name === 'string') this.relaySenderName.value = value.name;
            if (value && ['human', 'external_agent'].includes(value.type)) this.relaySenderType.value = value.type;
        } catch (error) {
            localStorage.removeItem(CONFIG.UI.STORAGE_KEYS.LAST_RELAY_SENDER);
        }
    }

    persistLastRelaySender(name, type) {
        localStorage.setItem(CONFIG.UI.STORAGE_KEYS.LAST_RELAY_SENDER, JSON.stringify({ name, type }));
    }

    async handlePasteEvent(event) {
        const imageFiles = extractImageFilesFromDataTransfer(event.clipboardData);
        if (!imageFiles.length) {
            return;
        }

        const pastedText = event.clipboardData?.getData('text/plain') || '';
        event.preventDefault();

        if (pastedText) {
            insertTextAtCursor(this.userInput, pastedText);
            this.userInput.dispatchEvent(new Event('input', { bubbles: true }));
        }

        await this.addPendingAttachments(imageFiles);
    }

    async addPendingAttachments(files) {
        const nextAttachments = [];

        for (const file of files) {
            if (!isImageFile(file)) {
                continue;
            }

            try {
                nextAttachments.push(await this.readImageFile(file));
            } catch (error) {
                console.warn(`Failed to read image ${file.name}:`, error);
            }
        }

        if (!nextAttachments.length) return;

        this.pendingAttachments.push(...nextAttachments);
        this.renderPendingAttachments();
    }

    readImageFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = () => {
                const result = typeof reader.result === 'string' ? reader.result : '';
                const base64Data = extractBase64Payload(result);
                if (!base64Data) {
                    reject(new Error('Image did not produce base64 data'));
                    return;
                }

                resolve({
                    id: this.createAttachmentId(),
                    name: file.name,
                    mimeType: file.type,
                    size: file.size,
                    data: base64Data,
                });
            };

            reader.onerror = () => {
                reject(reader.error || new Error('FileReader failed'));
            };

            reader.readAsDataURL(file);
        });
    }

    createAttachmentId() {
        if (window.crypto?.randomUUID) {
            return window.crypto.randomUUID();
        }

        return `attachment-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    removePendingAttachment(attachmentId) {
        if (!attachmentId) return;

        this.pendingAttachments = this.pendingAttachments.filter((attachment) => attachment.id !== attachmentId);
        this.renderPendingAttachments();
    }

    clearPendingAttachments() {
        this.pendingAttachments = [];
        this.renderPendingAttachments();
        if (this.imageInput) {
            this.imageInput.value = '';
        }
    }

    renderPendingAttachments() {
        if (!this.attachmentPreview) return;

        this.attachmentPreview.replaceChildren();
        this.attachmentPreview.classList.toggle('hidden', this.pendingAttachments.length === 0);

        for (const attachment of this.pendingAttachments) {
            const chip = this.createAttachmentNode(attachment, { removable: true });
            this.attachmentPreview.appendChild(chip);
        }
    }

    initConfigControls() {
        if (this.playbackVolumeInput && this.playbackVolumeValue) {
            this.setPlaybackVolume(this.readStoredPlaybackVolume(), { persist: false, notify: false });

            this.playbackVolumeInput.addEventListener('input', () => {
                const nextVolume = Number(this.playbackVolumeInput.value) / 100;
                this.setPlaybackVolume(nextVolume, { persist: true, notify: true });
            });
        }

        if (this.micHotkeyBtn && this.micHotkeyCurrent) {
            this.updateHotkeyDisplay();
            this.micHotkeyBtn.addEventListener('click', () => this.startHotkeyCapture());
        }

        this.setAgentMode(this.agentModeEnabled, { persist: false });
        this.agentModeToggle?.addEventListener('click', () => {
            this.setAgentMode(!this.agentModeEnabled);
        });

        this.setHideThinking(this.hideThinkingEnabled, { persist: false });
        this.hideThinkingToggle?.addEventListener('click', () => {
            this.setHideThinking(!this.hideThinkingEnabled);
        });

        this.setVoiceMode(this.voiceMode, { persist: false, activate: false });
        for (const button of this.voiceModeButtons) {
            button.addEventListener('click', () => {
                this.setVoiceMode(button.dataset.voiceMode, {
                    persist: true,
                    activate: true,
                });
            });
        }

        this.setScreenCapturePolicy(this.screenCapturePolicy, { persist: false, notify: false });
        for (const button of this.screenPolicyButtons) {
            button.addEventListener('click', () => {
                this.setScreenCapturePolicy(button.dataset.screenPolicy, {
                    persist: true,
                    notify: true,
                });
            });
        }

        this.reflectNowBtn?.addEventListener('click', () => {
            if (this.onReflectHandler) {
                this.onReflectHandler();
            }
        });

        this.setReflectRunning(false);
        this.setReflectStatus('');
    }

    readStoredPlaybackVolume() {
        try {
            const rawValue = localStorage.getItem(this.volumeStorageKey);
            if (rawValue === null) {
                return CONFIG.AUDIO.DEFAULT_VOLUME;
            }

            const parsedValue = Number(rawValue);
            if (Number.isFinite(parsedValue) && parsedValue >= 0 && parsedValue <= 1) {
                return parsedValue;
            }

            localStorage.removeItem(this.volumeStorageKey);
        } catch (error) {
            console.warn('Failed to restore playback volume:', error);
        }

        return CONFIG.AUDIO.DEFAULT_VOLUME;
    }

    setPlaybackVolume(volume, { persist = true, notify = true } = {}) {
        const nextVolume = Number.isFinite(volume)
            ? Math.min(1, Math.max(0, volume))
            : CONFIG.AUDIO.DEFAULT_VOLUME;
        const percentage = Math.round(nextVolume * 100);

        if (this.playbackVolumeInput) {
            this.playbackVolumeInput.value = String(percentage);
        }
        if (this.playbackVolumeValue) {
            this.playbackVolumeValue.textContent = `${percentage}%`;
        }

        if (persist) {
            try {
                localStorage.setItem(this.volumeStorageKey, String(nextVolume));
            } catch (error) {
                console.warn('Failed to persist playback volume:', error);
            }
        }

        if (notify && this.onVolumeChangeHandler) {
            this.onVolumeChangeHandler(nextVolume);
        }
    }

    getPlaybackVolume() {
        if (!this.playbackVolumeInput) {
            return CONFIG.AUDIO.DEFAULT_VOLUME;
        }

        return Number(this.playbackVolumeInput.value) / 100;
    }

    readStoredAgentMode() {
        try {
            const storedValue = localStorage.getItem(this.agentModeStorageKey);
            if (storedValue !== null) {
                return storedValue === 'true';
            }

            const legacyValue = localStorage.getItem(CONFIG.UI.STORAGE_KEYS.INSTANT_MODE);
            if (legacyValue !== null) {
                const migratedAgentMode = legacyValue === 'false';
                localStorage.setItem(this.agentModeStorageKey, String(migratedAgentMode));
                localStorage.removeItem(CONFIG.UI.STORAGE_KEYS.INSTANT_MODE);
                return migratedAgentMode;
            }

            return Boolean(CONFIG.UI.AGENT_MODE_DEFAULT);
        } catch (error) {
            console.warn('Failed to restore agent mode:', error);
            return Boolean(CONFIG.UI.AGENT_MODE_DEFAULT);
        }
    }

    readStoredHideThinking() {
        try {
            const storedValue = localStorage.getItem(this.hideThinkingStorageKey);
            if (storedValue === null) {
                return Boolean(CONFIG.UI.HIDE_THINKING_DEFAULT);
            }
            return storedValue === 'true';
        } catch (error) {
            console.warn('Failed to restore hide thinking setting:', error);
            return Boolean(CONFIG.UI.HIDE_THINKING_DEFAULT);
        }
    }

    setAgentMode(isEnabled, { persist = true } = {}) {
        this.agentModeEnabled = Boolean(isEnabled);
        this.instantModeEnabled = !this.agentModeEnabled;
        this.reasoningAlwaysEnabled = this.agentModeEnabled;

        if (this.agentModeToggle) {
            this.agentModeToggle.textContent = this.agentModeEnabled ? 'On' : 'Off';
            this.agentModeToggle.classList.toggle('active', this.agentModeEnabled);
            this.agentModeToggle.setAttribute('aria-pressed', String(this.agentModeEnabled));
        }

        if (this.reasoningAlwaysEnabled) {
            this.reasoningEnabledForNextSend = false;
            if (this.currentThinkingMessageDiv) {
                this.currentThinkingMessageDiv.remove();
                this.currentThinkingMessageDiv = null;
                this.persistChatHistory();
            }
        }

        this.syncReasoningToggle();

        if (persist) {
            try {
                localStorage.setItem(this.agentModeStorageKey, String(this.agentModeEnabled));
            } catch (error) {
                console.warn('Failed to persist agent mode:', error);
            }
        }
    }

    setHideThinking(isEnabled, { persist = true } = {}) {
        this.hideThinkingEnabled = Boolean(isEnabled);

        if (this.hideThinkingEnabled && this.currentThinkingMessageDiv) {
            this.currentThinkingMessageDiv.remove();
            this.currentThinkingMessageDiv = null;
            this.persistChatHistory();
        }

        if (this.hideThinkingToggle) {
            this.hideThinkingToggle.textContent = this.hideThinkingEnabled ? 'On' : 'Off';
            this.hideThinkingToggle.classList.toggle('active', this.hideThinkingEnabled);
            this.hideThinkingToggle.setAttribute('aria-pressed', String(this.hideThinkingEnabled));
        }

        if (persist) {
            try {
                localStorage.setItem(this.hideThinkingStorageKey, String(this.hideThinkingEnabled));
            } catch (error) {
                console.warn('Failed to persist hide thinking setting:', error);
            }
        }
    }

    isAgentModeEnabled() {
        return Boolean(this.agentModeEnabled);
    }

    isHideThinkingEnabled() {
        return Boolean(this.hideThinkingEnabled);
    }

    readStoredInstantMode() {
        return !this.readStoredAgentMode();
    }

    setInstantMode(isEnabled, { persist = true } = {}) {
        this.setAgentMode(!Boolean(isEnabled), { persist });
    }

    isInstantModeEnabled() {
        return !this.isAgentModeEnabled();
    }

    readStoredReasoningAlways() {
        return this.readStoredAgentMode();
    }

    setReasoningAlways(isEnabled, { persist = true } = {}) {
        this.setAgentMode(Boolean(isEnabled), { persist });
    }

    isReasoningAlwaysEnabled() {
        return this.isAgentModeEnabled();
    }

    shouldUseReasoningForSend() {
        return this.isReasoningAlwaysEnabled() || this.reasoningEnabledForNextSend;
    }

    normalizeVoiceMode(mode) {
        return CONFIG.UI.VOICE_MODES.includes(mode)
            ? mode
            : CONFIG.UI.VOICE_MODE_DEFAULT;
    }

    readStoredVoiceMode() {
        try {
            const rawValue = localStorage.getItem(this.voiceModeStorageKey);
            const mode = this.normalizeVoiceMode(rawValue);
            if (rawValue !== null && rawValue !== mode) {
                localStorage.removeItem(this.voiceModeStorageKey);
            }
            return mode;
        } catch (error) {
            console.warn('Failed to restore voice mode:', error);
            return CONFIG.UI.VOICE_MODE_DEFAULT;
        }
    }

    setVoiceMode(mode, { persist = true, activate = true } = {}) {
        const nextMode = this.normalizeVoiceMode(mode);
        this.voiceMode = nextMode;

        for (const button of this.voiceModeButtons) {
            const isActive = button.dataset.voiceMode === nextMode;
            button.classList.toggle('active', isActive);
            button.setAttribute('aria-checked', String(isActive));
        }

        if (persist) {
            try {
                localStorage.setItem(this.voiceModeStorageKey, nextMode);
            } catch (error) {
                console.warn('Failed to persist voice mode:', error);
            }
        }

        if (activate) {
            void this.applyVoiceMode(nextMode);
        }
    }

    normalizeScreenCapturePolicy(policy) {
        return CONFIG.UI.SCREEN_CAPTURE_POLICIES.includes(policy)
            ? policy
            : CONFIG.UI.SCREEN_CAPTURE_POLICY_DEFAULT;
    }

    readStoredScreenCapturePolicy() {
        try {
            const rawValue = localStorage.getItem(this.screenCapturePolicyStorageKey);
            const policy = this.normalizeScreenCapturePolicy(rawValue);
            if (rawValue !== null && rawValue !== policy) {
                localStorage.removeItem(this.screenCapturePolicyStorageKey);
            }
            return policy;
        } catch (error) {
            console.warn('Failed to restore screen capture policy:', error);
            return CONFIG.UI.SCREEN_CAPTURE_POLICY_DEFAULT;
        }
    }

    setScreenCapturePolicy(policy, { persist = true, notify = true } = {}) {
        const nextPolicy = this.normalizeScreenCapturePolicy(policy);
        this.screenCapturePolicy = nextPolicy;

        for (const button of this.screenPolicyButtons) {
            const isActive = button.dataset.screenPolicy === nextPolicy;
            button.classList.toggle('active', isActive);
            button.setAttribute('aria-checked', String(isActive));
        }

        if (persist) {
            try {
                localStorage.setItem(this.screenCapturePolicyStorageKey, nextPolicy);
            } catch (error) {
                console.warn('Failed to persist screen capture policy:', error);
            }
        }

        if (notify && this.onScreenCapturePolicyChangeHandler) {
            this.onScreenCapturePolicyChangeHandler(nextPolicy);
        }
    }

    getScreenCapturePolicy() {
        return this.screenCapturePolicy || CONFIG.UI.SCREEN_CAPTURE_POLICY_DEFAULT;
    }

    onScreenCapturePolicyChange(callback) {
        this.onScreenCapturePolicyChangeHandler = callback;
    }

    syncReasoningToggle() {
        const isReasoningAlwaysOn = this.isReasoningAlwaysEnabled();
        this.composerMenuBtn.classList.toggle('active', this.reasoningEnabledForNextSend || isReasoningAlwaysOn);
        this.reasoningToggle.classList.toggle('active', this.reasoningEnabledForNextSend || isReasoningAlwaysOn);
        this.reasoningToggle.classList.toggle('disabled', isReasoningAlwaysOn);
        this.reasoningToggle.disabled = isReasoningAlwaysOn;
        this.reasoningToggle.setAttribute('aria-disabled', String(isReasoningAlwaysOn));
        this.reasoningToggle.setAttribute('aria-checked', String(this.reasoningEnabledForNextSend || isReasoningAlwaysOn));
    }

    toggleComposerMenu() {
        const shouldOpen = this.composerMenuPopover.classList.contains('hidden');
        this.composerMenuPopover.classList.toggle('hidden', !shouldOpen);
        this.composerMenuBtn.setAttribute('aria-expanded', String(shouldOpen));
    }

    closeComposerMenu() {
        this.composerMenuPopover.classList.add('hidden');
        this.composerMenuBtn.setAttribute('aria-expanded', 'false');
    }

    updateStatus(state) {
        this.chatPanel.classList.remove(...Object.keys(CONFIG.UI.STATUS_TEXT).map((status) => `status-${status}`));
        this.chatPanel.classList.add(`status-${state}`);

        const text = CONFIG.UI.STATUS_TEXT[state] || CONFIG.UI.STATUS_TEXT.idle;
        this.statusText.innerText = text;
    }

    appendUserMessage(text, attachments = []) {
        this.currentThinkingMessageDiv = null;
        this.currentAiMessageDiv = null;
        this.createMessageDiv('user', text, attachments, {
            senderDisplayName: this.localHumanDisplayName,
            senderType: 'human',
            inputSource: 'local_text',
        });
    }

    appendRelayMessage(text, senderDisplayName, senderType) {
        this.currentThinkingMessageDiv = null;
        this.currentAiMessageDiv = null;
        this.createMessageDiv('user', text, [], {
            senderDisplayName,
            senderType,
            inputSource: 'manual_relay',
        });
    }

    addNoticeToLastUserMessage(text, tone = 'warning') {
        const noticeText = typeof text === 'string' ? text.trim() : '';
        if (!noticeText) return;

        const lastUserMessage = this.findLastUserMessage();
        if (!lastUserMessage) return;

        let notice = lastUserMessage.querySelector('.message-notice');
        if (!notice) {
            notice = document.createElement('div');
            notice.className = 'message-notice';
            lastUserMessage.prepend(notice);
        }

        notice.textContent = noticeText;
        notice.classList.toggle('warning', tone === 'warning');
        notice.classList.toggle('info', tone !== 'warning');

        this.persistChatHistory();
        this.scrollToBottom();
    }

    findLastUserMessage() {
        const messages = this.chatHistory.querySelectorAll('.message.user');
        if (!messages.length) return null;
        return messages[messages.length - 1];
    }

    startAiMessage() {
        if (!this.currentAiMessageDiv) {
            this.currentAiMessageDiv = this.createMessageDiv('astra', '', [], {
                senderDisplayName: this.localAssistantDisplayName,
                senderType: 'local_assistant',
                inputSource: 'assistant_generation',
            });
        }
    }

    startThinkingMessage() {
        if (!this.currentThinkingMessageDiv) {
            this.currentThinkingMessageDiv = this.createMessageDiv('thinking', '');
        }
    }

    appendToAiMessage(text) {
        if (!this.currentAiMessageDiv) this.startAiMessage();
        const rawText = (this.currentAiMessageDiv.dataset.rawText || '') + text;
        this.setMessageContent(this.currentAiMessageDiv, rawText);
        this.persistChatHistory();
        this.scrollToBottom();
    }

    appendToThinkingMessage(text) {
        if (!text) return;
        if (this.isHideThinkingEnabled()) return;

        if (!this.currentThinkingMessageDiv) this.startThinkingMessage();
        const rawText = (this.currentThinkingMessageDiv.dataset.rawText || '') + text;
        this.setMessageContent(this.currentThinkingMessageDiv, rawText);
        this.persistChatHistory();
        this.scrollToBottom();
    }

    finalizeAiMessage(text) {
        if (!this.currentAiMessageDiv && text) {
            this.currentAiMessageDiv = this.createMessageDiv('astra', '', [], {
                senderDisplayName: this.localAssistantDisplayName,
                senderType: 'local_assistant',
                inputSource: 'assistant_generation',
            });
        }

        if (this.currentAiMessageDiv) {
            this.setMessageContent(this.currentAiMessageDiv, text);
            this.persistChatHistory();
            this.currentAiMessageDiv = null;
        }
    }

    finalizeThinkingMessage() {
        this.currentThinkingMessageDiv = null;
    }

    createMessageDiv(sender, text, attachments = [], metadata = {}) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', sender);
        this.setMessageMetadata(msgDiv, metadata);
        this.setMessageContent(msgDiv, text, attachments);
        this.chatHistory.appendChild(msgDiv);
        this.persistChatHistory();
        this.scrollToBottom();
        return msgDiv;
    }

    setMessageContent(msgDiv, text, attachments = []) {
        const normalizedAttachments = this.normalizeAttachments(attachments);
        msgDiv.dataset.rawText = text;

        if (normalizedAttachments.length) {
            msgDiv.dataset.attachments = JSON.stringify(normalizedAttachments.map((attachment) => ({
                id: attachment.id,
                name: attachment.name,
                mimeType: attachment.mimeType,
                data: attachment.data,
                url: attachment.url,
                size: attachment.size,
            })));
        } else {
            delete msgDiv.dataset.attachments;
        }

        if (msgDiv.classList.contains('astra') || msgDiv.classList.contains('thinking')) {
            const unsafeHtml = marked.parse(text);
            const safeHtml = DOMPurify.sanitize(unsafeHtml, {
                USE_PROFILES: { html: true }
            });
            msgDiv.innerHTML = safeHtml;

            for (const link of msgDiv.querySelectorAll('a')) {
                link.setAttribute('target', '_blank');
                link.setAttribute('rel', 'noopener noreferrer');
            }
            this.prependSenderHeader(msgDiv);
            return;
        }

        msgDiv.replaceChildren();

        if (text) {
            const body = document.createElement('div');
            body.className = 'message-body';
            body.innerText = text;
            msgDiv.appendChild(body);
        }

        if (normalizedAttachments.length) {
            const gallery = document.createElement('div');
            gallery.className = 'message-attachments';

            for (const attachment of normalizedAttachments) {
                gallery.appendChild(this.createAttachmentNode(attachment));
            }

            msgDiv.appendChild(gallery);
        }
        this.prependSenderHeader(msgDiv);
    }

    setMessageMetadata(msgDiv, metadata = {}) {
        const senderType = metadata.senderType || '';
        const inputSource = metadata.inputSource || '';
        msgDiv.dataset.senderId = metadata.senderId || '';
        msgDiv.dataset.senderDisplayName = metadata.senderDisplayName || '';
        msgDiv.dataset.senderType = senderType;
        msgDiv.dataset.inputSource = inputSource;
        const controlledTypes = ['human', 'external_agent', 'local_assistant', 'system', 'tool', 'integration_runtime'];
        const controlledSources = ['local_text', 'local_voice', 'manual_relay', 'assistant_generation', 'system_runtime', 'tool_runtime', 'integration_runtime'];
        if (controlledTypes.includes(senderType)) msgDiv.classList.add(`sender-${senderType.replaceAll('_', '-')}`);
        if (controlledSources.includes(inputSource)) msgDiv.classList.add(`source-${inputSource.replaceAll('_', '-')}`);
        if (this.conversationMode === 'manual_group') msgDiv.classList.add('group-message');
    }

    prependSenderHeader(msgDiv) {
        if (this.conversationMode !== 'manual_group') return;
        const existing = msgDiv.querySelector('.message-sender-header');
        if (existing) existing.remove();
        const header = document.createElement('div');
        header.className = 'message-sender-header';
        header.textContent = msgDiv.dataset.senderDisplayName || this.fallbackSenderLabel(msgDiv);
        msgDiv.prepend(header);
    }

    fallbackSenderLabel(msgDiv) {
        if (msgDiv.classList.contains('astra')) return this.localAssistantDisplayName;
        if (msgDiv.classList.contains('thinking')) return `${this.localAssistantDisplayName} thinking`;
        if (msgDiv.classList.contains('system')) return 'System';
        if (msgDiv.classList.contains('tool')) return 'Tool';
        return this.localHumanDisplayName;
    }

    normalizeAttachments(attachments) {
        if (!Array.isArray(attachments)) {
            return [];
        }

        return attachments
            .map((attachment) => this.normalizeAttachment(attachment))
            .filter(Boolean);
    }

    normalizeAttachment(attachment) {
        if (!attachment || typeof attachment !== 'object') {
            return null;
        }

        const data = typeof attachment.data === 'string' ? attachment.data.trim() : '';
        const url = typeof attachment.url === 'string' ? attachment.url.trim() : '';
        if (!data && !url) {
            return null;
        }

        const mimeType = typeof attachment.mimeType === 'string'
            ? attachment.mimeType
            : typeof attachment.mime_type === 'string'
                ? attachment.mime_type
                : 'image/png';
        if (!mimeType.startsWith('image/')) {
            return null;
        }

        const size = Number.isFinite(attachment.size)
            ? attachment.size
            : Number.isFinite(attachment.size_bytes)
                ? attachment.size_bytes
                : null;

        return {
            id: typeof attachment.id === 'string' && attachment.id ? attachment.id : this.createAttachmentId(),
            name: typeof attachment.name === 'string' && attachment.name.trim() ? attachment.name.trim() : 'image',
            mimeType,
            data: data || null,
            url: url || null,
            size,
        };
    }

    createAttachmentNode(attachment, { removable = false } = {}) {
        const node = document.createElement('figure');
        node.className = removable ? 'attachment-chip' : 'message-attachment';

        const image = document.createElement('img');
        image.className = removable ? 'attachment-chip-image' : 'message-attachment-image';
        image.src = this.buildAttachmentSrc(attachment);
        image.alt = attachment.name;
        node.appendChild(image);

        const caption = document.createElement('figcaption');
        caption.className = removable ? 'attachment-chip-meta' : 'message-attachment-meta';
        caption.textContent = this.formatAttachmentLabel(attachment.name, removable ? 18 : 28);
        node.appendChild(caption);

        if (removable) {
            const removeButton = document.createElement('button');
            removeButton.type = 'button';
            removeButton.className = 'attachment-chip-remove';
            removeButton.dataset.attachmentRemove = attachment.id;
            removeButton.setAttribute('aria-label', `Remove ${attachment.name}`);
            removeButton.textContent = 'x';
            node.appendChild(removeButton);
        }

        return node;
    }

    buildAttachmentSrc(attachment) {
        if (attachment.url) {
            return attachment.url;
        }

        return `data:${attachment.mimeType};base64,${attachment.data}`;
    }

    formatAttachmentLabel(name, limit = 24) {
        if (name.length <= limit) {
            return name;
        }

        return `${name.slice(0, Math.max(0, limit - 1))}…`;
    }

    setSessionScope(serverInstanceId, sessionId) {
        const nextStorageKey = `${CONFIG.UI.STORAGE_KEYS.CHAT_HISTORY}:${serverInstanceId}:${sessionId}`;
        if (this.chatHistoryStorageKey === nextStorageKey) return;

        this.currentSessionId = sessionId;
        this.chatHistoryStorageKey = nextStorageKey;
        this.currentAiMessageDiv = null;
        this.currentThinkingMessageDiv = null;
        this.restoreChatHistory();
    }

    restoreChatHistory() {
        if (!this.chatHistoryStorageKey) return;

        const savedHistory = sessionStorage.getItem(this.chatHistoryStorageKey);
        if (!savedHistory) {
            this.renderMessages(this.defaultMessages);
            this.persistChatHistory();
            return;
        }

        try {
            const messages = JSON.parse(savedHistory);
            if (!Array.isArray(messages) || messages.length === 0) {
                this.renderMessages(this.defaultMessages);
                this.persistChatHistory();
                return;
            }
            this.renderMessages(messages);
        } catch (error) {
            console.warn('Failed to restore chat history from session storage:', error);
            sessionStorage.removeItem(this.chatHistoryStorageKey);
            this.renderMessages(this.defaultMessages);
            this.persistChatHistory();
        }
    }

    persistChatHistory() {
        if (!this.chatHistoryStorageKey) return;

        try {
            sessionStorage.setItem(this.chatHistoryStorageKey, JSON.stringify(this.serializeChatHistory()));
        } catch (error) {
            console.warn('Failed to persist chat history:', error);
        }
    }

    serializeChatHistory() {
        return Array.from(this.chatHistory.querySelectorAll('.message')).map((message) => {
            const sender = Array.from(message.classList).find((className) => className !== 'message') || 'astra';
            return {
                sender,
                text: message.dataset.rawText || '',
                attachments: this.readStoredAttachments(message.dataset.attachments),
                senderId: message.dataset.senderId || '',
                senderDisplayName: message.dataset.senderDisplayName || '',
                senderType: message.dataset.senderType || '',
                inputSource: message.dataset.inputSource || '',
            };
        });
    }

    readStoredAttachments(rawValue) {
        if (!rawValue) {
            return [];
        }

        try {
            const parsed = JSON.parse(rawValue);
            return this.normalizeAttachments(parsed).map((attachment) => ({
                name: attachment.name,
                mimeType: attachment.mimeType,
                data: attachment.data,
                url: attachment.url,
                size: attachment.size,
            }));
        } catch (error) {
            console.warn('Failed to parse stored attachments:', error);
            return [];
        }
    }

    renderMessages(messages) {
        this.chatHistory.replaceChildren();

        for (const message of messages) {
            if (!message || typeof message.sender !== 'string' || typeof message.text !== 'string') {
                continue;
            }

            const msgDiv = document.createElement('div');
            msgDiv.classList.add('message', message.sender);
            this.setMessageMetadata(msgDiv, message);
            this.setMessageContent(msgDiv, message.text, message.attachments || []);
            this.chatHistory.appendChild(msgDiv);
        }

        this.scrollToBottom();
    }

    renderSessionMessages(sessionData) {
        this.setConversationMode(sessionData.kind || 'direct');
        const messages = sessionData.messages.map((message) => ({
            sender: message.role === 'assistant' ? 'astra' : message.role,
            text: message.content,
            attachments: message.attachments || [],
            senderId: message.sender_id,
            senderDisplayName: message.sender_display_name,
            senderType: message.sender_type,
            inputSource: message.input_source,
        }));

        this.currentAiMessageDiv = null;
        this.currentThinkingMessageDiv = null;
        this.renderMessages(messages);
        this.persistChatHistory();
    }

    resetChatToDefault() {
        this.currentAiMessageDiv = null;
        this.currentThinkingMessageDiv = null;
        this.clearPendingAttachments();
        this.renderMessages(this.defaultMessages);
        this.persistChatHistory();
    }

    setHistoryLoading(isLoading) {
        this.historyRefreshBtn.disabled = isLoading;
        this.historyRefreshBtn.textContent = isLoading ? 'Refreshing...' : 'Refresh';
        this.historyNewChatBtn.disabled = isLoading;
    }

    setHistoryStatus(message = '', tone = 'info') {
        this.historyStatus.textContent = message;
        this.historyStatus.classList.toggle('hidden', !message);
        this.historyStatus.classList.toggle('error', tone === 'error');
    }

    renderHistorySessions(sessions, activeSessionId) {
        this.historyList.replaceChildren();

        if (!sessions.length) {
            const emptyState = document.createElement('div');
            emptyState.className = 'history-empty';
            emptyState.textContent = 'No saved conversations yet.';
            this.historyList.appendChild(emptyState);
            return;
        }

        for (const session of sessions) {
            const item = document.createElement('article');
            item.className = 'history-item';
            if (session.session_id === activeSessionId) {
                item.classList.add('active');
            }

            const preview = session.preview || 'No preview available.';
            const title = this.buildSessionTitle(session);
            const updatedAt = this.formatTimestamp(session.updated_at);
            const itemMeta = `${session.message_count} messages • ${updatedAt}`;

            item.innerHTML = `
                <div class="history-item-header">
                    <div>
                        <div class="history-item-title">${this.escapeHtml(title)}</div>
                        <div class="history-item-meta">${this.escapeHtml(itemMeta)}</div>
                    </div>
                </div>
                <div class="history-item-preview">${this.escapeHtml(preview)}</div>
                <div class="history-item-actions">
                    <button class="history-action-btn" type="button" data-history-action="open" data-session-id="${this.escapeHtml(session.session_id)}">Open</button>
                    <button class="history-action-btn delete" type="button" data-history-action="delete" data-session-id="${this.escapeHtml(session.session_id)}">Delete</button>
                </div>
            `;

            this.historyList.appendChild(item);
        }
    }

    buildSessionTitle(session) {
        if (session.preview) {
            return session.preview.slice(0, 48);
        }

        return `Session ${session.session_id}`;
    }

    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        if (Number.isNaN(date.getTime())) {
            return timestamp;
        }

        return date.toLocaleString();
    }

    escapeHtml(value) {
        return String(value)
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#39;');
    }

    onHistoryRefresh(callback) {
        this.onHistoryRefreshHandler = callback;
    }

    onHistoryOpen(callback) {
        this.onHistoryOpenHandler = callback;
    }

    onHistoryDelete(callback) {
        this.onHistoryDeleteHandler = callback;
    }

    onHistoryNewChat(callback) {
        this.onHistoryNewChatHandler = callback;
    }

    onVolumeChange(callback) {
        this.onVolumeChangeHandler = callback;
        callback(this.getPlaybackVolume());
    }

    setReflectRunning(isRunning) {
        if (!this.reflectNowBtn) return;

        this.reflectNowBtn.disabled = isRunning;
        this.reflectNowBtn.textContent = isRunning ? 'Dreaming...' : 'Run Reflection';
    }

    setReflectStatus(message = '', tone = 'info') {
        if (!this.reflectStatus) return;

        const hasMessage = Boolean(message);
        this.reflectStatus.textContent = message;
        this.reflectStatus.classList.toggle('hidden', !hasMessage);
        this.reflectStatus.classList.toggle('success', hasMessage && tone === 'success');
        this.reflectStatus.classList.toggle('error', hasMessage && tone === 'error');
    }

    onReflect(callback) {
        this.onReflectHandler = callback;
    }

    scrollToBottom() {
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
    }

    onSend(callback) {
        const handler = (event) => {
            if (event && event.type === 'keypress') {
                event.preventDefault();
            }

            const text = this.userInput.value.trim();
            const attachments = this.pendingAttachments.map((attachment) => ({
                name: attachment.name,
                mimeType: attachment.mimeType,
                data: attachment.data,
                url: attachment.url,
                size: attachment.size,
            }));
            if (!text && attachments.length === 0) return;

            const sendOptions = {
                reasoning: this.shouldUseReasoningForSend(),
                instantMode: !this.isAgentModeEnabled(),
                attachments,
            };

            callback(text, sendOptions);

            this.reasoningEnabledForNextSend = false;
            this.syncReasoningToggle();
            this.closeComposerMenu();
            this.clearPendingAttachments();

            this.userInput.value = '';
            this.userInput.style.height = 'auto';
            this.userInput.focus();
        };

        this.sendBtn.addEventListener('click', () => handler());

        this.userInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                handler(event);
            }
        });
    }
}
