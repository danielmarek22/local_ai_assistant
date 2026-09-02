import { CONFIG } from './config.js';

export function buildKnowledgeBeliefsUrl(filters = {}) {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(filters)) {
        if (value === undefined || value === null || value === '') continue;
        params.set(key, String(value));
    }
    const query = params.toString();
    return `/api/knowledge/beliefs${query ? `?${query}` : ''}`;
}

export class NetworkClient {
    constructor(handlers) {
        this.handlers = handlers; // Expects: onSessionInit, onState, onExpression, onAnimation, onOutfit, onThinkingChunk, onChunk, onAudio, onEnd
        this.ws = null;
        this.reconnectTimer = null;
        this.isExplicitlyClosed = false;
        this.connectionOptions = {};
    }

    connect(options = this.connectionOptions) {
        this.isExplicitlyClosed = false;
        this.connectionOptions = { ...options };

        try {
            const wsUrl = new URL(CONFIG.SYSTEM.WS_URL, window.location.href);
            if (this.connectionOptions.sessionId) {
                wsUrl.searchParams.set('session_id', this.connectionOptions.sessionId);
            }
            if (this.connectionOptions.serverInstanceId) {
                wsUrl.searchParams.set('server_instance_id', this.connectionOptions.serverInstanceId);
            }
            if (this.connectionOptions.sessionMode) {
                wsUrl.searchParams.set('session_mode', this.connectionOptions.sessionMode);
            }
            if (this.connectionOptions.sessionKind) {
                wsUrl.searchParams.set('session_kind', this.connectionOptions.sessionKind);
            }

            console.log(`Connecting to ${wsUrl.toString()}...`);
            const socket = new WebSocket(wsUrl.toString());
            this.ws = socket;

            socket.onopen = () => {
                if (this.ws !== socket) return;
                console.log('WS Connected');
                if (this.reconnectTimer) clearTimeout(this.reconnectTimer);

                if (this.handlers.onState) this.handlers.onState('idle');
            };

            socket.onclose = (event) => {
                if (this.ws !== socket) return;
                if (this.isExplicitlyClosed) return;

                console.warn(`WS Closed (Code: ${event.code}). Reconnecting in ${CONFIG.SYSTEM.RECONNECT_INTERVAL_MS}ms...`);
                this.scheduleReconnect();
            };

            socket.onerror = (err) => {
                if (this.ws !== socket) return;
                console.error('WS Error encountered. Closing socket to trigger reconnect.', err);
                socket.close();
            };

            socket.onmessage = (event) => {
                if (this.ws !== socket) return;
                const data = JSON.parse(event.data);

                if (data.type === 'session_init' && this.handlers.onSessionInit) {
                    this.handlers.onSessionInit({
                        serverInstanceId: data.server_instance_id,
                        sessionId: data.session_id,
                        gestureCatalog: data.gesture_catalog || {},
                        outfitCatalog: data.outfit_catalog || {},
                        currentOutfit: data.current_outfit || null,
                        sessionKind: data.session_kind || 'direct',
                        localHumanDisplayName: data.local_human_display_name || 'You',
                        localAssistantDisplayName: data.local_assistant_display_name || 'Astra',
                    });
                }
                else if (data.type === 'assistant_state' && this.handlers.onState) {
                    this.handlers.onState(data.state, data.turn_id || null);
                }
                else if (data.type === 'assistant_expression' && this.handlers.onExpression) {
                    this.handlers.onExpression(data.expression);
                }
                else if (data.type === 'assistant_animation' && this.handlers.onAnimation) {
                    this.handlers.onAnimation(data.animation, data.turn_id || null);
                }
                else if (data.type === 'assistant_outfit' && this.handlers.onOutfit) {
                    this.handlers.onOutfit({ outfit: data.outfit, url: data.url });
                }
                else if (data.type === 'assistant_thinking_chunk' && this.handlers.onThinkingChunk) {
                    this.handlers.onThinkingChunk(data.content);
                }
                else if (data.type === 'assistant_chunk' && this.handlers.onChunk) {
                    this.handlers.onChunk(data.content);
                }
                else if (data.type === 'assistant_audio' && this.handlers.onAudio) {
                    this.handlers.onAudio(data.url);
                }
                else if (data.type === 'assistant_end' && this.handlers.onEnd) {
                    this.handlers.onEnd(data.content);
                }
                else if (data.type === 'user_message_accepted' && this.handlers.onUserMessageAccepted) {
                    this.handlers.onUserMessageAccepted({
                        messageId: data.message_id,
                        isRetry: Boolean(data.is_retry),
                    });
                }
                else if (data.type === 'assistant_retryable_error' && this.handlers.onRetryableError) {
                    this.handlers.onRetryableError({
                        userMessageId: data.user_message_id,
                        message: data.message,
                        attempts: data.attempts,
                    });
                }
                else if (data.type === 'user_notice' && this.handlers.onUserNotice) {
                    this.handlers.onUserNotice(data);
                }
                else if (data.type === 'tool_approval_request' && this.handlers.onToolApprovalRequest) {
                    this.handlers.onToolApprovalRequest({
                        approvalId: data.approval_id,
                        tool: data.tool,
                        title: data.title,
                        reason: data.reason,
                        detailLabel: data.detail_label,
                        detail: data.detail,
                        timeoutSeconds: data.timeout_seconds,
                    });
                }
                else if (data.type === 'stt_transcript' && this.handlers.onSttTranscript) {
                    this.handlers.onSttTranscript({
                        text: data.text,
                        language: data.language,
                    });
                }
                else if (data.type === 'stt_silence' && this.handlers.onSttSilence) {
                    this.handlers.onSttSilence();
                }
            };

        } catch (e) {
            console.error('WS Connection Setup Failed:', e);
            this.scheduleReconnect();
        }
    }

    scheduleReconnect() {
        if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
        this.reconnectTimer = setTimeout(() => {
            this.connect(this.connectionOptions);
        }, CONFIG.SYSTEM.RECONNECT_INTERVAL_MS);
    }

    switchSession(options = {}) {
        this.isExplicitlyClosed = true;

        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        const previousSocket = this.ws;
        this.ws = null;

        if (previousSocket) {
            try {
                previousSocket.close();
            } catch (error) {
                console.warn('Failed to close previous websocket:', error);
            }
        }

        this.connect(options);
    }

    sendAudio(audioBlob) {
        this.ws.send(audioBlob);  // binary frame — no JSON wrapping
    }

    sendVisionFrame(type, attachment) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type,
                attachment,
            }));
        }
    }

    sendUserConfig(options = {}) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            const payload = {
                type: 'user_config',
                instant_mode: Boolean(options.instantMode),
            };

            if (Object.prototype.hasOwnProperty.call(options, 'reasoning')) {
                payload.reasoning = options.reasoning === null
                    ? null
                    : Boolean(options.reasoning);
            }

            this.ws.send(JSON.stringify(payload));
        }
    }

    sendToolApproval(approvalId, approved) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'tool_approval_response',
                approval_id: approvalId,
                approved: Boolean(approved),
            }));
        }
    }

    sendMessage(text, options = {}) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            const attachments = Array.isArray(options.attachments)
                ? options.attachments
                    .filter((attachment) => attachment && typeof attachment.data === 'string')
                    .map((attachment) => {
                        const payload = {
                            name: attachment.name || 'image',
                            mime_type: attachment.mimeType || attachment.mime_type || 'image/png',
                            data: attachment.data,
                        };

                        if (Number.isFinite(attachment.size)) {
                            payload.size_bytes = attachment.size;
                        }

                        return payload;
                    })
                : [];

            this.ws.send(JSON.stringify({
                type: 'user_message',
                text,
                reasoning: Boolean(options.reasoning),
                instant_mode: Boolean(options.instantMode),
                attachments,
            }));
        } else {
            console.warn('Cannot send message: WebSocket is not open.');
        }
    }

    sendRetry(messageId, options = {}) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.warn('Cannot retry message: WebSocket is not open.');
            return false;
        }
        this.ws.send(JSON.stringify({
            type: 'retry_message',
            message_id: Number(messageId),
            reasoning: Boolean(options.reasoning),
            instant_mode: Boolean(options.instantMode),
        }));
        return true;
    }

    sendRelayMessage(senderDisplayName, senderType, text) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.warn('Cannot relay message: WebSocket is not open.');
            return false;
        }
        this.ws.send(JSON.stringify({
            type: 'relay_message',
            sender_display_name: senderDisplayName,
            sender_type: senderType,
            text,
        }));
        return true;
    }

    async listSessions() {
        const response = await fetch('/api/sessions');
        if (!response.ok) {
            throw new Error(`Failed to load sessions (${response.status})`);
        }

        return response.json();
    }

    async getSession(sessionId) {
        const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
        if (!response.ok) {
            throw new Error(`Failed to load session (${response.status})`);
        }

        return response.json();
    }

    async deleteSession(sessionId) {
        const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
            method: 'DELETE'
        });
        if (!response.ok) {
            throw new Error(`Failed to delete session (${response.status})`);
        }

        return response.json();
    }

    async getEffectiveBeliefs(sessionId) {
        const params = new URLSearchParams({ session_id: sessionId });
        return this._getJson(`/api/knowledge/beliefs/effective?${params.toString()}`);
    }

    async listBeliefs(filters = {}) {
        return this._getJson(buildKnowledgeBeliefsUrl(filters));
    }

    async getBelief(beliefId) {
        return this._getJson(`/api/knowledge/beliefs/${encodeURIComponent(beliefId)}`);
    }

    async getBeliefContext(sessionId) {
        const params = new URLSearchParams({ session_id: sessionId });
        return this._getJson(`/api/knowledge/belief-context?${params.toString()}`);
    }

    async getSavedMemories() {
        return this._getJson('/api/knowledge/memories');
    }

    async _getJson(url) {
        const response = await fetch(url);
        if (!response.ok) {
            const error = new Error(`Request failed (${response.status})`);
            error.status = response.status;
            throw error;
        }
        return response.json();
    }

    async getAutonomyStatus() {
        const response = await fetch('/api/autonomy');
        if (!response.ok) throw new Error(`Failed to load autonomy status (${response.status})`);
        return response.json();
    }

    async setAutonomyPaused(paused) {
        const response = await fetch('/api/autonomy', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ paused: Boolean(paused) }),
        });
        if (!response.ok) throw new Error(`Failed to update autonomy status (${response.status})`);
        return response.json();
    }

    async reflectMemories(daysOld = 0) {
        const response = await fetch('/api/admin/reflect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                days_old: daysOld,
            }),
        });

        if (!response.ok) {
            let detail = `Failed to run reflection (${response.status})`;

            try {
                const payload = await response.json();
                const message = payload?.detail?.message;
                const error = payload?.detail?.error;
                if (message && error) {
                    detail = `${message}: ${error}`;
                } else if (message) {
                    detail = message;
                }
            } catch (_error) {
                // Keep default message when response is non-JSON.
            }

            throw new Error(detail);
        }

        return response.json();
    }
}
