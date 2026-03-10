import { CONFIG } from './config.js';

export class NetworkClient {
    constructor(handlers) {
        this.handlers = handlers; // Expects: onSessionInit, onState, onExpression, onChunk, onAudio, onEnd
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

            console.log(`Connecting to ${wsUrl.toString()}...`);
            const socket = new WebSocket(wsUrl.toString());
            this.ws = socket;
            
            socket.onopen = () => {
                if (this.ws !== socket) return;
                console.log("WS Connected");
                // Clear any pending reconnects
                if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
                
                if(this.handlers.onState) this.handlers.onState('idle');
            };

            socket.onclose = (event) => {
                if (this.ws !== socket) return;
                if (this.isExplicitlyClosed) return;
                
                console.warn(`WS Closed (Code: ${event.code}). Reconnecting in ${CONFIG.SYSTEM.RECONNECT_INTERVAL_MS}ms...`);
                this.scheduleReconnect();
            };

            socket.onerror = (err) => {
                if (this.ws !== socket) return;
                console.error("WS Error encountered. Closing socket to trigger reconnect.", err);
                socket.close();
            };

            socket.onmessage = (event) => {
                if (this.ws !== socket) return;
                const data = JSON.parse(event.data);
                
                if (data.type === "session_init" && this.handlers.onSessionInit) {
                    this.handlers.onSessionInit({
                        serverInstanceId: data.server_instance_id,
                        sessionId: data.session_id,
                    });
                }
                else if (data.type === "assistant_state" && this.handlers.onState) {
                    this.handlers.onState(data.state);
                } 
                else if (data.type === "assistant_expression" && this.handlers.onExpression) {
                    this.handlers.onExpression(data.expression);
                }
                else if (data.type === "assistant_chunk" && this.handlers.onChunk) {
                    this.handlers.onChunk(data.content);
                }
                else if (data.type === "assistant_audio" && this.handlers.onAudio) {
                    this.handlers.onAudio(data.url);
                }
                else if (data.type === "assistant_end" && this.handlers.onEnd) {
                    this.handlers.onEnd(data.content);
                }
            };

        } catch (e) {
            console.error("WS Connection Setup Failed:", e);
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

    sendMessage(text) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(text);
        } else {
            console.warn("Cannot send message: WebSocket is not open.");
        }
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
}
