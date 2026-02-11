import { CONFIG } from './config.js';

export class NetworkClient {
    constructor(handlers) {
        this.handlers = handlers; // Expects: onState, onChunk, onAudio, onEnd
        this.ws = null;
        this.reconnectTimer = null;
        this.isExplicitlyClosed = false;
    }

    connect() {
        this.isExplicitlyClosed = false;
        
        try {
            console.log(`Connecting to ${CONFIG.SYSTEM.WS_URL}...`);
            this.ws = new WebSocket(CONFIG.SYSTEM.WS_URL);
            
            this.ws.onopen = () => {
                console.log("WS Connected");
                // Clear any pending reconnects
                if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
                
                if(this.handlers.onState) this.handlers.onState('idle');
            };

            this.ws.onclose = (event) => {
                if (this.isExplicitlyClosed) return;
                
                console.warn(`WS Closed (Code: ${event.code}). Reconnecting in ${CONFIG.SYSTEM.RECONNECT_INTERVAL_MS}ms...`);
                this.scheduleReconnect();
            };

            this.ws.onerror = (err) => {
                console.error("WS Error encountered. Closing socket to trigger reconnect.", err);
                this.ws.close();
            };

            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === "assistant_state" && this.handlers.onState) {
                    this.handlers.onState(data.state);
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
            this.connect();
        }, CONFIG.SYSTEM.RECONNECT_INTERVAL_MS);
    }

    sendMessage(text) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(text);
        } else {
            console.warn("Cannot send message: WebSocket is not open.");
        }
    }
}