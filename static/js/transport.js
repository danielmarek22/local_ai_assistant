export function connectWebSocket(handlers) {
    const ws = new WebSocket("ws://localhost:8000/ws");

    ws.onmessage = e => {
        const msg = JSON.parse(e.data);

        if (msg.type === "assistant_state") handlers.onState(msg.state);
        if (msg.type === "assistant_chunk") handlers.onChunk(msg.content);
        if (msg.type === "assistant_audio") handlers.onAudio(msg.url);
        if (msg.type === "assistant_end") handlers.onEnd(msg.content);
    };

    const input = document.getElementById('user-input');
    document.getElementById('send-btn').onclick = () => send(input.value);

    function send(text) {
        if (!text) return;
        handlers.onChunk('');
        ws.send(text);
        input.value = "";
    }
}
