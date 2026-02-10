const chat = document.getElementById('chat-history');
const panel = document.getElementById('chat-panel');
const statusText = document.getElementById('status-text');

let currentAiMessage = null;

export function updateStateUI(state) {
    panel.className = `status-${state}`;
    statusText.innerText = `Astra is ${state}`;
}

export function appendMessage(sender, text, streaming=false, final=false) {
    if (sender === 'astra' && streaming) {
        if (!currentAiMessage) {
            currentAiMessage = createMessage('astra');
        }
        currentAiMessage.innerText += text;
    } else {
        createMessage(sender).innerText = text;
        if (final) currentAiMessage = null;
    }
    chat.scrollTop = chat.scrollHeight;
}

function createMessage(sender) {
    const div = document.createElement('div');
    div.className = `message ${sender}`;
    chat.appendChild(div);
    return div;
}
