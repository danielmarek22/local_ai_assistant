import { initAvatar, setAvatarState } from './avatar.js';
import { initAudio, enqueueAudio } from './audio.js';
import { updateStateUI, appendMessage } from './ui.js';
import { connectWebSocket } from './transport.js';

initAvatar();
initAudio();

connectWebSocket({
    onState: (state) => {
        setAvatarState(state);
        updateStateUI(state);
    },
    onChunk: (text) => {
        appendMessage('astra', text, true);
    },
    onEnd: (text) => {
        appendMessage('astra', text, false, true);
        setAvatarState('idle');
        updateStateUI('idle');
    },
    onAudio: (url) => {
        enqueueAudio(url);
    }
});
