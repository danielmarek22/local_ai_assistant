import { attachAnalyser } from './avatar.js';

const audioQueue = [];
const audioEl = new Audio();
let audioContext, analyser, dataArray;
let playing = false;

export function initAudio() {
    document.addEventListener('click', () => {
        if (!audioContext) {
            audioContext = new AudioContext();
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            dataArray = new Uint8Array(analyser.frequencyBinCount);

            const src = audioContext.createMediaElementSource(audioEl);
            src.connect(analyser);
            analyser.connect(audioContext.destination);

            attachAnalyser(analyser, dataArray);
        }
    }, { once: true });
}

export function enqueueAudio(url) {
    audioQueue.push(url);
    playNext();
}

function playNext() {
    if (playing || audioQueue.length === 0) return;
    playing = true;

    audioEl.src = audioQueue.shift();
    audioEl.play();

    audioEl.onended = () => {
        playing = false;
        playNext();
    };
}
