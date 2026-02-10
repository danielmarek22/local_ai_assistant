import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

// --- GLOBAL STATE ---
let avatarState = "idle"; 
const poseTarget = { neckX: 0, neckY: 0, spineY: 0 };

// --- AUDIO SYSTEM ---
const audioQueue = [];
let isPlaying = false;
let audioContext = null;
let analyser = null;
let dataArray = null;
let audioSource = null;

// Create a single HTML Audio Element to reuse
const audioEl = new Audio();
audioEl.crossOrigin = "anonymous";

// Setup Audio Context (Must be triggered by user interaction first)
function initAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256; // Defines resolution of analysis
        dataArray = new Uint8Array(analyser.frequencyBinCount);
        
        // Connect Audio Element -> Analyser -> Speakers
        const track = audioContext.createMediaElementSource(audioEl);
        track.connect(analyser);
        analyser.connect(audioContext.destination);
        console.log("Audio Context Initialized");
    } else if (audioContext.state === 'suspended') {
        audioContext.resume();
    }
}

// --- 1. SETUP 3D SCENE ---
const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.getElementById('canvas-container').appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30.0, window.innerWidth / window.innerHeight, 0.1, 20.0);
camera.position.set(1.0, 1.0, 4.0); 

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.target.set(0.0, 0.9, 0.0);
controls.update();

const light = new THREE.DirectionalLight(0xffffff, 1.0);
light.position.set(1.0, 1.0, 1.0).normalize();
scene.add(light);
const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambientLight);

// --- 2. LOAD VRM MODEL ---
let currentVrm = null;
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

loader.load(
    '/static/avatar.vrm', 
    (gltf) => {
        const vrm = gltf.userData.vrm;
        VRMUtils.removeUnnecessaryVertices(gltf.scene);
        VRMUtils.removeUnnecessaryJoints(gltf.scene);
        currentVrm = vrm;
        scene.add(vrm.scene);
        // vrm.scene.rotation.y = Math.PI; 
        vrm.humanoid.getNormalizedBoneNode('leftUpperArm').rotation.z = -1.2; 
        vrm.humanoid.getNormalizedBoneNode('rightUpperArm').rotation.z = 1.2;
    },
    (progress) => {},
    (error) => console.error(error)
);

// --- 3. ANIMATION LOOP ---
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    const deltaTime = clock.getDelta();
    const time = clock.elapsedTime;
    controls.update(); 

    if (currentVrm) {
        currentVrm.update(deltaTime);
        
        // --- LIP SYNC LOGIC ---
        // If audio is playing and we have an analyser, get volume
        if (analyser && !audioEl.paused) {
            analyser.getByteFrequencyData(dataArray);
            
            // Calculate average volume (RMS-ish)
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
                sum += dataArray[i];
            }
            const average = sum / dataArray.length;
            
            // Map volume (0-255) to mouth open (0.0-1.0)
            // Sensitivity: divide by 80 to make it open easier
            let openValue = Math.min(1.0, average / 60); 
            
            // Smooth the mouth movement
            const currentMouth = currentVrm.expressionManager.getValue('aa');
            const smoothedMouth = THREE.MathUtils.lerp(currentMouth, openValue, 0.3);
            
            currentVrm.expressionManager.setValue('aa', smoothedMouth);
        } else {
            // Close mouth if silent
            currentVrm.expressionManager.setValue('aa', 0);
        }

        // --- STATE ANIMATIONS ---
        const neck = currentVrm.humanoid.getNormalizedBoneNode('neck');
        const chest = currentVrm.humanoid.getNormalizedBoneNode('chest');

        // Standard Breathing
        const breath = Math.sin(time * 1.0);
        chest.rotation.x = breath * 0.03;

        // Pose Smoothing
        neck.rotation.x = THREE.MathUtils.lerp(neck.rotation.x, poseTarget.neckX, 0.05);
        neck.rotation.y = THREE.MathUtils.lerp(neck.rotation.y, poseTarget.neckY, 0.05);
        currentVrm.humanoid.getNormalizedBoneNode('spine').rotation.y =
            THREE.MathUtils.lerp(
                currentVrm.humanoid.getNormalizedBoneNode('spine').rotation.y,
                poseTarget.spineY,
                0.05
            );

        // Idle Sway
        const sway = Math.sin(time * 0.5) * 0.02;
        neck.rotation.y += sway;
        
        // Blinking
        const blinkChance = (avatarState === "thinking" || avatarState === "searching") ? 0.002 : 0.005;
        if (Math.random() < blinkChance) blink();
    }
    renderer.render(scene, camera);
}
animate();

function blink() {
    if (!currentVrm) return;
    currentVrm.expressionManager.setValue('blink', 1.0);
    setTimeout(() => currentVrm.expressionManager.setValue('blink', 0.0), 150);
}

// --- 4. AUDIO QUEUE PLAYBACK ---
function playNextAudio() {
    if (isPlaying || audioQueue.length === 0) return;

    isPlaying = true;
    const audioUrl = audioQueue.shift(); // Get first URL

    // Load and Play
    audioEl.src = audioUrl;
    
    // Try/Catch for AutoPlay policy
    audioEl.play().catch(e => {
        console.error("Audio play failed (interaction needed?):", e);
        isPlaying = false;
    });

    audioEl.onended = () => {
        isPlaying = false;
        playNextAudio(); // Play next in queue
    };
}

// --- 5. UI HELPERS ---
const chatHistory = document.getElementById('chat-history');
const userInput = document.getElementById('user-input');
const chatPanel = document.getElementById('chat-panel');
const statusText = document.getElementById('status-text');
let currentAiMessageDiv = null; 

function updateState(state) {
    avatarState = state;
    chatPanel.classList.remove('status-idle', 'status-thinking', 'status-searching', 'status-responding');
    chatPanel.classList.add(`status-${state}`);

    if (state === 'idle') {
        statusText.innerText = "Astra is Idle";
        poseTarget.neckX = 0; poseTarget.neckY = 0; poseTarget.spineY = 0;
    }
    if (state === 'thinking') {
        statusText.innerText = "Astra is Thinking...";
        poseTarget.neckX = -0.25; poseTarget.neckY = 0.35; poseTarget.spineY = 0.05;
    }
    if (state === 'searching') {
        statusText.innerText = "Searching Knowledge Base...";
        poseTarget.neckX = -0.15; poseTarget.neckY = -0.4; poseTarget.spineY = -0.05;
    }
    if (state === 'responding') {
        statusText.innerText = "Astra is Responding";
        poseTarget.neckX = 0.05; poseTarget.neckY = 0; poseTarget.spineY = 0;
    }
}

function appendMessage(sender, text) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message');
    msgDiv.classList.add(sender === 'user' ? 'user' : 'astra');
    msgDiv.innerText = text;
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return msgDiv; 
}

// --- 6. WEBSOCKET CONNECTION ---
try {
    const ws = new WebSocket("ws://localhost:8000/ws");
    
    ws.onopen = () => { updateState('idle'); };

    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        // 1. STATE EVENT
        if (data.type === "assistant_state") {
            updateState(data.state);
            if (data.state === "responding" && !currentAiMessageDiv) {
                currentAiMessageDiv = appendMessage('astra', '');
            }
        } 
        
        // 2. CHUNK EVENT (Text)
        else if (data.type === "assistant_chunk") {
            if(avatarState !== 'responding') updateState('responding');
            if (!currentAiMessageDiv) currentAiMessageDiv = appendMessage('astra', '');
            
            currentAiMessageDiv.innerText += data.content;
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
        
        // 3. AUDIO EVENT (New!)
        else if (data.type === "assistant_audio") {
            // Add URL to queue and attempt playback
            // Ensure the URL matches your local path structure
            audioQueue.push(data.url);
            playNextAudio();
        }

        // 4. END EVENT
        else if (data.type === "assistant_end") {
            if (currentAiMessageDiv) currentAiMessageDiv.innerText = data.content;
            currentAiMessageDiv = null; 
            updateState('idle'); 
        }
    };

    function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        // INITIALIZE AUDIO ON FIRST INTERACTION
        initAudioContext();
        
        if (ws.readyState === WebSocket.OPEN) {
            appendMessage('user', text);
            ws.send(text);
            userInput.value = "";
        }
    }

    document.getElementById('send-btn').addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

} catch (e) { console.log(e); }