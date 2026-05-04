import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { CONFIG } from './config.js';

// Import the Mixamo loader helper 
import { loadMixamoAnimation } from './loadMixamoAnimation.js'; 

export class AvatarManager {
    constructor(containerId, getAudioLevelCallback) {
        this.container = document.getElementById(containerId);
        this.getAudioLevel = getAudioLevelCallback; 
        
        this.currentVrm = null;
        this.currentState = "idle";
        this.currentExpression = "neutral";
        
        // Animation Mixer & Storage
        this.mixer = null;
        this.animations = {}; 
        this.currentAction = null;
        this.stateAnimations = {}; 
        this.gestureCatalog = {};
        this.gestureAnimations = {};
        this.gestureQueue = [];
        this.isGesturePlaying = false;
        this.activeGestureName = null;
        this.dreamingPhase = null; // intro | holding | outro
        this.pendingDreamingOutroStart = false;
        this.dreamingOutroPromise = null;
        this.dreamingOutroResolver = null;
        this.dreamingOutroTimeoutId = null;
        this.forceEyesClosed = false;
        
        // Blink State Management
        this.blinkState = 'open'; 
        this.blinkTimer = 0;
        this.nextBlinkTime = Math.random() * 3 + 2; 

        // --- NEW: Eye Tracking State Management ---
        this.lookAtTarget = new THREE.Object3D(); 
        this.lookAtOffset = new THREE.Vector3(0, 0, 0); // How far away from the camera to look
        this.eyeTimer = 0;
        this.nextEyeMoveTime = Math.random() * 3 + 2;
        this.isLookingAtCamera = true;

        this.clock = new THREE.Clock();
        
        this.initScene();
        this.initLoader();
        
        this.animate = this.animate.bind(this);
        this.animate();
    }

    initScene() {
        this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(30.0, window.innerWidth / window.innerHeight, 0.1, 20.0);
        
        // --- UPDATED: Move the camera HIGHER, near head-level (1.8) ---
        this.camera.position.set(0.0, 1.6, 3.2); 

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        // --- UPDATED: Keep the look-target LOWER, near waist-level (0.8) ---
        // This forces the camera to angle downward.
        this.controls.target.set(0.0, 0.8, 0.0); 
        this.controls.update();

        // --- NEW: Add the invisible eye target to the scene ---
        this.scene.add(this.lookAtTarget);

        const light = new THREE.DirectionalLight(0xffffff, 1.0);
        light.position.set(1.0, 1.0, 1.0).normalize();
        this.scene.add(light);
        
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambientLight);
    }

    initLoader() {
        const loader = new GLTFLoader();
        loader.register((parser) => new VRMLoaderPlugin(parser));

        loader.load(
            CONFIG.AVATAR.MODEL_PATH, 
            (gltf) => {
                const vrm = gltf.userData.vrm;
                VRMUtils.removeUnnecessaryVertices(gltf.scene);
                VRMUtils.removeUnnecessaryJoints(gltf.scene);
                
                this.currentVrm = vrm;
                this.scene.add(vrm.scene);
                
                // --- NEW: Assign the lookAt target to the VRM ---
                if (this.currentVrm.lookAt) {
                    this.currentVrm.lookAt.target = this.lookAtTarget;
                }
                
                // Initialize the Animation Mixer tied to the VRM scene
                this.mixer = new THREE.AnimationMixer(vrm.scene);

                // Listen for animation loops to trigger variety
                this.mixer.addEventListener('loop', (e) => {
                    if (this.currentState === 'dreaming' || this.isGesturePlaying) {
                        return;
                    }

                    // Default 30% chance to switch for idle, thinking, etc.
                    let switchChance = 0.3; 
                    
                    // If talking, increase the chance to 85% for dynamic hand gestures
                    if (this.currentState === 'responding') {
                        switchChance = 0.85; 
                    }

                    if (Math.random() < switchChance) {
                        this.playRandomVariant(this.currentState);
                    }
                });

                this.mixer.addEventListener('finished', (event) => {
                    if (this.isGesturePlaying && this.activeGestureName) {
                        const gestureKey = this.gestureAnimations[this.activeGestureName];
                        const gestureAction = gestureKey ? this.animations[gestureKey] : null;
                        if (gestureAction && event.action === gestureAction) {
                            this.isGesturePlaying = false;
                            this.activeGestureName = null;

                            this.processGestureQueue();
                            if (!this.isGesturePlaying) {
                                this.playRandomVariant(this.currentState);
                            }
                            return;
                        }
                    }

                    if (this.currentState !== 'dreaming') {
                        return;
                    }

                    const introKey = this.stateAnimations.dreaming?.[0];
                    const outroKey = this.stateAnimations.dreaming?.[1];
                    const introAction = introKey ? this.animations[introKey] : null;
                    const outroAction = outroKey ? this.animations[outroKey] : null;

                    if (this.dreamingPhase === 'intro' && introAction && event.action === introAction) {
                        this.dreamingPhase = 'holding';
                        this.forceEyesClosed = true;
                        if (this.pendingDreamingOutroStart) {
                            this.pendingDreamingOutroStart = false;
                            this.startDreamingOutroPlayback();
                        }
                        return;
                    }

                    if (this.dreamingPhase === 'outro' && outroAction && event.action === outroAction) {
                        this.dreamingPhase = null;
                        this.forceEyesClosed = false;
                        if (this.dreamingOutroTimeoutId) {
                            clearTimeout(this.dreamingOutroTimeoutId);
                            this.dreamingOutroTimeoutId = null;
                        }
                        const resolve = this.dreamingOutroResolver;
                        this.dreamingOutroResolver = null;
                        this.dreamingOutroPromise = null;
                        if (resolve) {
                            resolve(true);
                        }
                    }
                });

                // Load all animations dynamically
                Object.keys(CONFIG.AVATAR.ANIMATIONS).forEach(state => {
                    const paths = CONFIG.AVATAR.ANIMATIONS[state];
                    this.stateAnimations[state] = []; 
                    
                    paths.forEach((path, index) => {
                        const animKey = `${state}_${index}`; 
                        this.stateAnimations[state].push(animKey);
                        
                        const playImmediately = (state === 'idle' && index === 0);
                        this.loadAnimation(path, animKey, playImmediately);
                    });
                });
                this.loadGestureAnimations();
            },
            (progress) => {},
            (error) => console.error("Error loading VRM:", error)
        );
    }

    // --- Mixamo FBX Loader Method ---
    loadAnimation(url, name, playImmediately = false, onLoaded = null) {
        loadMixamoAnimation(url, this.currentVrm)
            .then((clip) => {
                if (clip) {
                    const action = this.mixer.clipAction(clip);
                    this.animations[name] = action;
                    if (onLoaded) {
                        onLoaded(action);
                    }

                    if (playImmediately) {
                        this.fadeToAction(name, 0.0); 
                    }
                }
            })
            .catch((error) => console.error(`Failed to load Mixamo animation "${name}":`, error));
    }

    setGestureCatalog(gestureCatalog = {}) {
        this.gestureCatalog = {};
        this.gestureAnimations = {};
        this.gestureQueue = [];
        this.isGesturePlaying = false;
        this.activeGestureName = null;

        for (const [name, url] of Object.entries(gestureCatalog)) {
            if (!url) continue;
            const normalizedName = String(name).trim().toLowerCase();
            if (!normalizedName) continue;
            this.gestureCatalog[normalizedName] = url;
        }

        this.loadGestureAnimations();
    }

    loadGestureAnimations() {
        if (!this.currentVrm || !this.mixer) {
            return;
        }

        for (const [gestureName, url] of Object.entries(this.gestureCatalog)) {
            if (this.gestureAnimations[gestureName]) {
                continue;
            }

            const animationKey = `gesture_${gestureName}`;
            this.gestureAnimations[gestureName] = animationKey;
            this.loadAnimation(url, animationKey, false, () => {
                this.processGestureQueue();
            });
        }
    }

    queueGesture(animationName) {
        const normalized = String(animationName || '').trim().toLowerCase();
        if (!normalized) {
            return;
        }

        if (!this.gestureCatalog[normalized]) {
            console.debug(`Ignoring unknown gesture animation "${normalized}"`);
            return;
        }

        this.gestureQueue.push(normalized);
        this.processGestureQueue();
    }

    processGestureQueue() {
        if (this.isGesturePlaying) {
            return;
        }

        const nextGestureName = this.gestureQueue[0];
        if (!nextGestureName) {
            return;
        }

        const animationKey = this.gestureAnimations[nextGestureName];
        const action = animationKey ? this.animations[animationKey] : null;
        if (!action) {
            return;
        }

        this.gestureQueue.shift();
        this.isGesturePlaying = true;
        this.activeGestureName = nextGestureName;
        action.setLoop(THREE.LoopOnce, 1);
        action.clampWhenFinished = true;
        this.fadeToAction(animationKey, 0.35);
    }

    fadeToAction(name, duration = 0.5) {
        const nextAction = this.animations[name];
        if (!nextAction || this.currentAction === nextAction) return;

        nextAction.reset();
        // --- NEW: Ensure the incoming animation isn't paused ---
        nextAction.paused = false; 
        nextAction.play();

        if (this.currentAction) {
            // --- NEW: Unpause the outgoing animation ---
            // This is the secret sauce. By unpausing the thinking animation right as 
            // the crossfade begins, her arm naturally starts moving down while simultaneously 
            // blending into her talking gesture. It looks incredibly fluid!
            this.currentAction.paused = false; 
            
            nextAction.crossFadeFrom(this.currentAction, duration, true);
        }

        this.currentAction = nextAction;
    }

    // --- Random Variant Selector ---
    getPlayableAnimationKeys(state) {
        const keys = this.stateAnimations[state] || [];
        return keys.filter((key) => Boolean(this.animations[key]));
    }

    resolveAnimationState(state) {
        const hasStateAnimations = this.getPlayableAnimationKeys(state).length > 0;
        if (hasStateAnimations) {
            return state;
        }

        if (state === 'dreaming') {
            const hasThinkingAnimations = this.getPlayableAnimationKeys('thinking').length > 0;
            if (hasThinkingAnimations) {
                return 'thinking';
            }
        }

        const hasIdleAnimations = this.getPlayableAnimationKeys('idle').length > 0;
        if (hasIdleAnimations) {
            return 'idle';
        }

        return state;
    }

    playRandomVariant(state) {
        if (this.isGesturePlaying) {
            return;
        }

        const resolvedState = this.resolveAnimationState(state);
        const availableAnimations = this.getPlayableAnimationKeys(resolvedState);
        
        if (availableAnimations && availableAnimations.length > 0) {
            let selectedAnimKey;
            if (availableAnimations.length > 1 && this.currentAction) {
                do {
                    const randomIndex = Math.floor(Math.random() * availableAnimations.length);
                    selectedAnimKey = availableAnimations[randomIndex];
                } while (this.animations[selectedAnimKey] === this.currentAction);
            } else {
                selectedAnimKey = availableAnimations[0];
            }
            const action = this.animations[selectedAnimKey];
            if (action) {
                action.setLoop(THREE.LoopRepeat, Infinity);
                action.clampWhenFinished = false;
            }
            this.fadeToAction(selectedAnimKey, 0.8);
        }
    }

    setState(state) {
        if (this.currentState === state) return;
        const previousState = this.currentState;
        this.currentState = state;

        if (state === 'dreaming') {
            this.startDreamingIntro();
            return;
        }

        if (previousState === 'dreaming') {
            this.dreamingPhase = null;
            this.pendingDreamingOutroStart = false;
            this.forceEyesClosed = false;
            if (this.dreamingOutroTimeoutId) {
                clearTimeout(this.dreamingOutroTimeoutId);
                this.dreamingOutroTimeoutId = null;
            }
            if (this.dreamingOutroResolver) {
                this.dreamingOutroResolver(false);
                this.dreamingOutroResolver = null;
            }
            this.dreamingOutroPromise = null;
        }

        if (this.isGesturePlaying) {
            return;
        }

        this.playRandomVariant(state);
    }

    startDreamingIntro() {
        this.dreamingPhase = 'intro';
        this.pendingDreamingOutroStart = false;
        this.forceEyesClosed = false;
        if (this.dreamingOutroTimeoutId) {
            clearTimeout(this.dreamingOutroTimeoutId);
            this.dreamingOutroTimeoutId = null;
        }
        this.dreamingOutroPromise = null;
        this.dreamingOutroResolver = null;

        const introKey = this.stateAnimations.dreaming?.[0];
        if (!introKey || !this.playAnimationOnce(introKey, 0.5)) {
            this.dreamingPhase = 'holding';
            return;
        }
    }

    playDreamingOutro() {
        if (this.currentState !== 'dreaming') {
            return Promise.resolve(false);
        }

        if (this.dreamingOutroPromise) {
            return this.dreamingOutroPromise;
        }

        this.dreamingOutroPromise = new Promise((resolve) => {
            this.dreamingOutroResolver = resolve;
        });

        if (this.dreamingPhase === 'intro') {
            this.pendingDreamingOutroStart = true;
            return this.dreamingOutroPromise;
        }

        this.startDreamingOutroPlayback();
        return this.dreamingOutroPromise;
    }

    startDreamingOutroPlayback() {
        if (this.currentState !== 'dreaming') {
            return false;
        }

        this.pendingDreamingOutroStart = false;
        this.forceEyesClosed = false;
        this.dreamingPhase = 'outro';

        const outroKey = this.stateAnimations.dreaming?.[1];
        if (!outroKey || !this.playAnimationOnce(outroKey, 0.35)) {
            this.dreamingPhase = null;
            const resolve = this.dreamingOutroResolver;
            this.dreamingOutroResolver = null;
            this.dreamingOutroPromise = null;
            if (resolve) {
                resolve(false);
            }
            return false;
        }

        const outroAction = this.animations[outroKey];
        const outroDurationSeconds = Number(outroAction?.getClip?.().duration) || 0;
        const timeoutMs = Math.max(3000, Math.ceil((outroDurationSeconds * 1000) + 1200));

        // Safety net: never let UI remain stuck waiting for a missing finished event.
        this.dreamingOutroTimeoutId = setTimeout(() => {
            this.dreamingOutroTimeoutId = null;
            this.dreamingPhase = null;
            this.forceEyesClosed = false;
            const pendingResolve = this.dreamingOutroResolver;
            this.dreamingOutroResolver = null;
            this.dreamingOutroPromise = null;
            if (pendingResolve) {
                pendingResolve(false);
            }
        }, timeoutMs);
        return true;
    }

    playAnimationOnce(animationKey, duration = 0.5) {
        const action = this.animations[animationKey];
        if (!action) {
            return false;
        }

        action.setLoop(THREE.LoopOnce, 1);
        action.clampWhenFinished = true;
        this.fadeToAction(animationKey, duration);
        return true;
    }

    setExpression(expression) {
        if (!CONFIG.AVATAR.EXPRESSIONS.includes(expression)) {
            expression = 'neutral';
        }

        this.currentExpression = expression;
    }

    animate() {
        requestAnimationFrame(this.animate);
        
        const deltaTime = this.clock.getDelta();
        this.controls.update(); 

        if (this.currentVrm) {
            // --- Update procedural features BEFORE updating the VRM ---
            this.updateEyes(deltaTime);
            this.updateBlinking(deltaTime);
            this.updateExpression(deltaTime);
            
            const visemes = this.getAudioLevel(); // rename callback to getVisemeData

            const VISEME_KEYS = ['aa', 'ih', 'ou', 'ee', 'oh'];
            const SMOOTHING = CONFIG.AUDIO.LIP_SYNC_SMOOTHING; // reuse existing value

            // Expression dampening factor (your existing logic)
            let dampen = 1.0;
            if (this.currentExpression !== 'neutral' && this.currentVrm.expressionManager) {
                const w = this.currentVrm.expressionManager.getValue(this.currentExpression) || 0;
                dampen = 1.0 - (w * 0.6);
            }

            for (const key of VISEME_KEYS) {
                const target = (visemes[key] ?? 0) * dampen;
                const current = this.currentVrm.expressionManager.getValue(key) || 0;
                const next = THREE.MathUtils.lerp(current, target, SMOOTHING);
                this.currentVrm.expressionManager.setValue(key, next);
            }

            // --- NEW: Dynamic Animation Pausing (Hold the pose) ---
            // Never pause one-shot gesture clips; they must reach `finished`
            // so we can resume queued/default state animations.
            if (this.currentAction && this.currentState === 'thinking' && !this.isGesturePlaying) {
                const duration = this.currentAction.getClip().duration;
                
                // Freeze at the 50% mark. Tweak this 0.5 value if her hand hasn't 
                // fully reached her chin yet (e.g., try 0.6 or 0.7).
                if (this.currentAction.time >= duration * 0.5) {
                    this.currentAction.paused = true;
                }
            }

            // Update the Animation Mixer (bones)
            if (this.mixer) this.mixer.update(deltaTime);

            // Finally, update the VRM to apply everything to the model
            this.currentVrm.update(deltaTime);
        }
        
        this.renderer.render(this.scene, this.camera);
    }
    // --- NEW: Eye Tracking Logic ---
    updateEyes(deltaTime) {
        this.eyeTimer += deltaTime;

        // Time to change where we are looking?
        if (this.eyeTimer >= this.nextEyeMoveTime) {
            this.eyeTimer = 0;
            this.isLookingAtCamera = !this.isLookingAtCamera; // Toggle state
            
            if (this.isLookingAtCamera) {
                // Stare at the camera for a longer period (3 to 8 seconds)
                this.nextEyeMoveTime = Math.random() * 5.0 + 3.0;
                this.lookAtOffset.set(0, 0, 0);
            } else {
                // Dart eyes away for a shorter period (0.5 to 2 seconds)
                this.nextEyeMoveTime = Math.random() * 1.5 + 0.5;
                
                // Pick a random spot near the camera to look at
                const xOffset = (Math.random() - 0.5) * 5.0; // Left or right
                const yOffset = (Math.random() - 0.5) * 2.0; // Up or down
                this.lookAtOffset.set(xOffset, yOffset, 0);
            }
        }

        // Calculate the exact 3D position we want the eyes to aim at
        // (Camera position + our random offset)
        const targetPos = this.camera.position.clone().add(this.lookAtOffset);
        
        // Smoothly move our invisible target object to that position
        // A lerp factor of 10.0 gives a nice, quick "eye dart" feel
        this.lookAtTarget.position.lerp(targetPos, 10.0 * deltaTime);
    }

    updateExpression(deltaTime) {
        if (!this.currentVrm?.expressionManager) return;

        const blendSpeed = Math.min(1, deltaTime * 6);

        for (const expression of CONFIG.AVATAR.EXPRESSIONS) {
            if (expression === 'neutral') continue;

            const targetValue = this.currentExpression === expression ? 1 : 0;
            const currentValue = this.currentVrm.expressionManager.getValue(expression) || 0;
            const nextValue = THREE.MathUtils.lerp(currentValue, targetValue, blendSpeed);
            this.currentVrm.expressionManager.setValue(expression, nextValue);
        }
    }

    updateBlinking(deltaTime) {
        if (!this.currentVrm?.expressionManager) {
            return;
        }

        if (this.forceEyesClosed) {
            this.currentVrm.expressionManager.setValue('blink', 1.0);
            return;
        }

        const blinkSpeed = 15.0; 
        let currentBlink = this.currentVrm.expressionManager.getValue('blink');

        switch (this.blinkState) {
            case 'open':
                this.blinkTimer += deltaTime;
                if (this.blinkTimer >= this.nextBlinkTime) {
                    this.blinkState = 'closing';
                    this.blinkTimer = 0;
                }
                break;
            case 'closing':
                currentBlink += blinkSpeed * deltaTime;
                if (currentBlink >= 1.0) {
                    currentBlink = 1.0;
                    this.blinkState = 'closed';
                }
                break;
            case 'closed':
                this.blinkTimer += deltaTime;
                if (this.blinkTimer >= 0.1) {
                    this.blinkState = 'opening';
                    this.blinkTimer = 0;
                }
                break;
            case 'opening':
                currentBlink -= blinkSpeed * deltaTime;
                if (currentBlink <= 0.0) {
                    currentBlink = 0.0;
                    this.blinkState = 'open';
                    
                    const isIdle = (this.currentState === 'idle');
                    const baseTime = isIdle ? 3.0 : 1.5;
                    this.nextBlinkTime = Math.random() * 2.0 + baseTime; 
                }
                break;
        }

        this.currentVrm.expressionManager.setValue('blink', currentBlink);
    }
}
