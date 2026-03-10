import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { CONFIG } from './config.js';

// Import the Mixamo loader helper (ensure this file and mixamoVRMRigMap.js are in your directory)
import { loadMixamoAnimation } from './loadMixamoAnimation.js'; 

export class AvatarManager {
    constructor(containerId, getAudioLevelCallback) {
        this.container = document.getElementById(containerId);
        this.getAudioLevel = getAudioLevelCallback; 
        
        this.currentVrm = null;
        this.currentState = "idle";
        
        // Animation Mixer & Storage
        this.mixer = null;
        this.animations = {}; 
        this.currentAction = null;
        this.stateAnimations = {}; // NEW: Keeps track of which animation keys belong to which state
        
        // Blink State Management
        this.blinkState = 'open'; 
        this.blinkTimer = 0;
        this.nextBlinkTime = Math.random() * 3 + 2; 

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
        this.camera.position.set(0.0, 1.4, 3.0); 

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(0.0, 1.4, 0.0);
        this.controls.update();

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
                
                // Initialize the Animation Mixer tied to the VRM scene
                this.mixer = new THREE.AnimationMixer(vrm.scene);

                // --- NEW: Listen for animation loops to trigger variety ---
                this.mixer.addEventListener('loop', (e) => {
                    // 30% chance to switch to a different animation in the same state when one finishes
                    if (Math.random() < 0.3) {
                        this.playRandomVariant(this.currentState);
                    }
                });

                // --- NEW: Load all animations from the arrays dynamically ---
                Object.keys(CONFIG.AVATAR.ANIMATIONS).forEach(state => {
                    const paths = CONFIG.AVATAR.ANIMATIONS[state];
                    this.stateAnimations[state] = []; // Initialize empty array for this state
                    
                    paths.forEach((path, index) => {
                        const animKey = `${state}_${index}`; // e.g., "idle_0", "idle_1"
                        this.stateAnimations[state].push(animKey);
                        
                        // Play the very first idle animation immediately when loaded
                        const playImmediately = (state === 'idle' && index === 0);
                        this.loadAnimation(path, animKey, playImmediately);
                    });
                });
            },
            (progress) => {},
            (error) => console.error("Error loading VRM:", error)
        );
    }

    // --- Mixamo FBX Loader Method ---
    loadAnimation(url, name, playImmediately = false) {
        loadMixamoAnimation(url, this.currentVrm)
            .then((clip) => {
                if (clip) {
                    const action = this.mixer.clipAction(clip);
                    this.animations[name] = action;

                    if (playImmediately) {
                        this.fadeToAction(name, 0.0); 
                    }
                }
            })
            .catch((error) => {
                console.error(`Failed to load Mixamo animation "${name}":`, error);
            });
    }

    // --- Crossfade Logic ---
    fadeToAction(name, duration = 0.5) {
        const nextAction = this.animations[name];
        if (!nextAction) {
            console.warn(`Animation "${name}" not found or not loaded yet.`);
            return;
        }

        if (this.currentAction === nextAction) return;

        nextAction.reset();
        nextAction.play();

        if (this.currentAction) {
            nextAction.crossFadeFrom(this.currentAction, duration, true);
        }

        this.currentAction = nextAction;
        // Note: this.currentState is now updated in setState()
    }

    // --- NEW: Random Variant Selector ---
    playRandomVariant(state) {
        const availableAnimations = this.stateAnimations[state];
        
        if (availableAnimations && availableAnimations.length > 0) {
            let selectedAnimKey;
            
            // If we have multiple animations, ensure we don't pick the one currently playing
            if (availableAnimations.length > 1 && this.currentAction) {
                do {
                    const randomIndex = Math.floor(Math.random() * availableAnimations.length);
                    selectedAnimKey = availableAnimations[randomIndex];
                } while (this.animations[selectedAnimKey] === this.currentAction);
            } else {
                // If there's only one, just pick it
                selectedAnimKey = availableAnimations[0];
            }
            
            // Crossfade to the new variant
            this.fadeToAction(selectedAnimKey, 0.8);
        }
    }

    // --- UPDATED: Set State ---
    setState(state) {
        this.currentState = state;
        // Trigger a random animation from the requested state's pool
        this.playRandomVariant(state);
    }

    animate() {
        requestAnimationFrame(this.animate);
        
        const deltaTime = this.clock.getDelta();
        this.controls.update(); 

        if (this.currentVrm) {
            this.currentVrm.update(deltaTime);
            
            if (this.mixer) {
                this.mixer.update(deltaTime);
            }

            // --- LIP SYNC ---
            const targetMouthOpen = this.getAudioLevel();
            const currentMouth = this.currentVrm.expressionManager.getValue('aa');
            const smoothedMouth = THREE.MathUtils.lerp(
                currentMouth, 
                targetMouthOpen, 
                CONFIG.AUDIO.LIP_SYNC_SMOOTHING
            );
            this.currentVrm.expressionManager.setValue('aa', smoothedMouth);

            // --- EXPRESSIONAL ANIMATIONS ---
            this.updateBlinking(deltaTime);
        }
        
        this.renderer.render(this.scene, this.camera);
    }

    updateBlinking(deltaTime) {
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