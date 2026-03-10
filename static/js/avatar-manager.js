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
            .catch((error) => console.error(`Failed to load Mixamo animation "${name}":`, error));
    }

    // --- Crossfade Logic ---
    fadeToAction(name, duration = 0.5) {
        const nextAction = this.animations[name];
        if (!nextAction || this.currentAction === nextAction) return;

        nextAction.reset();
        nextAction.play();

        if (this.currentAction) {
            nextAction.crossFadeFrom(this.currentAction, duration, true);
        }

        this.currentAction = nextAction;
    }

    // --- Random Variant Selector ---
    playRandomVariant(state) {
        const availableAnimations = this.stateAnimations[state];
        
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
            this.fadeToAction(selectedAnimKey, 0.8);
        }
    }

    setState(state) {
        if (this.currentState === state) return;
        this.currentState = state;
        this.playRandomVariant(state);
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
            
            // --- UPDATED: Dynamic Audio Lip Sync ---
            let targetMouthOpen = this.getAudioLevel();

            // Prevent blendshape clashing by dampening lip-sync during strong expressions
            if (this.currentExpression !== 'neutral' && this.currentVrm.expressionManager) {
                const activeExpWeight = this.currentVrm.expressionManager.getValue(this.currentExpression) || 0;
                
                // If expression is at 1.0, reduce lip-sync by 60%. 
                // You can tweak this 0.6 value based on how extreme your specific VRM's expressions are.
                const dampeningFactor = 1.0 - (activeExpWeight * 0.6); 
                targetMouthOpen *= dampeningFactor;
            }

            const currentMouth = this.currentVrm.expressionManager.getValue('aa');
            const smoothedMouth = THREE.MathUtils.lerp(currentMouth, targetMouthOpen, CONFIG.AUDIO.LIP_SYNC_SMOOTHING);
            this.currentVrm.expressionManager.setValue('aa', smoothedMouth);

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
