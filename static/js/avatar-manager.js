import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { CONFIG } from './config.js';

export class AvatarManager {
    constructor(containerId, getAudioLevelCallback) {
        this.container = document.getElementById(containerId);
        this.getAudioLevel = getAudioLevelCallback; 
        
        this.currentVrm = null;
        this.currentState = "idle";
        this.poseTarget = CONFIG.AVATAR.POSES.idle;
        
        this.initScene();
        this.initLoader();
        this.clock = new THREE.Clock();
        
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
        this.camera.position.set(1.0, 1.0, 4.0); 

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(0.0, 0.9, 0.0);
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
                
                // Initial Arm Pose (Tuck arms down)
                vrm.humanoid.getNormalizedBoneNode('leftUpperArm').rotation.z = -1.2; 
                vrm.humanoid.getNormalizedBoneNode('rightUpperArm').rotation.z = 1.2;
            },
            (progress) => {},
            (error) => console.error("Error loading VRM:", error)
        );
    }

    setState(state) {
        this.currentState = state;
        // Look up pose in config, default to idle if not found
        this.poseTarget = CONFIG.AVATAR.POSES[state] || CONFIG.AVATAR.POSES.idle;
    }

    animate() {
        requestAnimationFrame(this.animate);
        const deltaTime = this.clock.getDelta();
        const time = this.clock.elapsedTime;
        this.controls.update(); 

        if (this.currentVrm) {
            this.currentVrm.update(deltaTime);
            
            // --- LIP SYNC ---
            const targetMouthOpen = this.getAudioLevel();
            const currentMouth = this.currentVrm.expressionManager.getValue('aa');
            const smoothedMouth = THREE.MathUtils.lerp(
                currentMouth, 
                targetMouthOpen, 
                CONFIG.AUDIO.LIP_SYNC_SMOOTHING
            );
            this.currentVrm.expressionManager.setValue('aa', smoothedMouth);

            // --- ANIMATIONS ---
            this.processIdleAnimations(time);
        }
        this.renderer.render(this.scene, this.camera);
    }

    processIdleAnimations(time) {
        const neck = this.currentVrm.humanoid.getNormalizedBoneNode('neck');
        const chest = this.currentVrm.humanoid.getNormalizedBoneNode('chest');
        const spine = this.currentVrm.humanoid.getNormalizedBoneNode('spine');

        // Breathing
        chest.rotation.x = Math.sin(time * 1.0) * 0.03;

        // Pose Smoothing
        const lerpFactor = CONFIG.AVATAR.POSE_LERP_FACTOR;
        neck.rotation.x = THREE.MathUtils.lerp(neck.rotation.x, this.poseTarget.neckX, lerpFactor);
        neck.rotation.y = THREE.MathUtils.lerp(neck.rotation.y, this.poseTarget.neckY, lerpFactor);
        spine.rotation.y = THREE.MathUtils.lerp(spine.rotation.y, this.poseTarget.spineY, lerpFactor);

        // Sway
        neck.rotation.y += Math.sin(time * 0.5) * 0.02;
        
        // Blink logic based on state
        const isIdle = (this.currentState === 'idle');
        const blinkChance = isIdle ? CONFIG.AVATAR.BLINK_CHANCE_IDLE : CONFIG.AVATAR.BLINK_CHANCE_ACTIVE;
        
        if (Math.random() < blinkChance) {
            this.currentVrm.expressionManager.setValue('blink', 1.0);
            setTimeout(() => this.currentVrm?.expressionManager.setValue('blink', 0.0), 150);
        }
    }
}