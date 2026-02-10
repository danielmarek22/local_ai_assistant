import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

let avatarState = "idle";
const poseTarget = { neckX: 0, neckY: 0, spineY: 0 };

let currentVrm = null;
let analyser = null;
let dataArray = null;

export function setAvatarState(state) {
    avatarState = state;
}

export function attachAnalyser(a, d) {
    analyser = a;
    dataArray = d;
}

export function initAvatar() {
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(30, window.innerWidth / window.innerHeight, 0.1, 20);
    camera.position.set(1, 1, 4);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.9, 0);
    controls.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.4));
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(1, 1, 1);
    scene.add(light);

    const loader = new GLTFLoader();
    loader.register(p => new VRMLoaderPlugin(p));

    loader.load('/static/avatar.vrm', gltf => {
        currentVrm = gltf.userData.vrm;
        VRMUtils.removeUnnecessaryVertices(gltf.scene);
        VRMUtils.removeUnnecessaryJoints(gltf.scene);
        scene.add(currentVrm.scene);
    });

    const clock = new THREE.Clock();

    function animate() {
        requestAnimationFrame(animate);
        const dt = clock.getDelta();
        const t = clock.elapsedTime;

        if (currentVrm) {
            currentVrm.update(dt);

            if (analyser) {
                analyser.getByteFrequencyData(dataArray);
                const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
                currentVrm.expressionManager.setValue('aa', Math.min(1, avg / 60));
            }

            const neck = currentVrm.humanoid.getNormalizedBoneNode('neck');
            const chest = currentVrm.humanoid.getNormalizedBoneNode('chest');
            chest.rotation.x = Math.sin(t) * 0.03;

            neck.rotation.x = THREE.MathUtils.lerp(neck.rotation.x, poseTarget.neckX, 0.05);
            neck.rotation.y += Math.sin(t * 0.5) * 0.02;
        }

        renderer.render(scene, camera);
    }

    animate();
}
