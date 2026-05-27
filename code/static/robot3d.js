// robot3d.js
window.Robot3D = (function() {
    let scene, camera, renderer, container;
    let robotBase, shoulder, elbow, wristPitch, wristRoll, gripperBase, leftFinger, rightFinger;
    let objects = {}; // store beakers, flasks, centerStage by name
    let isInitialized = false;

    function init(containerEl, loaderEl) {
        if (isInitialized) return;
        container = containerEl;
        
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0f172a); // slate-900

        const aspect = container.clientWidth / container.clientHeight;
        camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 100);
        // Moved camera a bit closer, looking directly at the center stage
        camera.position.set(0, 8, 16); 
        camera.lookAt(0, 1, 0);

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        const existing = container.querySelector('canvas');
        if (existing) existing.remove();
        
        container.appendChild(renderer.domElement);
        renderer.domElement.className = "absolute inset-0 w-full h-full object-cover z-10 cursor-grab active:cursor-grabbing";

        if (typeof THREE.OrbitControls !== 'undefined') {
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.target.set(0, 1.5, 0);
            window.__orbitControls = controls;
        }

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(10, 15, 10);
        dirLight.castShadow = true;
        dirLight.shadow.mapSize.width = 1024;
        dirLight.shadow.mapSize.height = 1024;
        scene.add(dirLight);

        buildEnvironment();
        buildRobotArm();
        buildObjects();

        window.addEventListener('resize', onWindowResize);
        renderer.setAnimationLoop(animate);
        
        isInitialized = true;
        if (loaderEl) {
            loaderEl.classList.add('opacity-0');
            setTimeout(() => loaderEl.classList.add('hidden'), 300);
        }
    }

    function buildEnvironment() {
        // Glove box floor: 20x10
        const floorGeo = new THREE.BoxGeometry(20, 0.2, 14);
        const floorMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, roughness: 0.9 });
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.position.y = -0.1;
        floor.receiveShadow = true;
        scene.add(floor);

        // Glass walls (20x10 size)
        const wallMat = new THREE.MeshPhysicalMaterial({ 
            color: 0xffffff, transmission: 0.5, opacity: 1, transparent: true, roughness: 0.1, side: THREE.DoubleSide
        });
        
        const backWall = new THREE.Mesh(new THREE.BoxGeometry(20, 10, 0.1), wallMat);
        backWall.position.set(0, 5, -7);
        scene.add(backWall);

        const leftWall = new THREE.Mesh(new THREE.BoxGeometry(0.1, 10, 14), wallMat);
        leftWall.position.set(-10, 5, 0);
        scene.add(leftWall);

        const rightWall = new THREE.Mesh(new THREE.BoxGeometry(0.1, 10, 14), wallMat);
        rightWall.position.set(10, 5, 0);
        scene.add(rightWall);

        // Add back shelves
        const shelfMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.6 });
        const shelf1 = new THREE.Mesh(new THREE.BoxGeometry(18, 0.1, 1.5), shelfMat);
        shelf1.position.set(0, 2.5, -6.25);
        shelf1.receiveShadow = true;
        scene.add(shelf1);

        const shelf2 = new THREE.Mesh(new THREE.BoxGeometry(18, 0.1, 1.5), shelfMat);
        shelf2.position.set(0, 5.0, -6.25);
        shelf2.receiveShadow = true;
        scene.add(shelf2);

        // Add Center Stage (Moved forward)
        const stageMat = new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.7 });
        const stage = new THREE.Mesh(new THREE.BoxGeometry(4, 0.6, 3), stageMat);
        stage.position.set(0, 0.3, 3); // rests on floor, forward
        stage.castShadow = true;
        stage.receiveShadow = true;
        scene.add(stage);
        
        objects.centerStage = stage;
        
        const stageGraspTarget = new THREE.Object3D();
        stageGraspTarget.position.set(0, 0.6, 0); // On top of the stage
        stage.add(stageGraspTarget);
        stage.userData.graspTarget = stageGraspTarget;
        stage.userData.isStage = true;

        // Add Drawer Unit to the right under shelf
        const drawerUnitMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.8 });
        const drawerBox = new THREE.Mesh(new THREE.BoxGeometry(4, 2.5, 3), drawerUnitMat);
        drawerBox.position.set(4, 1.25, -4.5);
        drawerBox.castShadow = true;
        drawerBox.receiveShadow = true;
        scene.add(drawerBox);

        // Actual sliding drawer
        const drawerInnerMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8 });
        const drawerSlide = new THREE.Mesh(new THREE.BoxGeometry(3.6, 1.0, 2.9), drawerInnerMat);
        drawerSlide.position.set(4, 2.0, -4.5); // align with top of box initially
        drawerSlide.castShadow = true;
        scene.add(drawerSlide);

        // Handle
        const handleMat = new THREE.MeshStandardMaterial({ color: 0x475569 });
        const handle = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.2, 0.2), handleMat);
        handle.position.set(0, 0, 1.5);
        drawerSlide.add(handle);

        objects.drawer = drawerSlide;
        drawerSlide.userData = { isOpen: false, closedZ: -4.5, openZ: -2.5 };
    }

    function buildRobotArm() {
        const matBlack = new THREE.MeshStandardMaterial({ color: 0x1f1f1f, roughness: 0.8 }); // 3D printed PETG/PLA
        const matMotor = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.9 }); // Darker servos
        const matPCB = new THREE.MeshStandardMaterial({ color: 0x065f46, roughness: 0.5 }); // Dark green board
        const matSilver = new THREE.MeshStandardMaterial({ color: 0x888888, metalness: 0.8 }); // Component

        // 1. Base (Yaw - Y)
        robotBase = new THREE.Group();
        robotBase.position.set(0, 0, -1);
        
        // Scale down by 30%
        robotBase.scale.set(0.7, 0.7, 0.7);
        scene.add(robotBase);

        const baseBtm = new THREE.Mesh(new THREE.BoxGeometry(2.4, 0.4, 2.4), matBlack);
        baseBtm.position.y = 0.2;
        baseBtm.castShadow = true;
        robotBase.add(baseBtm);
        
        const baseMotor = new THREE.Mesh(new THREE.BoxGeometry(1.6, 1.2, 1.6), matMotor);
        baseMotor.position.y = 1.0;
        baseMotor.castShadow = true;
        robotBase.add(baseMotor);

        // 2. Shoulder (Pitch - X)
        shoulder = new THREE.Group();
        shoulder.position.y = 1.2; 
        robotBase.add(shoulder);

        const shoulderBracket = new THREE.Mesh(new THREE.BoxGeometry(1.8, 1.2, 1.5), matBlack);
        shoulderBracket.castShadow = true;
        shoulder.add(shoulderBracket);

        // Link 1 (Upper Arm, Length = 4.5, Truss structure)
        const link1L = new THREE.Mesh(new THREE.BoxGeometry(0.5, 4.5, 1.0), matBlack);
        link1L.position.set(-0.65, 2.25, 0);
        link1L.castShadow = true;
        shoulder.add(link1L);

        const link1R = new THREE.Mesh(new THREE.BoxGeometry(0.5, 4.5, 1.0), matBlack);
        link1R.position.set(0.65, 2.25, 0);
        link1R.castShadow = true;
        shoulder.add(link1R);

        const link1Top = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.6, 1.0), matBlack);
        link1Top.position.set(0, 4.2, 0);
        link1Top.castShadow = true;
        shoulder.add(link1Top);

        // 3. Elbow (Pitch - X)
        elbow = new THREE.Group();
        elbow.position.y = 4.5; 
        shoulder.add(elbow);

        const elbowMotor = new THREE.Mesh(new THREE.BoxGeometry(1.4, 1.4, 1.4), matMotor);
        elbowMotor.castShadow = true;
        elbow.add(elbowMotor);

        // Link 2 (Lower Arm, Length = 4.0)
        const link2Wrap = new THREE.Mesh(new THREE.BoxGeometry(1.2, 4.0, 1.0), matBlack);
        link2Wrap.position.set(0, 2.0, 0);
        link2Wrap.castShadow = true;
        elbow.add(link2Wrap);

        // 4. Wrist Pitch (Pitch - X)
        wristPitch = new THREE.Group();
        wristPitch.position.y = 4.0;
        elbow.add(wristPitch);

        const wpMotor = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.2, 1.2), matMotor);
        wpMotor.castShadow = true;
        wristPitch.add(wpMotor);

        const link3 = new THREE.Mesh(new THREE.BoxGeometry(0.8, 1.0, 0.8), matBlack);
        link3.position.y = 0.5;
        link3.castShadow = true;
        wristPitch.add(link3);

        // 5. Wrist Roll (Roll - Y locally)
        wristRoll = new THREE.Group();
        wristRoll.position.y = 1.0;
        wristPitch.add(wristRoll);

        const wrMotor = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.6, 0.9), matMotor);
        wrMotor.position.y = 0.3;
        wrMotor.castShadow = true;
        wristRoll.add(wrMotor);

        // 6. Gripper Base
        gripperBase = new THREE.Group();
        gripperBase.position.y = 0.6;
        wristRoll.add(gripperBase);

        const gripBaseMesh = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.4, 0.6), matBlack);
        gripBaseMesh.castShadow = true;
        gripperBase.add(gripBaseMesh);

        // PCB Decorative Board
        const pcb = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.8, 0.05), matPCB);
        pcb.position.set(0, -0.4, 0.35); 
        pcb.castShadow = true;
        gripperBase.add(pcb);
        
        const pcbChip = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.3, 0.06), matSilver);
        pcbChip.position.set(0, -0.4, 0.38);
        pcbChip.castShadow = true;
        gripperBase.add(pcbChip);
        
        const wire = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.6), new THREE.MeshStandardMaterial({color:0x222222}));
        wire.position.set(0, -0.1, 0.35);
        wire.castShadow = true;
        gripperBase.add(wire);

        // Gripper Fingers
        leftFinger = new THREE.Mesh(new THREE.BoxGeometry(0.2, 1.2, 0.4), matBlack);
        leftFinger.position.set(-0.6, 0.6, 0); 
        leftFinger.castShadow = true;
        gripperBase.add(leftFinger);

        rightFinger = new THREE.Mesh(new THREE.BoxGeometry(0.2, 1.2, 0.4), matBlack);
        rightFinger.position.set(0.6, 0.6, 0);
        rightFinger.castShadow = true;
        gripperBase.add(rightFinger);

        // Default pose pointing "forward" towards center stage
        robotBase.rotation.y = 0;
        shoulder.rotation.x = Math.PI / 6;     // 30 deg forward
        elbow.rotation.x = Math.PI / 4;        // 45 deg forward
        wristPitch.rotation.x = Math.PI / 8;   // slight bend
        wristRoll.rotation.y = 0;
    }

    function buildObjects() {
        // Move objects in slightly so the scaled-down 30% smaller robot can reach them
        const beakerA = createBeakerMesh(0x3b82f6, "A"); // blue
        beakerA.position.set(-3.5, 0, 0.5);
        scene.add(beakerA);
        objects.beaker_A_blue = beakerA;

        const beakerB = createBeakerMesh(0xef4444, "B"); // red
        beakerB.position.set(3.5, 0, 0.5);
        scene.add(beakerB);
        objects.beaker_B_red = beakerB;

        const flaskC = createFlaskMesh(0x22c55e, "C"); // green
        flaskC.position.set(-3.5, 2.55, -4.5);
        scene.add(flaskC);
        objects.flask_C_green = flaskC;

        const petriDish = createPetriDishMesh(0xec4899); // transparent
        petriDish.position.set(4, 2.0, -4.5); // inside the drawer, drawer needs to be moved closer too?
        petriDish.userData.liquid.scale.y = 0.001; 
        scene.add(petriDish);
        objects.petriDish = petriDish;
    }

    function createLabelSprite(text) {
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 128;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.beginPath();
        ctx.arc(64, 64, 50, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 4;
        ctx.stroke();
        
        ctx.font = 'bold 64px Inter, sans-serif';
        ctx.fillStyle = '#1e293b';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 64, 68);
        
        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ map: texture, depthTest: false });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.set(0.6, 0.6, 1);
        sprite.renderOrder = 999;
        return sprite;
    }

    function createBeakerMesh(colorCode, labelText) {
        const group = new THREE.Group();
        const glass = new THREE.Mesh(
            new THREE.CylinderGeometry(0.4, 0.4, 1.2, 16),
            new THREE.MeshPhysicalMaterial({ color: 0xffffff, transmission: 0.9, opacity: 1, transparent: true, roughness: 0.1, depthWrite: false })
        );
        glass.position.y = 0.6;
        glass.castShadow = true;
        group.add(glass);

        // Use dense geometry for sloshing (heightSegments=4, radialSegments=16)
        const liquidGeo = new THREE.CylinderGeometry(0.35, 0.35, 0.8, 16, 4);
        const liquidMat = new THREE.MeshStandardMaterial({ color: colorCode, transparent: true, opacity: 0.9 });
        const liquid = new THREE.Mesh(liquidGeo, liquidMat);
        liquid.position.y = 0.4;
        group.add(liquid);

        if (labelText) {
            const label = createLabelSprite(labelText);
            label.position.set(0, 0.6, 0.5);
            group.add(label);
        }

        const grasp = new THREE.Object3D();
        grasp.position.y = 0.8; // Beakers are grabbed slightly above middle
        group.add(grasp);
        group.userData.graspTarget = grasp;
        group.userData.liquid = liquid;
        group.userData.liquidBaseY = 0.4;
        group.userData.liquidHeight = 0.8;
        
        return group;
    }

    function createFlaskMesh(colorCode, labelText) {
        const group = new THREE.Group();
        const glass = new THREE.Mesh(
            new THREE.CylinderGeometry(0.2, 0.6, 1.2, 16),
            new THREE.MeshPhysicalMaterial({ color: 0xffffff, transmission: 0.9, opacity: 1, transparent: true, roughness: 0.1, depthWrite: false })
        );
        glass.position.y = 0.6;
        glass.castShadow = true;
        group.add(glass);

        const liquidGeo = new THREE.CylinderGeometry(0.18, 0.55, 0.8, 16, 4);
        const liquidMat = new THREE.MeshStandardMaterial({ color: colorCode, transparent: true, opacity: 0.9 });
        const liquid = new THREE.Mesh(liquidGeo, liquidMat);
        liquid.position.y = 0.4;
        group.add(liquid);

        if (labelText) {
            const label = createLabelSprite(labelText);
            label.position.set(0, 0.4, 0.7);
            group.add(label);
        }

        const grasp = new THREE.Object3D();
        // Flask neck is at the very top, grab it by the neck perfectly
        grasp.position.y = 1.0; 
        group.add(grasp);
        group.userData.graspTarget = grasp;
        group.userData.liquid = liquid;
        group.userData.liquidBaseY = 0.4;
        group.userData.liquidHeight = 0.8;
        
        return group;
    }

    function createPetriDishMesh(colorCode) {
        const group = new THREE.Group();
        const glass = new THREE.Mesh(
            new THREE.CylinderGeometry(0.6, 0.6, 0.15, 24),
            new THREE.MeshPhysicalMaterial({ color: 0xffffff, transmission: 0.9, opacity: 1, transparent: true, roughness: 0.1, depthWrite: false })
        );
        glass.position.y = 0.075;
        glass.castShadow = true;
        group.add(glass);

        const liquidGeo = new THREE.CylinderGeometry(0.55, 0.55, 0.1, 24, 2);
        const liquidMat = new THREE.MeshStandardMaterial({ color: colorCode, transparent: true, opacity: 0.9 });
        const liquid = new THREE.Mesh(liquidGeo, liquidMat);
        liquid.position.y = 0.06;
        group.add(liquid);

        const grasp = new THREE.Object3D();
        grasp.position.y = 0.1; // Grab from the edge
        group.add(grasp);
        group.userData.graspTarget = grasp;
        group.userData.liquid = liquid;
        group.userData.liquidBaseY = 0.06;
        group.userData.liquidHeight = 0.1;
        
        return group;
    }

    // --- AGENT ACTIONS EXECUTOR INTERFACE ---
    
    function tweenJoints(targets, duration = 1000) {
        return new Promise(resolve => {
            const currentObj = {
                baseY: robotBase.rotation.y,
                shoulderX: shoulder.rotation.x,
                elbowX: elbow.rotation.x,
                wristPitchX: wristPitch.rotation.x,
                wristRollY: wristRoll.rotation.y,
                gripX: rightFinger.position.x
            };

            new TWEEN.Tween(currentObj)
                .to(targets, duration)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .onUpdate(() => {
                    if (targets.baseY !== undefined) robotBase.rotation.y = currentObj.baseY;
                    if (targets.shoulderX !== undefined) shoulder.rotation.x = currentObj.shoulderX;
                    if (targets.elbowX !== undefined) elbow.rotation.x = currentObj.elbowX;
                    if (targets.wristPitchX !== undefined) wristPitch.rotation.x = currentObj.wristPitchX;
                    if (targets.wristRollY !== undefined) wristRoll.rotation.y = currentObj.wristRollY;
                    if (targets.gripX !== undefined) {
                        rightFinger.position.x = currentObj.gripX;
                        leftFinger.position.x = -currentObj.gripX;
                    }
                })
                .onComplete(resolve)
                .start();
        });
    }

    function alignObjToGripper(obj, duration = 300) {
        return new Promise(resolve => {
            // align graspTarget exactly to the gripper's grab point (y=0.5 in gripper space)
            const targetLocalY = -obj.userData.graspTarget.position.y + 0.5; 
            
            const start = {
                x: obj.position.x, y: obj.position.y, z: obj.position.z,
                rx: obj.rotation.x, ry: obj.rotation.y, rz: obj.rotation.z
            };
            const to = { x: 0, y: targetLocalY, z: 0, rx: 0, ry: 0, rz: 0 };
            
            new TWEEN.Tween(start).to(to, duration).easing(TWEEN.Easing.Quadratic.Out)
                .onUpdate(() => {
                    obj.position.set(start.x, start.y, start.z);
                    obj.rotation.set(start.rx, start.ry, start.rz);
                })
                .onComplete(resolve).start();
        });
    }

    const poses = {
        default: { baseY: 0, shoulderX: Math.PI/6, elbowX: Math.PI/4, wristPitchX: Math.PI/6, wristRollY: 0 }
    };

    function solveIK(world_tx, world_ty, world_tz) {
        // Convert world target to robot Base local space scaling
        let tx = world_tx / 0.7;
        let ty = world_ty / 0.7;
        let tz = (world_tz - (-1)) / 0.7; // robotBase is at z=-1
        
        let baseY = Math.atan2(tx, tz); 
        let currentBaseY = robotBase.rotation.y;
        while(baseY - currentBaseY > Math.PI) baseY -= Math.PI * 2;
        while(baseY - currentBaseY < -Math.PI) baseY += Math.PI * 2;
        
        let r = Math.sqrt(tx*tx + tz*tz);
        
        // Target Y is `ty` in local space. Wrist Pitch Y must be `ty + wrist_len`.
        const wrist_len = 2.2; // 1.0(wristRoll) + 0.6(gripperBase) + 0.6(finger grasp point)
        const Sy = 1.2;        // Shoulder pivot height
        let dWy = (ty + wrist_len) - Sy;
        
        let D = Math.sqrt(r*r + dWy*dWy);
        
        const L1 = 4.5;
        const L2 = 4.0;
        
        if (D > L1 + L2 - 0.01) D = L1 + L2 - 0.01; // Clamp stretch
        
        // Triangle math
        // gamma is in-between L1 and L2
        // t2 is the outer angle (elbow bend)
        let t2 = Math.acos((D*D - L1*L1 - L2*L2) / (2 * L1 * L2));
        
        // t1 is shoulder bend from +Y vertical axis
        let t1 = Math.atan2(r, dWy) - Math.acos((L1*L1 + D*D - L2*L2) / (2 * L1 * D));
        
        // t3 makes wrist point straight down to grasp object vertically
        let t3 = Math.PI - t1 - t2;
        
        return { baseY, shoulderX: t1, elbowX: t2, wristPitchX: t3, wristRollY: 0 };
    }

    let heldObject = null;

    async function actionPick(targetName) {
        const obj = objects[targetName];
        if (!obj || heldObject) return;
        
        const targetPos = new THREE.Vector3();
        obj.userData.graspTarget.getWorldPosition(targetPos);
        
        const prePose = solveIK(targetPos.x, targetPos.y + 2.5, targetPos.z);
        await tweenJoints({ ...prePose, gripX: 0.6 }, 1000);
        
        const grabPose = solveIK(targetPos.x, targetPos.y, targetPos.z);
        await tweenJoints(grabPose, 800);
        
        await tweenJoints({ gripX: 0.25 }, 300);
        gripperBase.attach(obj);
        heldObject = obj;
        
        await tweenJoints(prePose, 800);
    }

    async function actionMoveTo(targetName) {
        if (!heldObject) return;
        const targetObj = objects[targetName];
        if (!targetObj) return;
        
        const targetPos = new THREE.Vector3();
        targetObj.userData.graspTarget.getWorldPosition(targetPos);
        
        let pourMode = !targetObj.userData.isStage;
        if (pourMode) targetPos.y += 1.3;
        
        const safeY = Math.max(heldObject.position.y, targetPos.y) + 3.0;
        
        const transitPose = solveIK(targetPos.x, safeY, targetPos.z);
        await tweenJoints(transitPose, 1200);
        
        if (pourMode) {
            const pourPose = solveIK(targetPos.x, targetPos.y, targetPos.z);
            await tweenJoints(pourPose, 600);
        }
    }

    async function actionPour(targetName) {
        if (!heldObject) return;
        const currentPitch = wristPitch.rotation.x;
        
        const targetObj = objects[targetName];
        if (!targetObj) return;

        // Position above target already handled by moveTo. Now Tilt.
        await tweenJoints({ wristPitchX: currentPitch - Math.PI/2, wristRollY: Math.PI/2 }, 800);
        
        // --- Visual Fluid Transfer ---
        // Create a falling stream
        const streamGeo = new THREE.CylinderGeometry(0.05, 0.05, 1.5, 8);
        const streamMat = heldObject.userData.liquid.material.clone();
        const stream = new THREE.Mesh(streamGeo, streamMat);
        
        const targetPos = new THREE.Vector3();
        targetObj.userData.graspTarget.getWorldPosition(targetPos);
        stream.position.set(0, -0.6, 0); // Positioned relative to held object's lip
        heldObject.add(stream);

        // Animate volumes
        const pourDuration = 1000;
        const sourceLiquid = heldObject.userData.liquid;
        const targetLiquid = targetObj.userData.liquid;
        
        // Start pouring
        new TWEEN.Tween({ t: 0 })
            .to({ t: 1 }, pourDuration)
            .onUpdate((obj) => {
                sourceLiquid.scale.y = Math.max(0.001, 1 - obj.t * 0.8); // decrease source
                sourceLiquid.position.y = heldObject.userData.liquidBaseY - (heldObject.userData.liquidHeight * (1 - sourceLiquid.scale.y)) / 2;
                
                targetLiquid.scale.y = Math.min(1, targetLiquid.scale.y + obj.t * 0.05); // increase target
                targetLiquid.position.y = targetObj.userData.liquidBaseY - (targetObj.userData.liquidHeight * (1 - targetLiquid.scale.y)) / 2;
            })
            .start();

        await new Promise(r => setTimeout(r, pourDuration));
        
        heldObject.remove(stream);
        streamGeo.dispose();
        
        // Return wrist
        await tweenJoints({ wristPitchX: currentPitch, wristRollY: 0 }, 800);
    }
    
    async function actionDrawer(targetName, open) {
        const drawer = objects[targetName];
        if (!drawer || !drawer.userData) return;
        
        const targetZ = open ? drawer.userData.openZ : drawer.userData.closedZ;
        if (open) {
            // move arm to handle
            const prePose = solveIK(drawer.position.x, drawer.position.y, drawer.position.z + 1.5);
            await tweenJoints(prePose, 800);
        }
        
        new TWEEN.Tween(drawer.position)
            .to({ z: targetZ }, 800)
            .easing(TWEEN.Easing.Cubic.Out)
            .start();
            
        await new Promise(r => setTimeout(r, 800));
        drawer.userData.isOpen = open;
    }
    
    async function actionMix() {
        if (!heldObject) return;
        const currentPitch = wristPitch.rotation.x;
        for(let i=0; i<3; i++) {
            await tweenJoints({ wristRollY: Math.PI/4, wristPitchX: currentPitch + 0.1 }, 200);
            await tweenJoints({ wristRollY: -Math.PI/4, wristPitchX: currentPitch - 0.1 }, 400);
            await tweenJoints({ wristRollY: 0, wristPitchX: currentPitch }, 200);
        }
    }

    async function actionPlace(targetName) {
        if (!heldObject) return;
        
        let tx = 0, ty = 0, tz = 0;
        if (targetName === 'centerStage') { tx = 0; ty = 0.6; tz = 3; }
        else if (targetName === 'beaker_A_blue') { tx = -3.5; ty = 0; tz = 0.5; }
        else if (targetName === 'beaker_B_red') { tx = 3.5; ty = 0; tz = 0.5; }
        else if (targetName === 'flask_C_green') { tx = -3.5; ty = 2.55; tz = -4.5; }
        else if (targetName === 'petriDish') { tx = 4; ty = 2.0; tz = -4.5; }
        
        const graspOffsetY = heldObject.userData.graspTarget.position.y;
        const targetY = ty + graspOffsetY;
        
        const prePose = solveIK(tx, targetY + 2.5, tz);
        await tweenJoints(prePose, 800);
        
        const placePose = solveIK(tx, targetY, tz);
        await tweenJoints(placePose, 800);
        
        scene.attach(heldObject);
        new TWEEN.Tween(heldObject.position).to({ x: tx, y: ty, z: tz }, 300).start();
        new TWEEN.Tween(heldObject.rotation).to({ x:0, y:0, z:0 }, 300).start();
        
        await tweenJoints({ gripX: 0.6 }, 300);
        heldObject = null;
        
        await tweenJoints(prePose, 800);
        await tweenJoints(poses.default, 800);
    }

    function getEnvironmentState() {
        return Object.keys(objects).filter(k => k !== 'centerStage').join(", ");
    }

    let lastTime = 0;
    
    function animateFluid(time) {
        // Iterate all objects with liquid to apply a wobbly slosh effect on their top vertices
        Object.values(objects).forEach(obj => {
            if (!obj.userData || !obj.userData.liquid) return;
            const liquid = obj.userData.liquid;
            const geo = liquid.geometry;
            const posAttr = geo.attributes.position;
            
            // if no original positions stored, store them
            if (!geo.userData.originalPositions) {
                geo.userData.originalPositions = new Float32Array(posAttr.array);
            }
            const orig = geo.userData.originalPositions;
            
            // Compute a fake velocity based on recent position deltas
            if (!obj.userData.lastPos) obj.userData.lastPos = new THREE.Vector3().copy(obj.position);
            const currentWorldPos = new THREE.Vector3();
            obj.getWorldPosition(currentWorldPos);
            const delta = currentWorldPos.distanceTo(obj.userData.lastPos);
            obj.userData.lastPos.copy(currentWorldPos);
            
            // Sloshing amplitude depends on movement, decays over time
            if (!obj.userData.sloshAmp) obj.userData.sloshAmp = 0;
            obj.userData.sloshAmp = Math.min(obj.userData.sloshAmp + delta * 2.0, 0.2); // max 0.2
            obj.userData.sloshAmp *= 0.95; // decay
            
            const timeSec = time * 0.005;
            let needsUpdate = false;
            
            for (let i = 0; i < posAttr.count; i++) {
                const y = orig[i * 3 + 1];
                // Only wobble the top vertices (y > 0 in local cylinder space usually)
                if (y > 0.01) {
                    const x = orig[i * 3];
                    const z = orig[i * 3 + 2];
                    
                    // Ripple effect using sine waves over x and z combined with time
                    const wave = Math.sin(x * 10 + timeSec) * Math.cos(z * 10 + timeSec * 1.2);
                    posAttr.array[i * 3 + 1] = y + wave * obj.userData.sloshAmp;
                    needsUpdate = true;
                }
            }
            if (needsUpdate) posAttr.needsUpdate = true;
        });
    }

    function animate(time) {
        TWEEN.update(time);
        animateFluid(time);
        if (window.__orbitControls) window.__orbitControls.update();
        if (renderer && scene && camera) renderer.render(scene, camera);
    }

    function onWindowResize() {
        if (!camera || !renderer || !container) return;
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    }

    return {
        init,
        actionPick,
        actionMoveTo,
        actionPour,
        actionMix,
        actionPlace,
        actionDrawer,
        getEnvironmentState
    };
})();
