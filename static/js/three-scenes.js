/* Atlas — Three.js scenes (globe, particles, neural net) */
(function () {
  if (!window.THREE) return;

  // ---------- Globe ----------
  window.AtlasGlobe = function (mount) {
    const w = mount.clientWidth, h = mount.clientHeight;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 100);
    camera.position.z = 5;
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(2, devicePixelRatio));
    renderer.setSize(w, h);
    mount.appendChild(renderer.domElement);

    // Wireframe sphere
    const geo = new THREE.IcosahedronGeometry(1.6, 4);
    const mat = new THREE.MeshBasicMaterial({ color: 0x6366f1, wireframe: true, transparent: true, opacity: 0.35 });
    const globe = new THREE.Mesh(geo, mat);
    scene.add(globe);

    // Inner glow
    const innerMat = new THREE.MeshBasicMaterial({ color: 0x8b5cf6, transparent: true, opacity: 0.08 });
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1.55, 48, 48), innerMat));

    // Orbit points
    const pts = new THREE.BufferGeometry();
    const N = 600, arr = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = 2 * Math.PI * Math.random();
      const r = 1.62;
      arr[i*3]   = r * Math.sin(phi) * Math.cos(theta);
      arr[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
      arr[i*3+2] = r * Math.cos(phi);
    }
    pts.setAttribute('position', new THREE.BufferAttribute(arr, 3));
    const ptsMat = new THREE.PointsMaterial({ color: 0x06b6d4, size: 0.025, transparent: true, opacity: 0.9 });
    scene.add(new THREE.Points(pts, ptsMat));

    // Outer particle ring
    const ring = new THREE.BufferGeometry();
    const M = 1200, arr2 = new Float32Array(M * 3);
    for (let i = 0; i < M; i++) {
      const r = 2.5 + Math.random() * 1.5;
      const t = Math.random() * Math.PI * 2;
      arr2[i*3] = r * Math.cos(t);
      arr2[i*3+1] = (Math.random() - 0.5) * 2.5;
      arr2[i*3+2] = r * Math.sin(t);
    }
    ring.setAttribute('position', new THREE.BufferAttribute(arr2, 3));
    const ringMesh = new THREE.Points(ring, new THREE.PointsMaterial({ color: 0xffffff, size: 0.012, transparent: true, opacity: 0.5 }));
    scene.add(ringMesh);

    let mouseX = 0, mouseY = 0;
    window.addEventListener('mousemove', (e) => {
      mouseX = (e.clientX / window.innerWidth - 0.5) * 0.5;
      mouseY = (e.clientY / window.innerHeight - 0.5) * 0.5;
    });

    function animate() {
      requestAnimationFrame(animate);
      globe.rotation.y += 0.002;
      globe.rotation.x += 0.0005;
      ringMesh.rotation.y -= 0.0008;
      camera.position.x += (mouseX * 1.2 - camera.position.x) * 0.04;
      camera.position.y += (-mouseY * 1.2 - camera.position.y) * 0.04;
      camera.lookAt(scene.position);
      renderer.render(scene, camera);
    }
    animate();

    new ResizeObserver(() => {
      const W = mount.clientWidth, H = mount.clientHeight;
      camera.aspect = W / H; camera.updateProjectionMatrix();
      renderer.setSize(W, H);
    }).observe(mount);
  };

  // ---------- Floating particles backdrop ----------
  window.AtlasParticles = function (mount, count = 80) {
    const w = mount.clientWidth, h = mount.clientHeight;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 100);
    camera.position.z = 6;
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(2, devicePixelRatio));
    renderer.setSize(w, h);
    mount.appendChild(renderer.domElement);

    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const speeds = [];
    for (let i = 0; i < count; i++) {
      positions[i*3]   = (Math.random() - .5) * 12;
      positions[i*3+1] = (Math.random() - .5) * 8;
      positions[i*3+2] = (Math.random() - .5) * 6;
      speeds.push(Math.random() * 0.003 + 0.001);
    }
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({ color: 0x8b5cf6, size: 0.06, transparent: true, opacity: 0.7 });
    const points = new THREE.Points(geo, mat);
    scene.add(points);

    function animate() {
      requestAnimationFrame(animate);
      const arr = geo.attributes.position.array;
      for (let i = 0; i < count; i++) {
        arr[i*3+1] += speeds[i];
        if (arr[i*3+1] > 4) arr[i*3+1] = -4;
      }
      geo.attributes.position.needsUpdate = true;
      points.rotation.y += 0.0008;
      renderer.render(scene, camera);
    }
    animate();
    new ResizeObserver(() => {
      const W = mount.clientWidth, H = mount.clientHeight;
      camera.aspect = W / H; camera.updateProjectionMatrix();
      renderer.setSize(W, H);
    }).observe(mount);
  };

  // ---------- Neural network ----------
  window.AtlasNeural = function (mount) {
    const w = mount.clientWidth, h = mount.clientHeight;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 100);
    camera.position.z = 8;
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(2, devicePixelRatio));
    renderer.setSize(w, h);
    mount.appendChild(renderer.domElement);

    const layers = [4, 6, 6, 3];
    const nodes = [];
    const nodeGeo = new THREE.SphereGeometry(0.12, 16, 16);
    const nodeMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4 });

    layers.forEach((count, li) => {
      const x = (li - (layers.length - 1) / 2) * 2.2;
      for (let i = 0; i < count; i++) {
        const y = (i - (count - 1) / 2) * 0.9;
        const m = new THREE.Mesh(nodeGeo, nodeMat.clone());
        m.position.set(x, y, 0);
        scene.add(m);
        nodes.push({ mesh: m, layer: li });
      }
    });

    const lineMat = new THREE.LineBasicMaterial({ color: 0x6366f1, transparent: true, opacity: 0.2 });
    for (let li = 0; li < layers.length - 1; li++) {
      const a = nodes.filter(n => n.layer === li);
      const b = nodes.filter(n => n.layer === li + 1);
      a.forEach(n1 => b.forEach(n2 => {
        const g = new THREE.BufferGeometry().setFromPoints([n1.mesh.position, n2.mesh.position]);
        scene.add(new THREE.Line(g, lineMat));
      }));
    }

    let t = 0;
    function animate() {
      requestAnimationFrame(animate);
      t += 0.02;
      nodes.forEach((n, i) => {
        const pulse = 0.5 + 0.5 * Math.sin(t + i * 0.4);
        n.mesh.material.color.setHSL(0.55 + pulse * 0.1, 0.7, 0.4 + pulse * 0.3);
        n.mesh.scale.setScalar(0.8 + pulse * 0.5);
      });
      scene.rotation.y = Math.sin(t * 0.2) * 0.2;
      renderer.render(scene, camera);
    }
    animate();
    new ResizeObserver(() => {
      const W = mount.clientWidth, H = mount.clientHeight;
      camera.aspect = W / H; camera.updateProjectionMatrix();
      renderer.setSize(W, H);
    }).observe(mount);
  };
})();
