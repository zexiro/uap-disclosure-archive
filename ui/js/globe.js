/* globe.js — DisclosureGlobe: hand-rolled three.js globe for the archive.
   Procedural earth (rasterized Natural Earth geojson), fresnel atmosphere,
   starfield, glowing per-record markers with pulse rings, drag-to-rotate
   with inertia, and a hover card linking to record pages.

   Exposes window.DisclosureGlobe:
     await DisclosureGlobe.create(container, opts) → instance
     DisclosureGlobe.buildMarkers(docs, { filter }) → marker[]
     DisclosureGlobe.loadLocations() → fetches geo/locations.json
   Depends on shared.js (escapeHtml, badge, recordUrl, docById, ROOT_REL). */
"use strict";

window.DisclosureGlobe = (function () {
  const THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js";
  const GEO_URL = "geo/ne_110m_admin_0_countries.geojson";
  const LOCATIONS_URL = "geo/locations.json";
  const R = 1; // globe radius in scene units

  const TYPE_COLORS = {
    PDF: "#7bb8d4",
    VID: "#e4b463",
    AUD: "#ffd56b",
    IMG: "#b6c265",
    "Case File": "#7fd4b5",
  };
  const TYPE_SIZES = { PDF: 4.6, VID: 6.2, AUD: 6.0, IMG: 5.2, "Case File": 6.6 };

  let threePromise = null;
  function loadThree() {
    if (window.THREE) return Promise.resolve(window.THREE);
    if (threePromise) return threePromise;
    threePromise = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = THREE_URL;
      s.onload = () => resolve(window.THREE);
      s.onerror = () => { threePromise = null; reject(new Error("three.js failed to load")); };
      document.head.appendChild(s);
    });
    return threePromise;
  }

  async function loadLocations() {
    if (window.GLOBE_LOCATIONS) return window.GLOBE_LOCATIONS;
    try {
      const r = await fetch(LOCATIONS_URL);
      window.GLOBE_LOCATIONS = r.ok ? await r.json() : {};
    } catch (_) { window.GLOBE_LOCATIONS = {}; }
    return window.GLOBE_LOCATIONS;
  }

  // ── Marker building ────────────────────────────────────────────────
  // Per-record coords: precise lat/lng from correlations.json where the
  // pipeline resolved one, else the geocoded centroid of the record's
  // incident_location string. Records sharing a coordinate are spread in
  // a golden-angle spiral so every marker stays individually hoverable.
  function buildMarkers(docs, opts = {}) {
    const filter = opts.filter || null;
    const corr = window.CORRELATIONS || {};
    const locs = window.GLOBE_LOCATIONS || {};

    const coordsByDocId = {};
    for (const [key, entry] of Object.entries(corr)) {
      const off = entry && entry.official;
      const loc = off && off.location;
      if (!loc || loc.lat == null || loc.lng == null) continue;
      let id = null;
      if (typeof docById === "function") {
        if (docById(key)) id = key;
        else {
          const k2 = key.replace(":", "_");
          if (docById(k2)) id = k2;
        }
      }
      if (id) coordsByDocId[id] = [loc.lat, loc.lng];
    }

    const byCoord = new Map();
    for (const d of docs) {
      if (filter && !filter(d)) continue;
      let c = coordsByDocId[d.id];
      if (!c) c = locs[(d.incident_location || "").trim()] || null;
      if (!c) continue;
      const k = c[0].toFixed(2) + "," + c[1].toFixed(2);
      if (!byCoord.has(k)) byCoord.set(k, { lat: c[0], lng: c[1], items: [] });
      byCoord.get(k).items.push(d);
    }

    const markers = [];
    for (const g of byCoord.values()) {
      const n = g.items.length;
      g.items.forEach((d, i) => {
        let lat = g.lat, lng = g.lng;
        if (n > 1) {
          const a = i * 2.399963; // golden angle
          const r = Math.min(4.5, 0.6 * Math.sqrt(i)); // degrees
          lat += r * Math.sin(a) * 0.6;
          lng += r * Math.cos(a) / Math.max(0.35, Math.cos(g.lat * Math.PI / 180));
        }
        lat = Math.max(-85, Math.min(85, lat));
        markers.push({ id: d.id, lat, lng, doc: d });
      });
    }
    return markers;
  }

  // ── Geometry helpers ───────────────────────────────────────────────
  // Must stay consistent with the equirectangular texture mapping used
  // for the sphere (see paintEarth): u = (lng+180)/360.
  function latLngToVec3(lat, lng, radius) {
    const phi = lat * Math.PI / 180;
    const lam = lng * Math.PI / 180;
    return new THREE.Vector3(
      radius * Math.cos(phi) * Math.cos(lam),
      radius * Math.sin(phi),
      -radius * Math.cos(phi) * Math.sin(lam)
    );
  }

  // Rasterize country polygons to an equirectangular canvas texture.
  function paintEarth(geo) {
    const W = 4096, H = 2048;
    const cv = document.createElement("canvas");
    cv.width = W; cv.height = H;
    const ctx = cv.getContext("2d");

    const ocean = ctx.createLinearGradient(0, 0, 0, H);
    ocean.addColorStop(0, "#0a1216");
    ocean.addColorStop(0.5, "#0c1418");
    ocean.addColorStop(1, "#0a1116");
    ctx.fillStyle = ocean;
    ctx.fillRect(0, 0, W, H);

    const px = (lng) => (lng + 180) / 360 * W;
    const py = (lat) => (90 - lat) / 180 * H;

    function drawRing(ring) {
      for (let i = 0; i < ring.length; i++) {
        const [lng, lat] = ring[i];
        if (i === 0) ctx.moveTo(px(lng), py(lat));
        else ctx.lineTo(px(lng), py(lat));
      }
      ctx.closePath();
    }
    function drawFeature(f) {
      const g = f.geometry;
      if (!g) return;
      const polys = g.type === "Polygon" ? [g.coordinates]
        : g.type === "MultiPolygon" ? g.coordinates : [];
      for (const poly of polys) {
        for (const ring of poly) drawRing(ring);
      }
    }

    if (geo && geo.features) {
      // One path, many subpaths — beginPath per ring would wipe everything
      // but the last ring before fill().
      ctx.beginPath();
      for (const f of geo.features) drawFeature(f);
      ctx.fillStyle = "#182b25";
      ctx.fill();
      ctx.strokeStyle = "rgba(123, 184, 212, 0.55)";
      ctx.lineWidth = 2.2;
      ctx.stroke();
    }

    // Vignette poles slightly darker for depth.
    const pole = ctx.createLinearGradient(0, 0, 0, H);
    pole.addColorStop(0, "rgba(4,8,10,0.55)");
    pole.addColorStop(0.18, "rgba(4,8,10,0)");
    pole.addColorStop(0.82, "rgba(4,8,10,0)");
    pole.addColorStop(1, "rgba(4,8,10,0.55)");
    ctx.fillStyle = pole;
    ctx.fillRect(0, 0, W, H);

    const tex = new THREE.CanvasTexture(cv);
    return tex;
  }

  function buildGraticule() {
    const pts = [];
    const step = 20, seg = 4;
    const rr = R * 1.002;
    for (let lng = -180; lng < 180; lng += step) {
      for (let lat = -80; lat < 80; lat += seg) {
        pts.push(latLngToVec3(lat, lng, rr), latLngToVec3(lat + seg, lng, rr));
      }
    }
    for (let lat = -80; lat <= 80; lat += step) {
      for (let lng = -180; lng < 180; lng += seg) {
        pts.push(latLngToVec3(lat, lng, rr), latLngToVec3(lat, lng + seg, rr));
      }
    }
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({
      color: 0x7bb8d4, transparent: true, opacity: 0.055, depthWrite: false,
    });
    return new THREE.LineSegments(geo, mat);
  }

  function buildStars() {
    const N = 1800;
    const pos = new Float32Array(N * 3);
    const col = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const v = new THREE.Vector3().randomDirection().multiplyScalar(38 + Math.random() * 30);
      pos.set([v.x, v.y, v.z], i * 3);
      const b = 0.25 + Math.random() * 0.75;
      const warm = Math.random() < 0.12;
      col.set(warm ? [b, b * 0.85, b * 0.6] : [b * 0.85, b * 0.92, b], i * 3);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    const mat = new THREE.PointsMaterial({
      size: 0.55, vertexColors: true, sizeAttenuation: true,
      transparent: true, opacity: 0.9, depthWrite: false,
    });
    return new THREE.Points(geo, mat);
  }

  // ── Shaders ────────────────────────────────────────────────────────
  const EARTH_VERT = `
    varying vec2 vUv;
    varying vec3 vNormal;
    varying vec3 vView;
    void main() {
      vUv = uv;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vNormal = normalize(normalMatrix * normal);
      vView = -mv.xyz;
      gl_Position = projectionMatrix * mv;
    }`;
  const EARTH_FRAG = `
    uniform sampler2D uMap;
    uniform vec3 uRim;
    varying vec2 vUv;
    varying vec3 vNormal;
    varying vec3 vView;
    void main() {
      vec3 tex = texture2D(uMap, vUv).rgb;
      vec3 N = normalize(vNormal);
      vec3 V = normalize(vView);
      vec3 L = normalize(vec3(-0.45, 0.4, 0.8));
      float diff = max(dot(N, L), 0.0);
      float shade = 0.62 + 0.38 * diff;
      float rim = pow(1.0 - max(dot(N, V), 0.0), 3.2);
      vec3 col = tex * shade + uRim * rim * 0.55;
      gl_FragColor = vec4(col, 1.0);
    }`;

  const ATMO_VERT = `
    varying vec3 vNormal;
    void main() {
      vNormal = normalize(normalMatrix * normal);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`;
  const ATMO_FRAG = `
    uniform vec3 uColor;
    varying vec3 vNormal;
    void main() {
      float intensity = pow(max(0.0, 0.66 - dot(vNormal, vec3(0.0, 0.0, 1.0))), 3.6);
      gl_FragColor = vec4(uColor, 1.0) * intensity * 0.62;
    }`;

  const MARKER_VERT = `
    attribute float aSize;
    attribute vec3 aColor;
    attribute float aPhase;
    attribute float aIndex;
    uniform float uTime;
    uniform float uHover;
    uniform float uPixelRatio;
    uniform float uScale;
    varying vec3 vColor;
    varying float vBoost;
    void main() {
      vColor = aColor;
      float hover = 1.0 - step(0.5, abs(aIndex - uHover));
      vBoost = hover;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      float tw = 0.88 + 0.12 * sin(uTime * 1.5 + aPhase);
      float size = aSize * (1.0 + hover * 1.3) * tw;
      gl_PointSize = size * uPixelRatio * (uScale / -mv.z);
      gl_Position = projectionMatrix * mv;
    }`;
  const MARKER_FRAG = `
    varying vec3 vColor;
    varying float vBoost;
    void main() {
      vec2 uv = gl_PointCoord - 0.5;
      float d = length(uv);
      if (d > 0.5) discard;
      float core = smoothstep(0.5, 0.04, d);
      float hot = smoothstep(0.16, 0.0, d);
      vec3 col = mix(vColor, vec3(1.0), hot * 0.7 + vBoost * 0.25);
      float alpha = core * (0.72 + 0.28 * vBoost);
      gl_FragColor = vec4(col, alpha);
    }`;

  const RING_VERT = `
    attribute float aSize;
    attribute float aAlpha;
    attribute vec3 aColor;
    uniform float uPixelRatio;
    uniform float uScale;
    varying float vAlpha;
    varying vec3 vColor;
    void main() {
      vAlpha = aAlpha;
      vColor = aColor;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = aSize * uPixelRatio * (uScale / -mv.z);
      gl_Position = projectionMatrix * mv;
    }`;
  const RING_FRAG = `
    varying float vAlpha;
    varying vec3 vColor;
    void main() {
      vec2 uv = gl_PointCoord - 0.5;
      float d = length(uv) * 2.0;
      float ring = smoothstep(0.55, 0.8, d) * (1.0 - smoothstep(0.86, 1.0, d));
      if (ring <= 0.001) discard;
      gl_FragColor = vec4(vColor, ring * vAlpha);
    }`;

  // ── Hover card ─────────────────────────────────────────────────────
  function thumbOf(d) {
    return d.thumb_small
      || (Array.isArray(d.thumbnail_local) ? d.thumbnail_local[0] : d.thumbnail_local)
      || "";
  }

  function cardHtml(d) {
    const thumb = thumbOf(d);
    const date = (d.incident_date || "").trim() || "undated";
    const meta = [d.agency, d.incident_location].filter(Boolean).join(" · ");
    const blurb = (d.blurb || "").slice(0, 140);
    return `
      ${thumb
        ? `<div class="ghc-media"><img src="${ROOT_REL}/${thumb}" onerror="this.parentNode.style.display='none'" alt=""/></div>`
        : ""}
      <div class="ghc-body">
        <div class="ghc-top">${badge(d.type)}<span class="ghc-date">${escapeHtml(date)}</span></div>
        <div class="ghc-title">${escapeHtml(d.title || "Untitled record")}</div>
        ${meta ? `<div class="ghc-meta">${escapeHtml(meta)}</div>` : ""}
        ${blurb ? `<div class="ghc-blurb">${escapeHtml(blurb)}${(d.blurb || "").length > 140 ? "…" : ""}</div>` : ""}
        <div class="ghc-link">OPEN FILE →</div>
      </div>`;
  }

  // ── Main factory ───────────────────────────────────────────────────
  async function create(container, opts = {}) {
    await loadThree();
    // Authored hex colors are used verbatim (canvas texture + shader output);
    // skip three's linear-workflow conversion so they match the CSS palette.
    THREE.ColorManagement.enabled = false;

    const accent = (opts.accent
      || getComputedStyle(document.documentElement).getPropertyValue("--accent").trim()
      || "#7fd4b5");
    const cyan = getComputedStyle(document.documentElement).getPropertyValue("--cyan").trim() || "#7bb8d4";
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const interactive = opts.interactive !== false;
    const offsetXFrac = opts.offsetX || 0;

    container.classList.add("dglobe");
    container.innerHTML = "";

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 200);
    const CAM_HOME = 3.38, CAM_MIN = 1.45, CAM_MAX = 4.6;
    let camZ = opts.intro === false || reduced ? CAM_HOME : CAM_MAX;
    camera.position.set(0, 0, camZ);

    const tiltGroup = new THREE.Group();
    const spinGroup = new THREE.Group();
    tiltGroup.add(spinGroup);
    tiltGroup.rotation.x = 0.28;
    tiltGroup.rotation.z = -0.06;
    scene.add(tiltGroup);

    // Earth
    let geo = null;
    try { geo = await (await fetch(GEO_URL)).json(); } catch (_) { /* texture-less fallback */ }
    const earthTex = paintEarth(geo);
    earthTex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    const earth = new THREE.Mesh(
      new THREE.SphereGeometry(R, 96, 64),
      new THREE.ShaderMaterial({
        uniforms: {
          uMap: { value: earthTex },
          uRim: { value: new THREE.Color(accent) },
        },
        vertexShader: EARTH_VERT,
        fragmentShader: EARTH_FRAG,
      })
    );
    spinGroup.add(earth);
    spinGroup.add(buildGraticule());

    const atmo = new THREE.Mesh(
      new THREE.SphereGeometry(R * 1.24, 64, 48),
      new THREE.ShaderMaterial({
        uniforms: { uColor: { value: new THREE.Color(accent) } },
        vertexShader: ATMO_VERT,
        fragmentShader: ATMO_FRAG,
        side: THREE.BackSide,
        blending: THREE.AdditiveBlending,
        transparent: true,
        depthWrite: false,
      })
    );
    scene.add(atmo);
    scene.add(buildStars());

    // Markers
    const markerGeo = new THREE.BufferGeometry();
    const markerMat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uHover: { value: -1 },
        uPixelRatio: { value: renderer.getPixelRatio() },
        uScale: { value: CAM_HOME },
      },
      vertexShader: MARKER_VERT,
      fragmentShader: MARKER_FRAG,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: THREE.AdditiveBlending,
    });
    const markerPoints = new THREE.Points(markerGeo, markerMat);
    spinGroup.add(markerPoints);

    // Pulse rings
    const RING_COUNT = reduced ? 0 : 14;
    const RING_LIFE = 2.6;
    const ringGeo = new THREE.BufferGeometry();
    const ringPos = new Float32Array(RING_COUNT * 3);
    const ringSize = new Float32Array(RING_COUNT);
    const ringAlpha = new Float32Array(RING_COUNT);
    const ringCol = new Float32Array(RING_COUNT * 3);
    ringGeo.setAttribute("position", new THREE.BufferAttribute(ringPos, 3));
    ringGeo.setAttribute("aSize", new THREE.BufferAttribute(ringSize, 1));
    ringGeo.setAttribute("aAlpha", new THREE.BufferAttribute(ringAlpha, 1));
    ringGeo.setAttribute("aColor", new THREE.BufferAttribute(ringCol, 3));
    const ringMat = new THREE.ShaderMaterial({
      uniforms: { uPixelRatio: { value: renderer.getPixelRatio() }, uScale: { value: CAM_HOME } },
      vertexShader: RING_VERT,
      fragmentShader: RING_FRAG,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const ringPoints = new THREE.Points(ringGeo, ringMat);
    spinGroup.add(ringPoints);
    const rings = Array.from({ length: RING_COUNT }, (_, i) => ({ t: (i / RING_COUNT) * RING_LIFE, idx: -1 }));

    // Hover card DOM
    const card = document.createElement("div");
    card.className = "globe-hover-card";
    card.hidden = true;
    container.appendChild(card);

    // ── State ────────────────────────────────────────────────────────
    let markers = [];
    let markerVecs = [];
    let hovered = -1;
    let dragging = false;
    let dragMoved = 0;
    let lastX = 0, lastY = 0;
    let velY = 0, velX = 0;
    let idleAt = performance.now() + 1500;
    let autoSpeed = reduced ? 0 : 0.055;
    let targetCamZ = CAM_HOME;
    let running = true;
    let raf = 0;
    let cardPinned = false;
    let hideTimer = 0;
    let pointer = { x: 0, y: 0, inside: false };
    let destroyed = false;

    function setMarkers(list) {
      markers = list || [];
      markerVecs = markers.map(m => latLngToVec3(m.lat, m.lng, R * 1.012));
      const n = markers.length;
      const pos = new Float32Array(n * 3);
      const size = new Float32Array(n);
      const col = new Float32Array(n * 3);
      const phase = new Float32Array(n);
      const index = new Float32Array(n);
      const c = new THREE.Color();
      markers.forEach((m, i) => {
        pos.set([markerVecs[i].x, markerVecs[i].y, markerVecs[i].z], i * 3);
        const type = (m.doc && m.doc.type) || "";
        c.set(TYPE_COLORS[type] || accent);
        col.set([c.r, c.g, c.b], i * 3);
        size[i] = TYPE_SIZES[type] || 4.6;
        phase[i] = Math.random() * Math.PI * 2;
        index[i] = i;
      });
      markerGeo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      markerGeo.setAttribute("aSize", new THREE.BufferAttribute(size, 1));
      markerGeo.setAttribute("aColor", new THREE.BufferAttribute(col, 3));
      markerGeo.setAttribute("aPhase", new THREE.BufferAttribute(phase, 1));
      markerGeo.setAttribute("aIndex", new THREE.BufferAttribute(index, 1));
      markerGeo.computeBoundingSphere();
      hovered = -1;
      markerMat.uniforms.uHover.value = -1;
      hideCard();
      if (typeof opts.onCount === "function") opts.onCount(n);
    }

    // ── Card show/hide ───────────────────────────────────────────────
    function placeCard(px, py) {
      const rect = container.getBoundingClientRect();
      const cw = card.offsetWidth, ch = card.offsetHeight;
      let x = px + 18, y = py + 14;
      if (x + cw > rect.width - 8) x = px - cw - 14;
      if (y + ch > rect.height - 8) y = Math.max(8, py - ch - 10);
      card.style.left = x + "px";
      card.style.top = y + "px";
    }
    function showCard(i) {
      const m = markers[i];
      if (!m || !m.doc) return;
      card.innerHTML = cardHtml(m.doc);
      card.hidden = false;
      placeCard(pointer.x, pointer.y);
    }
    function hideCard() {
      if (cardPinned) return;
      card.hidden = true;
      hovered = -1;
      markerMat.uniforms.uHover.value = -1;
      container.style.cursor = "";
    }
    function scheduleHide() {
      clearTimeout(hideTimer);
      hideTimer = setTimeout(() => { cardPinned = false; hideCard(); }, 140);
    }
    card.addEventListener("pointerenter", () => { cardPinned = true; clearTimeout(hideTimer); });
    card.addEventListener("pointerleave", () => { cardPinned = false; scheduleHide(); });
    card.addEventListener("click", () => {
      const m = markers[hovered];
      if (m) location.href = recordUrl(m.id);
    });

    // ── Pointer interaction ──────────────────────────────────────────
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.016;
    const ndc = new THREE.Vector2();

    function pickMarker() {
      if (!pointer.inside || dragging || cardPinned || !markers.length) return;
      // Keep hover precision consistent across the zoom range.
      raycaster.params.Points.threshold = 0.016 * (camera.position.z / CAM_HOME);
      const rect = renderer.domElement.getBoundingClientRect();
      ndc.set(
        (pointer.px / rect.width) * 2 - 1,
        -(pointer.py / rect.height) * 2 + 1
      );
      raycaster.setFromCamera(ndc, camera);
      const hits = raycaster.intersectObject(markerPoints);
      const idx = hits.length ? hits[0].index : -1;
      if (idx !== hovered) {
        hovered = idx;
        markerMat.uniforms.uHover.value = idx;
        if (idx >= 0) {
          container.style.cursor = "pointer";
          showCard(idx);
        } else if (!cardPinned) {
          container.style.cursor = "";
          card.hidden = true;
        }
      } else if (idx >= 0 && !cardPinned) {
        placeCard(pointer.x, pointer.y);
      }
    }

    function trackPointer(e) {
      const rect = container.getBoundingClientRect();
      pointer.x = e.clientX - rect.left;
      pointer.y = e.clientY - rect.top;
      pointer.px = pointer.x; pointer.py = pointer.y;
      pointer.inside = pointer.x >= 0 && pointer.y >= 0
        && pointer.x <= rect.width && pointer.y <= rect.height;
    }

    function onPointerDown(e) {
      if (!interactive) return;
      trackPointer(e);
      pickMarker(); // resolves hover for touch taps (no prior pointermove)
      dragging = true;
      dragMoved = 0;
      lastX = e.clientX; lastY = e.clientY;
      velY = 0; velX = 0;
      container.setPointerCapture && container.setPointerCapture(e.pointerId);
      container.style.cursor = "grabbing";
    }
    function onPointerMove(e) {
      trackPointer(e);
      idleAt = performance.now();
      if (!dragging) return;
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      dragMoved += Math.abs(dx) + Math.abs(dy);
      const k = 0.005;
      spinGroup.rotation.y += dx * k;
      tiltGroup.rotation.x = Math.max(-1.1, Math.min(1.1, tiltGroup.rotation.x + dy * k));
      velY = dx * k; velX = dy * k;
      if (dragMoved > 8) { cardPinned = false; card.hidden = true; }
    }
    function onPointerUp(e) {
      if (!dragging) return;
      dragging = false;
      container.style.cursor = hovered >= 0 ? "pointer" : "";
      idleAt = performance.now();
      if (dragMoved <= 6 && hovered >= 0) {
        const m = markers[hovered];
        if (m) location.href = recordUrl(m.id);
      }
    }
    function onPointerLeave() {
      pointer.inside = false;
      if (!cardPinned) scheduleHide();
    }
    function onWheel(e) {
      if (!interactive || !opts.zoom) return;
      e.preventDefault();
      targetCamZ = Math.max(CAM_MIN, Math.min(CAM_MAX, targetCamZ + e.deltaY * 0.0016));
      idleAt = performance.now();
    }

    if (interactive) {
      container.addEventListener("pointerdown", onPointerDown);
      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", onPointerUp);
      container.addEventListener("pointerleave", onPointerLeave);
      container.addEventListener("wheel", onWheel, { passive: false });
      container.style.touchAction = "none";
    }

    // ── Resize ───────────────────────────────────────────────────────
    function resize() {
      const w = container.clientWidth || 1;
      const h = container.clientHeight || 1;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      const wide = w >= 900;
      const off = wide ? offsetXFrac : 0;
      if (off) camera.setViewOffset(w, h, -w * off, 0, w, h);
      else camera.clearViewOffset();
      camera.updateProjectionMatrix();
    }
    const ro = new ResizeObserver(resize);
    ro.observe(container);
    resize();

    // ── Render loop ──────────────────────────────────────────────────
    const clock = new THREE.Clock();
    function frame() {
      if (destroyed) return;
      raf = requestAnimationFrame(frame);
      if (!running) return;
      const dt = Math.min(clock.getDelta(), 0.05);
      const t = clock.elapsedTime;

      // Intro dolly
      if (camZ > CAM_HOME + 0.001) {
        camZ += (CAM_HOME - camZ) * Math.min(1, dt * 2.2);
        camera.position.z = camZ;
      }
      // Wheel zoom easing
      if (Math.abs(camera.position.z - targetCamZ) > 0.001 && camZ <= CAM_HOME + 0.001) {
        camera.position.z += (targetCamZ - camera.position.z) * Math.min(1, dt * 7);
      }

      // Inertia + auto-rotate
      if (!dragging) {
        if (Math.abs(velY) > 0.00004 || Math.abs(velX) > 0.00004) {
          spinGroup.rotation.y += velY;
          tiltGroup.rotation.x = Math.max(-1.1, Math.min(1.1, tiltGroup.rotation.x + velX));
          velY *= 0.94; velX *= 0.94;
        } else if (autoSpeed && performance.now() - idleAt > 2800) {
          spinGroup.rotation.y += autoSpeed * dt;
        }
      }

      markerMat.uniforms.uTime.value = t;
      markerMat.uniforms.uScale.value = camera.position.z;
      ringMat.uniforms.uScale.value = camera.position.z;

      // Pulse rings
      if (RING_COUNT && markers.length) {
        for (let i = 0; i < RING_COUNT; i++) {
          const ring = rings[i];
          ring.t += dt;
          if (ring.t >= RING_LIFE || ring.idx < 0 || ring.idx >= markers.length) {
            ring.t = 0;
            ring.idx = Math.floor(Math.random() * markers.length);
            const v = markerVecs[ring.idx];
            ringPos.set([v.x, v.y, v.z], i * 3);
            const mcol = new THREE.Color(TYPE_COLORS[(markers[ring.idx].doc || {}).type] || accent);
            ringCol.set([mcol.r, mcol.g, mcol.b], i * 3);
          }
          const p = ring.t / RING_LIFE;
          ringSize[i] = 8 + p * 46;
          ringAlpha[i] = (1 - p) * (1 - p) * 0.7;
        }
        ringGeo.attributes.position.needsUpdate = true;
        ringGeo.attributes.aSize.needsUpdate = true;
        ringGeo.attributes.aAlpha.needsUpdate = true;
        ringGeo.attributes.aColor.needsUpdate = true;
      }

      pickMarker();
      renderer.render(scene, camera);
    }
    frame();

    return {
      setMarkers,
      resize,
      pause() { running = false; },
      resume() { if (!running) { running = true; clock.getDelta(); } },
      get count() { return markers.length; },
      destroy() {
        destroyed = true;
        cancelAnimationFrame(raf);
        ro.disconnect();
        renderer.dispose();
        container.innerHTML = "";
      },
    };
  }

  return { create, buildMarkers, loadLocations, loadThree };
})();
