const $ = (sel) => document.querySelector(sel);

// ---------- API base configuration (for Vercel frontend + remote backend) ----------
// Pick API base from URL ?api=..., else localStorage, else same-origin
const urlParams = new URLSearchParams(window.location.search || "");
let API_BASE = (urlParams.get('api') || localStorage.getItem('VISOR_API_BASE') || '').trim();
if (urlParams.get('api')) {
  localStorage.setItem('VISOR_API_BASE', API_BASE);
}
function apiPath(path) {
  if (!API_BASE) return path; // same-origin
  if (API_BASE.endsWith('/') && path.startsWith('/')) return API_BASE.slice(0, -1) + path;
  return API_BASE + path;
}
const video = $("#video");
const overlay = $("#overlay");
const ctx = overlay.getContext("2d");
const hidden = $("#hiddenCanvas");
const startBtn = $("#startBtn");
const stopBtn = $("#stopBtn");
const intervalMsInp = $("#intervalMs");
const captionEl = $("#caption");
const vqaEl = $("#vqa");
const detList = $("#detections");
const speakToggle = $("#speakToggle");
const questionInp = $("#question");
const vqaToggle = $("#vqaToggle");
const micBtn = document.querySelector('#micBtn');
const speakVqaToggle = document.querySelector('#speakVqaToggle');
const assistantToggle = document.querySelector('#assistantToggle');
const guideToggle = document.querySelector('#guideToggle');
const assistantStatus = document.querySelector('#assistantStatus');
const guideStatus = document.querySelector('#guideStatus');
const connectVitalsBtn = document.querySelector('#connectVitalsBtn');
const heartRateEl = document.querySelector('#heartRate');
const notifyVitalsToggle = document.querySelector('#notifyVitalsToggle');
const speakVitalsNowBtn = document.querySelector('#speakVitalsNowBtn');

let timer = null;
let stream = null;
let lastText = ""; // last received narrative/caption
let lastSpokenCaption = ""; // last spoken text
let isSpeaking = false; // Track if currently speaking
let lastSpeakTime = 0; // ms
const MIN_SPEAK_INTERVAL_MS = 1200; // throttle speaking frequency
const ASSISTANT_INTERVAL_MS = 5000; // periodic guidance cadence
let assistantTimer = null;
let lastVqaSpoken = "";
const GUIDE_INTERVAL_MS = 1500; // faster cadence for obstacle updates
let guideTimer = null;
let lastGuideUtterance = "";
let latestDetections = [];
let inFlight = false; // prevent overlapping /analyze requests

// --- Web Bluetooth: Heart Rate ---
let hrDevice = null;
let hrServer = null;
let hrChar = null;
let vitalsTimer = null;

function parseHeartRate(value) {
  // See Bluetooth SIG Heart Rate Measurement characteristic
  const data = new DataView(value.buffer);
  const flags = data.getUint8(0);
  let index = 1;
  let hr;
  if (flags & 0x01) { // 16-bit
    hr = data.getUint16(index, /*littleEndian=*/true); index += 2;
  } else { // 8-bit
    hr = data.getUint8(index); index += 1;
  }
  return hr;
}

async function connectHeartRate() {
  if (!navigator.bluetooth) {
    alert('Web Bluetooth not supported in this browser. Use Chrome on desktop.');
    return;
  }
  try {
    connectVitalsBtn.disabled = true;
    hrDevice = await navigator.bluetooth.requestDevice({
      filters: [{ services: ['heart_rate'] }],
      optionalServices: ['battery_service']
    });
    hrServer = await hrDevice.gatt.connect();
    const service = await hrServer.getPrimaryService('heart_rate');
    hrChar = await service.getCharacteristic('heart_rate_measurement');
    await hrChar.startNotifications();
    hrChar.addEventListener('characteristicvaluechanged', (event) => {
      const bpm = parseHeartRate(event.target.value);
      if (heartRateEl) heartRateEl.textContent = bpm ? `${bpm} bpm` : '—';
    });
    if (heartRateEl) heartRateEl.textContent = 'Connected';
    console.log('❤️ Heart Rate connected');
  } catch (e) {
    console.error('Heart Rate connect error:', e);
    alert('Could not connect to a Heart Rate device. Your watch may not expose BLE Heart Rate.');
    connectVitalsBtn.disabled = false;
  }
}

async function fetchAndSpeakVitals() {
  try {
    const resp = await fetch(apiPath('/vitals/summary'));
    const data = await resp.json();
    const hr = data?.vitals?.heart_rate;
    if (heartRateEl && (typeof hr === 'number')) heartRateEl.textContent = `${hr} bpm`;
    const suggestion = (data?.suggestion || '').trim();
    if (suggestion) speak(suggestion);
  } catch (e) {
    console.error('Vitals summary error:', e);
  }
}

function describeGuidance(det, canvasW, canvasH) {
  if (!det || det.length === 0 || !canvasW || !canvasH) return "";
  // Pick the most salient detection by area * confidence
  let best = null;
  let bestScore = -1;
  for (const d of det) {
    const [x1, y1, x2, y2] = d.xyxy;
    const area = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const score = area * (d.confidence ?? 0.0);
    if (score > bestScore) { bestScore = score; best = d; }
  }
  if (!best) return "";
  const [bx1, by1, bx2, by2] = best.xyxy;
  const cx = (bx1 + bx2) / 2;
  const cy = (by1 + by2) / 2;
  const areaRatio = ((bx2 - bx1) * (by2 - by1)) / (canvasW * canvasH + 1e-6);
  const cls = (best.class_name || String(best.class_id || "object")).toLowerCase();

  // Direction buckets based on horizontal thirds
  let dir = "ahead";
  if (cx < canvasW / 3) dir = "left";
  else if (cx > (2 * canvasW) / 3) dir = "right";
  else dir = "ahead";

  // Distance proxy from box area
  let dist = "far";
  if (areaRatio > 0.20) dist = "very close";
  else if (areaRatio > 0.10) dist = "close";
  else if (areaRatio > 0.04) dist = "near";

  // Phrase
  if (dir === "ahead") return `${cls} ${dist} ahead`;
  return `${cls} ${dist} to the ${dir}`;
}

// Preferred voice (loaded asynchronously)
let preferredVoice = null;
function initVoicesOnce() {
  const setVoice = () => {
    const voices = window.speechSynthesis?.getVoices?.() || [];
    // Prefer an English local voice if available
    preferredVoice = voices.find(v => v.lang?.startsWith('en') && v.localService) ||
                     voices.find(v => v.lang?.startsWith('en')) || null;
  };
  try {
    setVoice();
    if (typeof speechSynthesis !== 'undefined') {
      speechSynthesis.onvoiceschanged = () => setVoice();
    }
  } catch (_) {}
}
initVoicesOnce();

function speak(text, opts = {}) {
  if (!speakToggle.checked) {
    console.log("🔇 TTS disabled by toggle");
    return;
  }
  if (!text) {
    console.log("⚠️ No text to speak");
    return;
  }
  if (recognizing) { console.log('⏸️ TTS paused during voice input'); return; }
  
  // Debounce: don't overlap or spam
  const now = Date.now();
  if (isSpeaking) return;
  if (text === lastSpokenCaption && now - lastSpeakTime < MIN_SPEAK_INTERVAL_MS) return;
  
  console.log("🔊 Speaking:", text);
  
  // Improve the text for better TTS
  let improvedText = text;
  
  // Add natural pauses with commas
  improvedText = improvedText.replace(/ and /g, ', and ');
  improvedText = improvedText.replace(/ with /g, ', with ');
  
  // Remove repeated words (like "self self self")
  improvedText = improvedText.replace(/\b(\w+)(\s+\1\b)+/gi, '$1');
  
  const u = new SpeechSynthesisUtterance(improvedText);
  u.rate = (typeof opts.rate === 'number') ? opts.rate : 0.9; // Slightly slower for clarity
  u.pitch = 1.0;
  u.volume = 1.0;
  u.lang = 'en-US';
  if (preferredVoice) u.voice = preferredVoice;
  
  // Add event listeners
  u.onstart = () => {
    isSpeaking = true;
    console.log("✓ TTS started");
  };
  
  u.onend = () => {
    isSpeaking = false;
    lastSpokenCaption = text; // Remember what we spoke
    lastSpeakTime = Date.now();
    console.log("✓ TTS ended");
  };
  
  u.onerror = (e) => {
    isSpeaking = false;
    console.error("✗ TTS error:", e);
    // Chrome may auto-cancel; try resuming and retry once
    try { window.speechSynthesis.resume(); } catch(_) {}
    if (!u._retried && e?.error === 'canceled') {
      u._retried = true;
      setTimeout(() => {
        try { window.speechSynthesis.speak(u); } catch(_) {}
      }, 200);
    }
  };
  
  // Speak immediately
  window.speechSynthesis.speak(u);
}

function drawDetections(det) {
  const { width, height } = overlay;
  ctx.clearRect(0, 0, width, height);
  if (!det || !Array.isArray(det)) return;
  ctx.lineWidth = 2;
  detList.innerHTML = "";
  for (const d of det) {
    const [x1, y1, x2, y2] = d.xyxy;
    ctx.strokeStyle = "#7cc4ff";
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${d.class_name ?? d.class_id} ${(d.confidence*100).toFixed(1)}%`;
    ctx.fillStyle = "rgba(10,20,50,0.8)";
    ctx.fillRect(x1, Math.max(y1 - 18, 0), ctx.measureText(label).width + 10, 18);
    ctx.fillStyle = "#e6eef8";
    ctx.font = "12px Inter, Arial";
    ctx.fillText(label, x1 + 4, Math.max(y1 - 5, 12));

    const li = document.createElement("li");
    li.innerHTML = `<span class="cls">${d.class_name ?? d.class_id}</span> <span class="conf">${(d.confidence*100).toFixed(1)}%</span>`;
    detList.appendChild(li);
  }
}

let isRunning = false;
let currentAbort = null;

async function captureAndSend() {
  if (!isRunning) return;
  if (inFlight) return; // wait for previous request to finish
  if (!video.videoWidth || !video.videoHeight) return;
  const w = video.videoWidth; const h = video.videoHeight;
  overlay.width = w; overlay.height = h;
  hidden.width = w; hidden.height = h;
  const hctx = hidden.getContext("2d");
  hctx.drawImage(video, 0, 0, w, h);
  // Debug: mark that we're sending a frame
  console.log(`➡️ sending frame ${w}x${h}`);
  const blob = await new Promise((res) => hidden.toBlob(res, "image/jpeg", 0.8));
  const fd = new FormData();
  fd.append("file", blob, "frame.jpg");
  let q = questionInp.value.trim();
  if (vqaToggle.checked) {
    if (!q) q = "What is in front of me?";
    fd.append("question", q);
  }
  try {
    inFlight = true;
    currentAbort = new AbortController();
    const resp = await fetch(apiPath("/analyze"), { method: "POST", body: fd, signal: currentAbort.signal });
    const data = await resp.json();
    const detCount = Array.isArray(data.detections) ? data.detections.length : 0;
    console.log(`📦 /analyze: ready=${data.ready} det=${detCount} src=${data.narrative_source} cap='${(data.caption||'').slice(0,40)}'`);
    if (detCount && data.detections?.[0]) {
      const d0 = data.detections[0];
      console.log(`🟦 first box: ${JSON.stringify(d0)}`);
    }
    const narrative = (data.narrative ?? "").trim();
    const captionCurrent = (data.caption ?? "").trim();
    const current = narrative || captionCurrent;

    // Log reasoner status for debugging
    if (typeof data.reasoner_enabled !== 'undefined') {
      console.log(`🤖 Reasoner enabled: ${data.reasoner_enabled} (${data.narrative_source || 'n/a'})`);
    }

    // Speak on change with debounce
    if (current && current !== lastText) {
      speak(`${current}`);
      lastText = current;
    }

    // Display caption (prefer narrative)
    captionEl.textContent = current || "—";
    vqaEl.textContent = data.vqa_answer ?? "—";
    drawDetections(data.detections);

    // Update latest detections for guidance mode
    latestDetections = Array.isArray(data.detections) ? data.detections : [];

    // Speak VQA answers if enabled
    const ans = (data.vqa_answer ?? '').trim();
    if (vqaToggle.checked && speakVqaToggle?.checked && ans && ans !== lastVqaSpoken) {
      speak(`Answer: ${ans}`);
      lastVqaSpoken = ans;
    }
  } catch (e) {
    // Ignore expected aborts when we cancel in-flight requests between frames or on Stop
    if (e && (e.name === 'AbortError' || e.code === 20)) {
      return;
    }
    console.error("❌ Request error:", e);
  } finally {
    inFlight = false;
  }
}

async function start() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ 
      video: { facingMode: "environment" }, 
      audio: false 
    });
    video.srcObject = stream;
    // Ensure no microphone loopback
    try { video.muted = true; video.volume = 0; } catch (_) {}
    startBtn.disabled = true;
    stopBtn.disabled = false;
    
    // Reset state
    lastText = "";
    lastSpokenCaption = ""; // Reset spoken tracking
    isSpeaking = false;
    lastSpeakTime = 0;
    lastVqaSpoken = "";
    
    isRunning = true;
    // Use setInterval but skip if inFlight; effective rate adapts to backend speed
    const tick = () => captureAndSend();
    timer = setInterval(tick, Math.max(500, Number(intervalMsInp.value) || 1500));
    
    console.log("🎥 Camera started");

    // Start assistant timer if enabled
    if (assistantToggle?.checked) {
      assistantTimer = setInterval(() => {
        // Suppress assistant summaries if Guide is running (to avoid overlap)
        if (guideToggle?.checked) return;
        if (speakToggle.checked && lastText) speak(`${lastText}`);
      }, ASSISTANT_INTERVAL_MS);
    }

  // Start guidance timer if enabled
  if (guideToggle?.checked) {
    guideTimer = setInterval(() => {
      const w = overlay.width, h = overlay.height;
      const phrase = describeGuidance(latestDetections, w, h);
      if (!phrase) return;
      if (phrase !== lastGuideUtterance) {
        speak(`Guidance: ${phrase}`, { rate: 1.0 });
        lastGuideUtterance = phrase;
      }
    }, GUIDE_INTERVAL_MS);
  }
  } catch (e) {
    alert("Camera access failed. Check permissions.");
    console.error(e);
  }
}

function stop() {
  if (timer) { clearInterval(timer); timer = null; }
  if (assistantTimer) { clearInterval(assistantTimer); assistantTimer = null; }
  if (guideTimer) { clearInterval(guideTimer); guideTimer = null; }
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  isRunning = false;
  if (currentAbort) { try { currentAbort.abort(); } catch (_) {} currentAbort = null; }
  window.speechSynthesis.cancel(); // Stop any ongoing speech
  startBtn.disabled = false; 
  stopBtn.disabled = true;
  console.log("🛑 Camera stopped");
  // Reset badges
  if (assistantStatus) assistantStatus.textContent = 'Assistant: Off';
  if (guideStatus) guideStatus.textContent = 'Guide: Off';
}

startBtn.addEventListener("click", start);
// Ensure TTS is active after a user gesture (Chrome can suspend it)
startBtn.addEventListener("click", () => { try { window.speechSynthesis.resume(); } catch(_) {} }, { once: true });
stopBtn.addEventListener("click", stop);
window.addEventListener("beforeunload", stop);

// Test TTS on page load
console.log("🔊 TTS available:", 'speechSynthesis' in window);
if ('speechSynthesis' in window) {
  console.log("📝 To test TTS, run: speechSynthesis.speak(new SpeechSynthesisUtterance('test'))");
}

// Voice input for VQA using Web Speech API
let recognition = null;
let recognizing = false;
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
  const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new Rec();
  recognition.lang = 'en-US';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  recognition.continuous = false;
  recognition.onresult = (event) => {
    const transcript = (event.results?.[0]?.[0]?.transcript || '').trim();
    if (transcript) {
      questionInp.value = transcript;
      vqaToggle.checked = true;
      // Trigger an immediate capture to answer
      captureAndSend();
    }
  };
  recognition.onstart = () => { recognizing = true; console.log('🎙️ start'); };
  recognition.onend = () => { recognizing = false; console.log('🎙️ end'); };
  recognition.onerror = (e) => {
    console.error('🎙️ Speech error', e);
    const msg = (e?.error === 'network')
      ? 'Voice recognition needs internet and works best on Chrome over HTTPS or localhost.'
      : 'Voice recognition error. Please allow microphone permissions and try again.';
    alert(msg);
  };
}

if (micBtn && recognition) {
  micBtn.addEventListener('click', async () => {
    try {
      // Ensure mic permission first to avoid network/permission edge cases
      await navigator.mediaDevices.getUserMedia({ audio: true });
      recognition.start();
      console.log('🎙️ Listening for VQA question...');
    } catch (e) {
      console.error('🎙️ Cannot start recognition', e);
      alert('Microphone not available. Please allow mic permissions.');
    }
  });
}

// Toggle assistant timer live
assistantToggle?.addEventListener('change', () => {
  if (assistantToggle.checked) {
    if (!assistantTimer) {
      assistantTimer = setInterval(() => {
        if (speakToggle.checked && lastText) speak(`${lastText}`);
      }, ASSISTANT_INTERVAL_MS);
    }
    if (assistantStatus) assistantStatus.textContent = 'Assistant: On';
  } else {
    if (assistantTimer) { clearInterval(assistantTimer); assistantTimer = null; }
    if (assistantStatus) assistantStatus.textContent = 'Assistant: Off';
  }
});

// Toggle guide timer live
guideToggle?.addEventListener('change', () => {
  if (guideToggle.checked) {
    if (!guideTimer) {
      guideTimer = setInterval(() => {
        const w = overlay.width, h = overlay.height;
        const phrase = describeGuidance(latestDetections, w, h);
        if (!phrase) return;
        if (phrase !== lastGuideUtterance) {
          speak(`Guidance: ${phrase}`, { rate: 1.0 });
          lastGuideUtterance = phrase;
        }
      }, GUIDE_INTERVAL_MS);
    }
    if (guideStatus) guideStatus.textContent = 'Guide: On';
  } else {
    if (guideTimer) { clearInterval(guideTimer); guideTimer = null; }
    if (guideStatus) guideStatus.textContent = 'Guide: Off';
  }
});

// Connect vitals button
connectVitalsBtn?.addEventListener('click', connectHeartRate);

// Vitals notify toggle
notifyVitalsToggle?.addEventListener('change', () => {
  if (notifyVitalsToggle.checked) {
    if (!vitalsTimer) {
      vitalsTimer = setInterval(fetchAndSpeakVitals, 5 * 60 * 1000);
    }
  } else {
    if (vitalsTimer) { clearInterval(vitalsTimer); vitalsTimer = null; }
  }
});

// Speak vitals now
speakVitalsNowBtn?.addEventListener('click', fetchAndSpeakVitals);