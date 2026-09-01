/**
 * Real-Time Voice Assistant Client Logic
 * Dual Visualizer (FFT Frequency Bars + Oscilloscope), Audio Recorder, Multi-Session Manager.
 */

let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;
let analyser = null;
let animationId = null;
let currentSessionId = "default";

const micBtn = document.getElementById("micBtn");
const statusText = document.getElementById("status");
const chatContainer = document.getElementById("chat");
const canvas = document.getElementById("visualizer");
const canvasCtx = canvas.getContext("2d");
const sessionListContainer = document.getElementById("sessionList");

function resizeCanvas() {
  if (canvas && canvas.parentElement) {
    canvas.width = canvas.parentElement.clientWidth - 44;
    canvas.height = 64;
  }
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

function drawDualVisualizer() {
  if (!analyser) return;

  const bufferLength = analyser.frequencyBinCount;
  const timeData = new Uint8Array(bufferLength);
  const freqData = new Uint8Array(bufferLength);

  analyser.getByteTimeDomainData(timeData);
  analyser.getByteFrequencyData(freqData);

  canvasCtx.fillStyle = "rgba(10, 14, 22, 0.4)";
  canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

  // 1. Draw FFT Frequency Bars in background
  const barWidth = (canvas.width / 48);
  for (let i = 0; i < 48; i++) {
    const barHeight = (freqData[i * 2] / 255) * (canvas.height * 0.85);
    const x = i * (barWidth + 2);
    const y = canvas.height - barHeight;

    canvasCtx.fillStyle = isRecording
      ? `rgba(0, 242, 254, ${0.15 + (barHeight / canvas.height) * 0.5})`
      : "rgba(255, 255, 255, 0.05)";
    canvasCtx.fillRect(x, y, barWidth, barHeight);
  }

  // 2. Draw Smooth Oscilloscope Waveform in foreground
  canvasCtx.lineWidth = 2;
  canvasCtx.strokeStyle = isRecording ? "#00f2fe" : "#334155";
  canvasCtx.beginPath();

  const sliceWidth = (canvas.width * 1.0) / bufferLength;
  let x = 0;

  for (let i = 0; i < bufferLength; i++) {
    const v = timeData[i] / 128.0;
    const y = v * (canvas.height / 2);

    if (i === 0) {
      canvasCtx.moveTo(x, y);
    } else {
      canvasCtx.lineTo(x, y);
    }
    x += sliceWidth;
  }

  canvasCtx.lineTo(canvas.width, canvas.height / 2);
  canvasCtx.stroke();

  animationId = requestAnimationFrame(drawDualVisualizer);
}

async function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);
    drawDualVisualizer();

    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
      await processVoiceInput(audioBlob);
      stream.getTracks().forEach((track) => track.stop());
    };

    mediaRecorder.start();
    isRecording = true;
    micBtn.classList.add("recording");
    micBtn.innerHTML = "⏹️";
    statusText.innerText = "Listening live... Click to process.";
  } catch (err) {
    alert("Microphone access error: " + err.message);
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    isRecording = false;
    micBtn.classList.remove("recording");
    micBtn.innerHTML = "🎤";
    statusText.innerText = "Processing audio & reasoning...";
  }
}

async function processVoiceInput(blob) {
  const formData = new FormData();
  formData.append("file", blob, "speech.wav");
  const useDsp = document.getElementById("dspToggle").checked;
  const useSpectralSub = document.getElementById("spectralSubToggle").checked;

  try {
    const response = await fetch(
      `/api/process-audio?session_id=${currentSessionId}&use_dsp=${useDsp}&apply_spectral_sub=${useSpectralSub}`,
      {
        method: "POST",
        body: formData,
      }
    );

    const data = await response.json();

    if (data.user) {
      appendMessage("user", data.user);
    }

    if (data.reply) {
      const sttMs = data.metrics?.stt_latency_sec ? `${data.metrics.stt_latency_sec}s STT` : "";
      const llmMs = data.metrics?.llm_latency_sec ? `${data.metrics.llm_latency_sec}s LLM` : "";
      const dspInfo = data.metrics?.dsp?.applied ? "DSP: On" : "DSP: Off";
      const meta = `${data.latency_sec}s total • ${sttMs} • ${llmMs} • ${dspInfo}`;

      appendMessage("assistant", data.reply, meta);

      if (document.getElementById("ttsToggle").checked) {
        speakBrowserTTS(data.reply);
      }
    } else if (data.error) {
      appendMessage("system", "⚠️ " + data.error);
    }

    statusText.innerText = "Ready! Click microphone to speak again.";
    loadSessions();
  } catch (err) {
    appendMessage("system", "❌ Backend communication error: " + err.message);
    statusText.innerText = "Connection error. Ensure web_app.py is running.";
  }
}

function appendMessage(sender, text, meta = "") {
  const msgDiv = document.createElement("div");
  msgDiv.className = `msg ${sender}`;
  msgDiv.innerHTML = text;

  if (meta) {
    const metaSpan = document.createElement("span");
    metaSpan.className = "meta-tag";
    metaSpan.innerText = meta;
    msgDiv.appendChild(metaSpan);
  }

  chatContainer.appendChild(msgDiv);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function speakBrowserTTS(text) {
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  }
}

async function loadSessions() {
  try {
    const res = await fetch("/api/sessions");
    const data = await res.json();
    sessionListContainer.innerHTML = "";

    data.sessions.forEach((s) => {
      const item = document.createElement("div");
      item.className = `session-item ${s.session_id === currentSessionId ? "active" : ""}`;
      item.innerHTML = `<span>Session: ${s.session_id}</span><span style="color:var(--text-muted); font-size:0.75rem">${s.turn_count} turns</span>`;
      item.onclick = () => switchSession(s.session_id);
      sessionListContainer.appendChild(item);
    });
  } catch (e) {
    console.error("Session load error", e);
  }
}

async function switchSession(sid) {
  currentSessionId = sid;
  chatContainer.innerHTML = "";
  try {
    const res = await fetch(`/api/sessions/${sid}`);
    const data = await res.json();
    if (data.turns && data.turns.length > 0) {
      data.turns.forEach((t) => appendMessage(t.role, t.content, `${t.latency_sec}s latency`));
    } else {
      chatContainer.innerHTML = `<div class="msg system">✨ Switched to session: ${sid}</div>`;
    }
  } catch (e) {
    chatContainer.innerHTML = `<div class="msg system">✨ Active session: ${sid}</div>`;
  }
  loadSessions();
}

async function createNewSession() {
  const newSid = "s_" + Math.random().toString(36).substring(2, 7);
  await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: newSid }),
  });
  switchSession(newSid);
}

async function exportCurrentSession() {
  window.open(`/api/sessions/${currentSessionId}/export?format=markdown`, "_blank");
}

// Initial session load & visualizer start
loadSessions();
drawDualVisualizer();
