/* Voice RAG frontend.
 *
 * Audio path:
 *   getUserMedia → AudioContext → AudioWorklet (falls back to ScriptProcessor)
 *   → downsample to 16 kHz PCM16-LE → WebSocket → server → Sarvam
 *
 * We target ~100ms send intervals (1600 samples × 1 ch × 2 bytes = 3.2 kB/frame)
 * rather than the old 2048-sample ScriptProcessor frames (~42ms at 48kHz, ~128ms
 * at 16kHz) which caused audible glitching and deprecation warnings.
 *
 * No API keys here. All credentials stay server-side.
 */
'use strict';

const TARGET_SR = 16000;
// How many 16kHz samples to accumulate before sending one PCM frame.
// 1600 = exactly 100ms at 16kHz — matches Sarvam's recommended chunk size.
const SEND_SAMPLES = 1600;

const el = (id) => document.getElementById(id);

const ui = {
    healthPill: el('health-pill'),
    micBtn: el('mic-btn'),
    micLabel: el('mic-label'),
    micWarning: el('mic-warning'),
    recState: el('rec-state'),
    levelBar: el('level-bar'),
    langSelect: el('lang-select'),
    textForm: el('text-form'),
    textInput: el('text-input'),
    sendBtn: el('send-btn'),
    transcriptCard: el('transcript-card'),
    transcript: el('transcript'),
    langBadge: el('lang-badge'),
    answerCard: el('answer-card'),
    answer: el('answer'),
    groundedBadge: el('grounded-badge'),
    latencyBadge: el('latency-badge'),
    abstainNote: el('abstain-note'),
    abstainReason: el('abstain-reason'),
    sourcesBlock: el('sources-block'),
    sources: el('sources'),
    sourcesCount: el('sources-count'),
    feedback: el('feedback'),
    feedbackAck: el('feedback-ack'),
    errorBox: el('error-box'),
    drawer: el('debug-drawer'),
    debugToggle: el('debug-toggle'),
    debugClose: el('debug-close'),
    footInfo: el('foot-info'),
};

let sttAvailable = false;
let currentTraceId = null;
let recording = false;
let audioCtx = null,
    mediaStream = null,
    sourceNode = null,
    procNode = null,
    socket = null;

/* ── helpers ─────────────────────────────────────────────── */
const ABSTAIN_TEXT = {
    input_blocked: 'The question was blocked by the input guardrail.',
    no_candidates: 'Retrieval found no candidate passages in the corpus.',
    low_confidence: 'Retrieved passages scored below the calibrated relevance threshold.',
    weak_margin: 'Top passages were not separable — retrieval is ambiguous.',
    invalid_citation: 'The generated answer cited a source that does not exist, so it was rejected.',
    not_grounded: 'The generated answer was not supported by the retrieved evidence.',
    generation_unavailable: 'The generation model is unavailable (check GROQ_API_KEY). Failing closed instead of guessing.',
    generation_malformed: 'The model returned unparseable output; failed closed.',
    model_refused: 'The model judged the evidence insufficient.',
    retrieval_error: 'The vector database could not be reached.',
    internal_error: 'An internal error occurred.',
};

function fmt(v, digits = 1) {
    if (v === null || v === undefined || Number.isNaN(v)) return 'n/a';
    return typeof v === 'number' ? v.toFixed(digits) : String(v);
}

function setText(node, value) {
    node.textContent = value ?? '—';
}

function showError(message) {
    ui.errorBox.textContent = message;
    ui.errorBox.hidden = false;
}

function clearError() {
    ui.errorBox.hidden = true;
}

function setBusy(busy) {
    ui.sendBtn.disabled = busy;
    ui.sendBtn.textContent = busy ? 'Working…' : 'Ask';
}

/* ── health ──────────────────────────────────────────────── */
async function loadHealth() {
    try {
        const res = await fetch('/health');
        const h = await res.json();
        sttAvailable = !(h.missing_secrets || []).includes('SARVAM_API_KEY') &&
            (h.components || []).some((c) => c.name === 'sarvam' && c.ok);

        const cls = h.status === 'ok' ? 'pill-ok' : h.status === 'degraded' ? 'pill-warn' : 'pill-err';
        ui.healthPill.className = `pill ${cls}`;
        ui.healthPill.textContent = `${h.status} · ${h.vectors ?? 0} vectors · ${h.device}`;

        ui.footInfo.textContent =
            `corpus=${h.corpus_mode} · collection=${h.collection} · languages=${(h.languages || []).join(',')}` +
            ` · device=${h.device} · thresholds ${h.thresholds_calibrated ? 'calibrated' : 'UNCALIBRATED'}`;

        const notes = [];
        if (!sttAvailable) {
            notes.push('Microphone transcription needs <code>SARVAM_API_KEY</code>. Typing works normally.');
            ui.micBtn.disabled = true;
            ui.micLabel.textContent = 'Mic unavailable';
            ui.recState.textContent = 'STT not configured';
        }
        const groq = (h.components || []).find((c) => c.name === 'groq');
        if (groq && !groq.ok) {
            notes.push('Generation is not configured, so the pipeline will <strong>abstain</strong> rather than invent an answer.');
        }
        if (!h.thresholds_calibrated) {
            notes.push('Abstention thresholds are uncalibrated — run <code>scripts/calibrate_thresholds.py</code>.');
        }
        if (notes.length) {
            ui.micWarning.innerHTML = notes.join('<br>');
            ui.micWarning.hidden = false;
        }
    } catch (err) {
        ui.healthPill.className = 'pill pill-err';
        ui.healthPill.textContent = 'backend unreachable';
    }
}

/* ── rendering ───────────────────────────────────────────── */
function renderTranscript(text, isPartial) {
    if (!text) return;
    ui.transcriptCard.hidden = false;
    ui.transcript.textContent = text;
    ui.transcript.classList.toggle('partial', !!isPartial);
}

function renderLanguage(code) {
    if (!code) {
        ui.langBadge.hidden = true;
        return;
    }
    const names = {
        hi: 'Hindi',
        mr: 'Marathi',
        ta: 'Tamil',
        te: 'Telugu',
        bn: 'Bengali',
        gu: 'Gujarati',
        kn: 'Kannada',
        ml: 'Malayalam',
        pa: 'Punjabi',
        or: 'Odia',
        as: 'Assamese',
        ur: 'Urdu',
        ne: 'Nepali',
        sa: 'Sanskrit',
    };
    ui.langBadge.hidden = false;
    ui.langBadge.textContent = `detected: ${names[code] || code} (${code})`;
}

function renderAnswer(data) {
    currentTraceId = data.trace_id;
    ui.answerCard.hidden = false;
    ui.answer.textContent = data.answer || '';

    if (data.transcript) renderTranscript(data.transcript, false);
    renderLanguage(data.detected_language || data.language);

    if (data.abstained) {
        ui.abstainNote.hidden = false;
        ui.abstainReason.textContent =
            ' ' + (ABSTAIN_TEXT[data.abstain_reason] || data.abstain_reason || '');
        ui.groundedBadge.hidden = false;
        ui.groundedBadge.className = 'pill pill-abstain';
        ui.groundedBadge.textContent = 'abstained';
    } else {
        ui.abstainNote.hidden = true;
        ui.groundedBadge.hidden = false;
        ui.groundedBadge.className = data.grounded ? 'pill pill-ok' : 'pill pill-warn';
        ui.groundedBadge.textContent = data.grounded ? 'grounded' : 'not verified';
    }

    const total = data.latency_ms ? data.latency_ms.total : null;
    ui.latencyBadge.hidden = false;
    ui.latencyBadge.textContent = total === null || total === undefined ?
        'latency n/a' :
        `${Math.round(total)} ms (RAG)`;

    // sources
    const citations = data.citations || [];
    ui.sources.innerHTML = '';
    ui.sourcesBlock.hidden = citations.length === 0;
    ui.sourcesCount.textContent = citations.length ? `(${citations.length})` : '';
    citations.forEach((c) => {
        const li = document.createElement('li');
        const meta = document.createElement('div');
        meta.className = 'src-meta';
        const id = document.createElement('span');
        id.className = 'src-id';
        id.textContent = c.chunk_id;
        const score = document.createElement('span');
        score.className = 'pill pill-muted';
        score.textContent = `score ${fmt(c.score, 3)}`;
        meta.append(id, score);
        if (c.language) {
            const lang = document.createElement('span');
            lang.className = 'pill pill-muted';
            lang.textContent = c.language;
            meta.append(lang);
        }
        const body = document.createElement('div');
        body.className = 'src-text';
        body.textContent = c.text || '';
        li.append(meta, body);
        ui.sources.append(li);
    });

    ui.feedback.hidden = false;
    ui.feedbackAck.textContent = '';
    renderDebug(data);
}

function renderDebug(data) {
    const d = data.debug;
    const lat = data.latency_detail || (d && d.latency) || {};

    setText(el('d-trace'), data.trace_id);
    setText(el('d-lang'), d ? d.detected_language : data.detected_language);
    setText(el('d-conf'), d ? fmt(d.language_confidence, 2) : 'n/a');
    setText(el('d-mixed'), d ? String(d.is_code_mixed) : '—');
    setText(el('d-mode'), d ? d.retrieval_mode : '—');
    setText(el('d-langs'), d && d.languages_searched ? d.languages_searched.join(', ') : '—');
    setText(el('d-nq'), d ? d.normalized_query : '—');
    setText(el('d-top'), d ? fmt(d.gate_top_score, 3) : '—');
    setText(el('d-margin'), d ? fmt(d.gate_margin, 3) : '—');
    setText(el('d-thresh'), d ? `${fmt(d.gate_threshold, 3)} ${d.thresholds_calibrated ? '' : '(UNCALIBRATED)'}` : '—');
    setText(el('d-ground'), d ? (d.grounding_status || '—') : '—');
    setText(el('d-genmodel'), d ? d.generation_model : '—');
    setText(el('d-corpus'), d ? d.corpus_mode : '—');

    const warnings = (d && d.warnings) || [];
    el('d-warnbox').hidden = warnings.length === 0;
    el('d-warnings').innerHTML = '';
    warnings.forEach((w) => {
        const li = document.createElement('li');
        li.textContent = w;
        el('d-warnings').append(li);
    });

    const stages = [
        ['STT', 'stt_latency'],
        ['Guardrail', 'guardrail_latency'],
        ['Embedding', 'query_embedding_latency'],
        ['Dense', 'dense_latency'],
        ['Sparse', 'sparse_latency'],
        ['RRF', 'rrf_latency'],
        ['Rerank', 'rerank_latency'],
        ['Gate', 'grounding_gate_latency'],
        ['Gen TTFT', 'generation_ttft'],
        ['Gen E2E', 'generation_e2e'],
        ['NLI', 'nli_latency'],
        ['Output guard', 'output_guardrail_latency'],
        ['TOTAL RAG', 'total_rag_latency'],
        ['TOTAL voice', 'total_voice_latency'],
        ['TOTAL completion', 'total_completion_latency'],
    ];
    const tbody = el('d-timings');
    tbody.innerHTML = '';
    stages.forEach(([label, key]) => {
        const tr = document.createElement('tr');
        const th = document.createElement('th');
        th.textContent = label;
        const td = document.createElement('td');
        td.className = 'num';
        td.textContent = fmt(lat[key], 2);
        tr.append(th, td);
        tbody.append(tr);
    });

    const path = el('d-path');
    path.innerHTML = '';
    ((d && d.stage_path) || []).forEach((stage) => {
        const span = document.createElement('span');
        span.textContent = stage;
        if (stage === 'ABSTAIN') span.className = 'abstain';
        else if (stage === 'DONE') span.className = 'done';
        else if (stage === 'ERROR') span.className = 'error';
        path.append(span);
    });

    const cands = el('d-cands');
    cands.innerHTML = '';
    ((d && d.candidates) || []).forEach((c, i) => {
        const tr = document.createElement('tr');
        if (i === 0) tr.className = 'top-hit';
        [
            String(i + 1), c.chunk_id, c.strategy,
            c.dense_rank ?? '—', c.sparse_rank ?? '—',
            fmt(c.fused_score, 4), fmt(c.rerank_score, 3),
        ].forEach((value, idx) => {
            const td = document.createElement('td');
            if (idx >= 3) td.className = 'num';
            td.textContent = value;
            td.title = idx === 1 ? (c.text_preview || '') : '';
            tr.append(td);
        });
        cands.append(tr);
    });

    const selected = el('d-selected');
    selected.innerHTML = '';
    ((d && d.selected_chunk_ids) || []).forEach((id) => {
        const li = document.createElement('li');
        li.textContent = id;
        selected.append(li);
    });
}

/* ── text query ──────────────────────────────────────────── */
async function askText(query, language) {
    clearError();
    setBusy(true);
    ui.recState.textContent = 'Retrieving…';
    ui.answerCard.hidden = true;
    renderTranscript(query, false);
    renderLanguage(language || null);
    try {
        const res = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query,
                language: language || null,
                include_debug: true
            }),
        });
        if (!res.ok) {
            const body = await res.json().catch(() => ({}));
            throw new Error(body.detail || `HTTP ${res.status}`);
        }
        renderAnswer(await res.json());
    } catch (err) {
        showError(`Request failed: ${err.message}`);
    } finally {
        setBusy(false);
        ui.recState.textContent = 'Idle';
    }
}

/* ── audio helpers ───────────────────────────────────────── */

/**
 * Box-average downsample from inputRate → TARGET_SR, return Int16Array.
 * Box averaging is cheap anti-aliasing — avoids aliasing high frequencies
 * into the 0–8kHz voice band that nearest-neighbour picking would cause.
 */
function downsampleToPCM16(input, inputRate) {
    const ratio = inputRate / TARGET_SR;
    const outLength = Math.floor(input.length / ratio);
    const out = new Int16Array(outLength);
    for (let i = 0; i < outLength; i++) {
        const start = Math.floor(i * ratio);
        const end = Math.min(Math.floor((i + 1) * ratio), input.length);
        let sum = 0, count = 0;
        for (let j = start; j < end; j++) { sum += input[j]; count++; }
        const s = count ? sum / count : (input[start] || 0);
        const c = Math.max(-1, Math.min(1, s));
        out[i] = c < 0 ? c * 0x8000 : c * 0x7fff;
    }
    return out;
}

/**
 * Worklet processor source — inlined as a Blob URL so we don't need a
 * separate file on the server. Accumulates samples until SEND_SAMPLES are
 * ready, then posts them as a Float32Array to the main thread.
 * Using AudioWorklet avoids the ScriptProcessor deprecation and runs off the
 * audio thread (no glitches from main-thread GC pauses).
 */
const WORKLET_SRC = `
class PcmSender extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = [];
    this._count = 0;
    this._limit = ${SEND_SAMPLES};
  }
  process(inputs) {
    const ch = inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this._buf.push(ch[i]);
      this._count++;
      if (this._count >= this._limit) {
        this.port.postMessage(new Float32Array(this._buf));
        this._buf = [];
        this._count = 0;
      }
    }
    return true;
  }
}
registerProcessor('pcm-sender', PcmSender);
`;

let workletUrl = null;
function getWorkletUrl() {
    if (!workletUrl) {
        workletUrl = URL.createObjectURL(new Blob([WORKLET_SRC], { type: 'application/javascript' }));
    }
    return workletUrl;
}

/* ── microphone streaming ────────────────────────────────── */

async function startRecording() {
    if (recording || !sttAvailable) return;
    clearError();

    // 1. Get mic permission first — before opening the WebSocket.
    //    The old code opened the socket and then asked for permission; if the
    //    user took >500ms to accept the browser dialog, the server-side config
    //    wait loop expired and language hints were silently dropped.
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, sampleRate: TARGET_SR },
        });
    } catch {
        showError('Microphone permission denied or unavailable.');
        return;
    }

    recording = true;
    ui.micBtn.classList.add('recording');
    ui.micBtn.setAttribute('aria-label', 'Stop recording');
    ui.micLabel.textContent = 'Listening…';
    ui.recState.textContent = 'Recording';
    ui.answerCard.hidden = true;
    ui.transcript.textContent = '';
    ui.transcriptCard.hidden = false;

    // 2. Open WebSocket now that we have audio ready.
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${proto}://${location.host}/api/voice/stream`);
    socket.binaryType = 'arraybuffer';

    socket.onopen = () => {
        // Pass the selected language so Sarvam skips its auto-detection model.
        // "unknown" triggers detection (~200-400ms extra); a pinned language code
        // (hi-IN, mr-IN, etc.) skips it entirely.
        const langMap = { hi: 'hi-IN', mr: 'mr-IN', ta: 'ta-IN', te: 'te-IN' };
        const selected = ui.langSelect.value;
        const sarvamLang = langMap[selected] || null; // null → server uses "unknown"
        socket.send(JSON.stringify({
            event: 'config',
            language: selected || null,      // ISO-639-1 for the RAG pipeline
            sarvam_language: sarvamLang,     // BCP-47 for Sarvam STT (optional field, server ignores if missing)
            is_wav: false,
            include_debug: true,
        }));
    };

    socket.onmessage = (event) => {
        let msg;
        try { msg = JSON.parse(event.data); } catch { return; }
        if (msg.type === 'partial') {
            renderTranscript(msg.text, true);
            ui.recState.textContent = 'Transcribing…';
        } else if (msg.type === 'transcript') {
            renderTranscript(msg.text, false);
            renderLanguage(msg.language);
            ui.recState.textContent = 'Retrieving…';
        } else if (msg.type === 'answer') {
            renderAnswer(msg);
            setBusy(false);
            ui.recState.textContent = 'Idle';
        } else if (msg.type === 'error') {
            // stt_auth errors come through here — give a clearer message
            const detail = msg.detail || 'Voice pipeline failed.';
            const friendly = msg.code === 'stt_auth'
                ? 'STT credentials missing or rejected. Check SARVAM_API_KEY.'
                : detail;
            showError(friendly);
            setBusy(false);
            ui.recState.textContent = 'Idle';
        }
    };

    socket.onerror = () => {
        if (recording) showError('WebSocket error — check the server is running.');
    };

    // 3. Set up the audio graph. Try AudioWorklet first (modern, non-deprecated).
    //    Fall back to ScriptProcessor if the browser doesn't support it.
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    // Resume in case autoplay policy suspended it
    if (audioCtx.state === 'suspended') await audioCtx.resume();
    sourceNode = audioCtx.createMediaStreamSource(mediaStream);

    const nativeRate = audioCtx.sampleRate;

    try {
        await audioCtx.audioWorklet.addModule(getWorkletUrl());
        const workletNode = new AudioWorkletNode(audioCtx, 'pcm-sender');
        workletNode.port.onmessage = (e) => {
            if (!recording) return;
            const f32 = e.data;
            // level meter + speaking indicator
            let peak = 0;
            for (let i = 0; i < f32.length; i++) peak = Math.max(peak, Math.abs(f32[i]));
            ui.levelBar.style.width = `${Math.min(100, peak * 180)}%`;
            if (peak > 0.01) ui.recState.textContent = 'Speaking…';
            // send PCM
            if (socket && socket.readyState === WebSocket.OPEN) {
                const pcm = downsampleToPCM16(f32, nativeRate);
                socket.send(pcm.buffer);
            }
        };
        sourceNode.connect(workletNode);
        // Store on procNode slot so stopRecording can disconnect it
        procNode = workletNode;
    } catch {
        // Worklet unavailable — fall back to deprecated ScriptProcessor.
        // Buffer size 4096 at ~48kHz = ~85ms per callback, close enough to
        // the 100ms target without the 2048-sample (42ms) choppiness.
        procNode = audioCtx.createScriptProcessor(4096, 1, 1);
        procNode.onaudioprocess = (ev) => {
            if (!recording) return;
            const input = ev.inputBuffer.getChannelData(0);
            let peak = 0;
            for (let i = 0; i < input.length; i++) peak = Math.max(peak, Math.abs(input[i]));
            ui.levelBar.style.width = `${Math.min(100, peak * 180)}%`;
            if (peak > 0.01) ui.recState.textContent = 'Speaking…';
            if (socket && socket.readyState === WebSocket.OPEN) {
                const pcm = downsampleToPCM16(input, nativeRate);
                socket.send(pcm.buffer);
            }
        };
        sourceNode.connect(procNode);
        procNode.connect(audioCtx.destination);
    }
}

function stopRecording() {
    if (!recording) return;
    recording = false;
    ui.micBtn.classList.remove('recording');
    ui.micBtn.setAttribute('aria-label', 'Start recording');
    ui.micLabel.textContent = 'Hold to speak';
    ui.recState.textContent = 'Transcribing…';
    ui.levelBar.style.width = '0%';
    setBusy(true);

    try {
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ event: 'end' }));
        }
    } catch {}
    try { if (procNode) { procNode.disconnect(); procNode.onaudioprocess = null; } } catch {}
    try { if (sourceNode) sourceNode.disconnect(); } catch {}
    try { if (audioCtx) audioCtx.close(); } catch {}
    try { if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop()); } catch {}
    audioCtx = sourceNode = procNode = mediaStream = null;
}

/* ── wiring ──────────────────────────────────────────────── */
ui.textForm.addEventListener('submit', (event) => {
    event.preventDefault();
    const query = ui.textInput.value.trim();
    if (query) askText(query, ui.langSelect.value);
});

document.querySelectorAll('.chip').forEach((chip) => {
    chip.addEventListener('click', () => {
        const q = chip.textContent.trim();
        const lang = chip.dataset.lang || '';
        ui.textInput.value = q;
        ui.langSelect.value = lang;
        askText(q, lang);
    });
});

// Press-and-hold (pointer) plus click-to-toggle for keyboard/assistive use.
ui.micBtn.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    startRecording();
});
ui.micBtn.addEventListener('pointerup', stopRecording);
ui.micBtn.addEventListener('pointerleave', stopRecording);
ui.micBtn.addEventListener('keydown', (event) => {
    if ((event.key === ' ' || event.key === 'Enter') && !recording) {
        event.preventDefault();
        startRecording();
    }
});
ui.micBtn.addEventListener('keyup', (event) => {
    if (event.key === ' ' || event.key === 'Enter') {
        event.preventDefault();
        stopRecording();
    }
});

function toggleDrawer(open) {
    ui.drawer.hidden = !open;
    ui.debugToggle.setAttribute('aria-expanded', String(open));
    if (open) ui.debugClose.focus();
    else ui.debugToggle.focus();
}
ui.debugToggle.addEventListener('click', () => toggleDrawer(ui.drawer.hidden));
ui.debugClose.addEventListener('click', () => toggleDrawer(false));
document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !ui.drawer.hidden) toggleDrawer(false);
});

ui.feedback.addEventListener('click', async (event) => {
    const button = event.target.closest('button[data-rating]');
    if (!button || !currentTraceId) return;
    try {
        await fetch('/api/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                trace_id: currentTraceId,
                rating: button.dataset.rating
            }),
        });
        ui.feedbackAck.textContent = 'thanks';
    } catch {
        ui.feedbackAck.textContent = 'could not save';
    }
});

loadHealth();