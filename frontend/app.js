/* Voice RAG frontend.
 *
 * Audio: Sarvam's streaming API accepts WAV or raw PCM only - not the WebM/Opus
 * that MediaRecorder produces by default. So we capture via the Web Audio API,
 * downsample to 16 kHz and send 16-bit little-endian PCM frames. That avoids a
 * server-side transcode on the latency-critical path.
 *
 * No API keys exist in this file. All credentials stay server-side; the browser
 * only ever talks to this app's own origin.
 */
'use strict';

const TARGET_SR = 16000;
const FRAME_SAMPLES = 2048;

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
    node.textContent = value ? ? '—';
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
    if (busy) ui.recState.textContent = 'Running pipeline…';
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
        en: 'English',
        bn: 'Bengali',
        gu: 'Gujarati',
        kn: 'Kannada',
        ml: 'Malayalam',
        pa: 'Punjabi',
        or: 'Odia',
        as: 'Assamese',
        ur: 'Urdu',
        ne: 'Nepali',
        sa: 'Sanskrit'
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
            c.dense_rank ? ? '—', c.sparse_rank ? ? '—',
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

/* ── microphone streaming ────────────────────────────────── */
function downsampleToPCM16(input, inputRate) {
    const ratio = inputRate / TARGET_SR;
    const outLength = Math.floor(input.length / ratio);
    const out = new Int16Array(outLength);
    for (let i = 0; i < outLength; i++) {
        // Box-average the source window: cheap anti-aliasing, better than
        // nearest-neighbour picking which aliases high frequencies into the voice band.
        const start = Math.floor(i * ratio);
        const end = Math.min(Math.floor((i + 1) * ratio), input.length);
        let sum = 0,
            count = 0;
        for (let j = start; j < end; j++) {
            sum += input[j];
            count++;
        }
        const sample = count ? sum / count : input[start] || 0;
        const clamped = Math.max(-1, Math.min(1, sample));
        out[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    }
    return out;
}

async function startRecording() {
    if (recording || !sttAvailable) return;
    clearError();
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            },
        });
    } catch (err) {
        showError('Microphone permission denied or unavailable.');
        return;
    }

    recording = true;
    ui.micBtn.classList.add('recording');
    ui.micBtn.setAttribute('aria-label', 'Stop recording');
    ui.micLabel.textContent = 'Listening…';
    ui.recState.textContent = 'Recording — release to send';
    ui.answerCard.hidden = true;
    ui.transcript.textContent = '';

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${proto}://${location.host}/api/voice/stream`);
    socket.binaryType = 'arraybuffer';

    socket.onopen = () => {
        socket.send(JSON.stringify({
            event: 'config',
            language: ui.langSelect.value || null,
            is_wav: false,
            include_debug: true,
        }));
    };

    socket.onmessage = (event) => {
        let msg;
        try {
            msg = JSON.parse(event.data);
        } catch {
            return;
        }
        if (msg.type === 'partial') {
            renderTranscript(msg.text, true);
            ui.recState.textContent = 'Transcribing…';
        } else if (msg.type === 'transcript') {
            renderTranscript(msg.text, false);
            renderLanguage(msg.language);
        } else if (msg.type === 'answer') {
            renderAnswer(msg);
            setBusy(false);
            ui.recState.textContent = 'Idle';
        } else if (msg.type === 'error') {
            showError(msg.detail || 'Voice pipeline failed.');
            setBusy(false);
            ui.recState.textContent = 'Idle';
        }
    };
    socket.onerror = () => showError('Voice WebSocket error.');

    audioCtx = new(window.AudioContext || window.webkitAudioContext)();
    sourceNode = audioCtx.createMediaStreamSource(mediaStream);
    procNode = audioCtx.createScriptProcessor(FRAME_SAMPLES, 1, 1);

    procNode.onaudioprocess = (event) => {
        if (!recording) return;
        const input = event.inputBuffer.getChannelData(0);

        let peak = 0;
        for (let i = 0; i < input.length; i++) peak = Math.max(peak, Math.abs(input[i]));
        ui.levelBar.style.width = `${Math.min(100, peak * 180)}%`;

        if (socket && socket.readyState === WebSocket.OPEN) {
            const pcm = downsampleToPCM16(input, audioCtx.sampleRate);
            socket.send(pcm.buffer);
        }
    };

    sourceNode.connect(procNode);
    procNode.connect(audioCtx.destination);
}

function stopRecording() {
    if (!recording) return;
    recording = false;
    ui.micBtn.classList.remove('recording');
    ui.micBtn.setAttribute('aria-label', 'Start recording');
    ui.micLabel.textContent = 'Hold to speak';
    ui.recState.textContent = 'Processing…';
    ui.levelBar.style.width = '0%';
    setBusy(true);

    try {
        if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({
            event: 'end'
        }));
    } catch {}
    try {
        if (procNode) {
            procNode.disconnect();
            procNode.onaudioprocess = null;
        }
    } catch {}
    try {
        if (sourceNode) sourceNode.disconnect();
    } catch {}
    try {
        if (audioCtx) audioCtx.close();
    } catch {}
    try {
        if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
    } catch {}
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
        ui.textInput.value = chip.textContent;
        ui.langSelect.value = chip.dataset.lang || '';
        askText(chip.textContent, chip.dataset.lang || '');
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