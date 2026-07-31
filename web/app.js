/* Model ablation study UI.
   No build step and no CDN: the Orin may well be on an isolated bench
   network, so everything here is plain DOM plus hand-drawn SVG charts. */

const $ = (id) => document.getElementById(id);
const SVG_NS = 'http://www.w3.org/2000/svg';

const PAIR_COLOR = { a: '#2dd1c4', b: '#f7aa3c' };

/* Stage colours are shared by both pairings so the stacked bars compare
   like for like; the pairing itself is identified by the row's dot. */
const STAGE_META = [
  { key: 'preprocess',  label: 'Preprocess',   color: '#3f4a57' },
  { key: 'detect',      label: 'Detect',       color: '#5b7cfa' },
  { key: 'seg_encode',  label: 'Seg encode',   color: '#c084fc' },
  { key: 'seg_decode',  label: 'Seg decode',   color: '#f472b6' },
  { key: 'postprocess', label: 'Postprocess',  color: '#64748b' },
];

const STEPS = [
  { key: 'upload',          name: 'Upload' },
  { key: 'validating',      name: 'Validate' },
  { key: 'running_a',       name: 'NanoOWL + NanoSAM' },
  { key: 'running_b',       name: 'NanoOWL + EfficientViT-SAM' },
  { key: 'aggregating',     name: 'Aggregate' },
  { key: 'rendering_video', name: 'Comparison video' },
];
const STATE_INDEX = {
  queued: 0, validating: 1, running_a: 2, running_b: 3,
  aggregating: 4, rendering_video: 5,
};

const state = {
  file: null,
  jobId: null,
  ws: null,
  reconnectDelay: 500,
  spark: [],
  uploading: false,
  serverConfig: null,
};

/* ── helpers ─────────────────────────────────────────────────────────── */
const fmt = (v, d = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d);

function fmtBytes(bytes) {
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0, n = bytes;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  children.flat().forEach((c) => c && node.appendChild(c));
  return node;
}

function svgEl(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  children.flat().forEach((c) => c && node.appendChild(c));
  return node;
}

function svgText(x, y, str, attrs = {}) {
  const node = svgEl('text', {
    x, y, fill: '#9aa4b0', 'font-size': 11,
    'font-family': 'ui-monospace, Menlo, monospace', ...attrs,
  });
  node.textContent = str;
  return node;
}

/* Chooses a round tick step so axes never show 37.4183. */
function niceTicks(max, count = 5) {
  if (!(max > 0)) return [0, 1];
  const raw = max / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].find((m) => raw <= m * mag) * mag;
  const ticks = [];
  for (let t = 0; t <= max + step * 0.5; t += step) ticks.push(t);
  return ticks;
}

/* ── setup / upload ──────────────────────────────────────────────────── */
function initUpload() {
  const dz = $('dropzone');
  const input = $('file-input');

  const pick = () => input.click();
  dz.addEventListener('click', (e) => { if (!e.target.closest('.chip-x')) pick(); });
  dz.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); }
  });
  input.addEventListener('change', () => setFile(input.files[0]));

  ['dragenter', 'dragover'].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('dragover'); }));
  ['dragleave', 'drop'].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('dragover'); }));
  dz.addEventListener('drop', (e) => setFile(e.dataTransfer.files[0]));

  $('file-clear').addEventListener('click', (e) => { e.stopPropagation(); setFile(null); });
  $('start-btn').addEventListener('click', startBenchmark);
  $('cancel-btn').addEventListener('click', cancelJob);
}

function setFile(file) {
  state.file = file || null;
  const chip = $('file-chip');
  const dz = $('dropzone');
  if (file) {
    $('file-name').textContent = file.name;
    $('file-size').textContent = fmtBytes(file.size);
    chip.classList.remove('hidden');
    dz.classList.add('has-file');
  } else {
    chip.classList.add('hidden');
    dz.classList.remove('has-file');
    $('file-input').value = '';
  }
  $('start-btn').disabled = !file || state.uploading;
  $('setup-error').textContent = '';
}

function startBenchmark() {
  if (!state.file || state.uploading) return;
  const prompts = $('prompts').value.trim();
  if (!prompts) {
    $('setup-error').textContent = 'NanoOWL needs at least one text prompt.';
    return;
  }

  const form = new FormData();
  form.append('video', state.file);
  form.append('prompts', prompts);
  form.append('threshold', $('threshold').value || '0.1');
  form.append('stride', $('stride').value || '1');
  form.append('max_frames', $('max-frames').value || '');
  form.append('warmup_frames', $('warmup').value || '0');
  form.append('record_masks', $('record-masks').checked ? 'true' : 'false');

  state.uploading = true;
  $('start-btn').disabled = true;
  $('start-btn').textContent = 'Uploading…';
  $('setup-error').textContent = '';
  $('results').classList.add('hidden');
  $('progress').classList.remove('hidden');
  showUploadStep(0);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/jobs');
  xhr.upload.addEventListener('progress', (e) => {
    if (e.lengthComputable) showUploadStep((e.loaded / e.total) * 100);
  });
  xhr.addEventListener('load', () => {
    state.uploading = false;
    $('start-btn').textContent = 'Run benchmark';
    $('start-btn').disabled = false;
    if (xhr.status >= 200 && xhr.status < 300) {
      const job = JSON.parse(xhr.responseText);
      attachToJob(job.job_id);
      renderJob(job);
      $('progress').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
      let msg = `Upload failed (${xhr.status}).`;
      try { msg = JSON.parse(xhr.responseText).detail || msg; } catch (_) {}
      $('setup-error').textContent = msg;
      $('progress').classList.add('hidden');
    }
  });
  xhr.addEventListener('error', () => {
    state.uploading = false;
    $('start-btn').textContent = 'Run benchmark';
    $('start-btn').disabled = false;
    $('setup-error').textContent = 'Network error during upload.';
  });
  xhr.send(form);
}

function showUploadStep(pct) {
  renderStepper({ state: pct >= 100 ? 'validating' : 'uploading' }, pct);
}

async function cancelJob() {
  if (!state.jobId) return;
  await fetch(`/api/jobs/${state.jobId}/cancel`, { method: 'POST' }).catch(() => {});
}

/* ── job tracking ────────────────────────────────────────────────────── */
function attachToJob(jobId) {
  state.jobId = jobId;
  state.spark = [];
  $('job-id-chip').textContent = jobId;
  connectSocket();
}

function connectSocket() {
  if (state.ws) { try { state.ws.close(); } catch (_) {} }
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${location.host}/ws/jobs/${state.jobId}`);
  state.ws = ws;

  ws.addEventListener('open', () => { state.reconnectDelay = 500; });
  ws.addEventListener('message', (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'job') renderJob(msg.job);
    else if (msg.type === 'progress') renderProgress(msg.progress);
  });
  ws.addEventListener('close', () => {
    // Keep trying: a benchmark can outlive a flaky bench-network link.
    if (state.jobId) {
      setTimeout(connectSocket, state.reconnectDelay);
      state.reconnectDelay = Math.min(state.reconnectDelay * 2, 8000);
    }
  });
}

function renderJob(job) {
  renderStepper(job);

  $('cancel-btn').classList.toggle('hidden', !!job.is_terminal);

  const queueNote = $('queue-note');
  if (job.state === 'queued' && job.queue_position) {
    queueNote.textContent =
      `Queued at position ${job.queue_position} — only one benchmark runs at a time so the ` +
      `latency numbers stay clean.`;
    queueNote.classList.remove('hidden');
  } else {
    queueNote.classList.add('hidden');
  }

  const errNote = $('job-error');
  if (job.error && (job.state === 'failed' || job.state === 'cancelled')) {
    errNote.textContent = job.error;
    errNote.classList.remove('hidden');
  } else {
    errNote.classList.add('hidden');
  }

  const running = ['running_a', 'running_b', 'rendering_video'].includes(job.state);
  $('live').classList.toggle('hidden', !running);

  if (job.has_results) loadResults(job);
  if (job.is_terminal) {
    renderVideoSlot(job);
    loadHistory();
  }
}

const STEP_MARK = { done: '✓', active: '•', failed: '!', skipped: '–' };
const STEP_CLASS = { done: 'is-done', active: 'is-active', failed: 'is-failed', skipped: 'is-skipped' };

/* Resolves one step to pending | active | done | failed | skipped. */
function stepStatus(step, i, job) {
  if (job.state === 'uploading') return i === 0 ? 'active' : 'pending';

  // Upload is settled the moment the server has a job for us.
  if (i === 0) return 'done';

  if (job.state === 'complete') return 'done';
  if (job.state === 'complete_no_video') {
    return step.key === 'rendering_video' ? 'skipped' : 'done';
  }
  if (job.state === 'failed' || job.state === 'cancelled') {
    // last_stage is where the server actually stopped.
    const stoppedAt = STATE_INDEX[job.last_stage] ?? 1;
    if (i < stoppedAt) return 'done';
    if (i === stoppedAt) return job.state === 'failed' ? 'failed' : 'skipped';
    return 'pending';
  }

  const current = STATE_INDEX[job.state];
  if (current === undefined) return 'pending';
  if (i < current) return 'done';
  if (i === current) return 'active';
  return 'pending';
}

function stepSubtitle(step, status, job, uploadPct) {
  if (step.key === 'upload' && status === 'active') return `${Math.round(uploadPct || 0)}%`;
  if (status === 'failed') return job.state === 'failed' ? 'failed' : 'stopped';
  if (step.key === 'rendering_video') {
    if (status === 'skipped') return job.video_error ? 'render failed' : 'skipped';
    if (status === 'active') return 'composing…';
  }
  if ((step.key === 'running_a' || step.key === 'running_b') && status === 'active') {
    return pairingSub(job);
  }
  return '';
}

function renderStepper(job, uploadPct) {
  const stepper = $('stepper');
  stepper.innerHTML = '';

  STEPS.forEach((step, i) => {
    const status = stepStatus(step, i, job);
    const sub = stepSubtitle(step, status, job, uploadPct);
    const cls = STEP_CLASS[status] ? `step ${STEP_CLASS[status]}` : 'step';

    stepper.appendChild(
      el('li', { class: cls },
        el('span', { class: 'step-dot', text: STEP_MARK[status] || String(i + 1) }),
        el('span', { class: 'step-text' },
          el('span', { class: 'step-name', text: step.name }),
          sub ? el('span', { class: 'step-sub', text: sub }) : null)));
  });
}

function pairingSub(job) {
  const p = job.progress;
  if (!p || !p.total) return 'running…';
  return `${p.processed}/${p.total} frames`;
}

function renderProgress(p) {
  if (!p) return;
  const isVideo = p.pairing_id === 'video';
  $('live-label').textContent = isVideo
    ? 'Rendering comparison video'
    : (p.pairing_id === 'a' ? 'NanoOWL + NanoSAM' : 'NanoOWL + EfficientViT-SAM')
      + (p.warmup ? '  ·  warm-up (excluded from stats)' : '');
  $('live-counter').textContent = p.total ? `${p.processed} / ${p.total}` : `${p.processed}`;
  $('live-bar').style.width = `${Math.min(100, p.pct || 0)}%`;

  if (isVideo) {
    $('live-latency').textContent = '—';
    $('live-fps').textContent = '—';
    $('live-dets').textContent = '—';
    return;
  }

  $('live-latency').textContent = `${fmt(p.pipeline_ms, 1)} ms`;
  $('live-fps').textContent = fmt(p.fps_instant, 1);
  $('live-dets').textContent = p.num_detections ?? '—';

  state.spark.push({ ms: p.pipeline_ms, pairing: p.pairing_id, warmup: p.warmup });
  if (state.spark.length > 240) state.spark.shift();
  drawSparkline();

  const step = STEPS.find((s) => s.key === `running_${p.pairing_id}`);
  if (step) {
    const node = $('stepper').children[STEPS.indexOf(step)];
    const sub = node?.querySelector('.step-sub');
    if (sub && p.total) sub.textContent = `${p.processed}/${p.total} frames`;
  }
}

function drawSparkline() {
  const svg = $('sparkline');
  svg.innerHTML = '';
  const data = state.spark;
  if (data.length < 2) return;

  const W = 600, H = 90, pad = 6;
  const max = Math.max(...data.map((d) => d.ms)) * 1.15 || 1;
  const dx = (W - pad * 2) / (data.length - 1);
  const y = (v) => H - pad - (v / max) * (H - pad * 2);

  const pts = data.map((d, i) => `${pad + i * dx},${y(d.ms)}`).join(' ');
  const color = PAIR_COLOR[data[data.length - 1].pairing] || '#2dd1c4';

  svg.appendChild(svgEl('polygon', {
    points: `${pad},${H - pad} ${pts} ${pad + (data.length - 1) * dx},${H - pad}`,
    fill: color, opacity: 0.13,
  }));
  svg.appendChild(svgEl('polyline', {
    points: pts, fill: 'none', stroke: color, 'stroke-width': 1.6,
    'stroke-linejoin': 'round', 'vector-effect': 'non-scaling-stroke',
  }));
}

/* ── results ─────────────────────────────────────────────────────────── */
let lastResultsJob = null;

async function loadResults(job) {
  if (lastResultsJob === job.job_id && $('results').dataset.rendered === job.job_id) return;
  const res = await fetch(`/api/jobs/${job.job_id}/results`);
  if (!res.ok) return;
  const data = await res.json();
  lastResultsJob = job.job_id;
  $('results').dataset.rendered = job.job_id;
  renderResults(data, job);
}

function renderResults(data, job) {
  const a = data.pairings.a;
  const b = data.pairings.b;
  const cmp = data.comparison;

  $('results').classList.remove('hidden');
  $('dl-json').href = `/api/jobs/${data.job_id}/results.json`;
  $('dl-csv').href = `/api/jobs/${data.job_id}/export.csv`;

  renderVerdict(data, a, b, cmp);
  renderPairingCards(a, b, cmp);
  renderRunMeta(data);
  renderStageChart(a, b);
  renderStageTable(a, b, cmp);
  renderSeriesChart(data);
  renderHistogram(data, a, b);
  renderPercentileTable(a, b);
  renderVideoSlot(job);
}

function renderVerdict(data, a, b, cmp) {
  const fasterId = cmp.faster_pairing;
  const faster = fasterId === 'a' ? a : b;
  const slower = fasterId === 'a' ? b : a;
  const pct = Math.abs(cmp.pipeline_p50.pct);

  const mockWarning = data.is_mock
    ? `<div class="note note-warn" style="margin-bottom:14px">
         <strong>Mock backend.</strong> These are synthetic latencies from CPU stand-ins,
         not Jetson measurements. Run on the Orin with the real engines for study data.
       </div>` : '';

  $('verdict').innerHTML = `${mockWarning}
    <strong style="color:${PAIR_COLOR[fasterId]}">${faster.label}</strong> is the faster pairing —
    <span class="num">${fmt(faster.pipeline.p50, 1)} ms</span> median frame latency versus
    <span class="num">${fmt(slower.pipeline.p50, 1)} ms</span>, a
    <span class="num">${fmt(pct, 1)}%</span> difference
    (<span class="num">${fmt(faster.throughput_fps, 1)}</span> vs
    <span class="num">${fmt(slower.throughput_fps, 1)}</span> FPS).
    Detection is shared, so the gap is the segmentation head:
    <span class="num">${fmt(a.stages.seg_encode.mean + a.stages.seg_decode.mean, 1)} ms</span> for NanoSAM
    against <span class="num">${fmt(b.stages.seg_encode.mean + b.stages.seg_decode.mean, 1)} ms</span>
    for EfficientViT-SAM.`;
}

function deltaBadge(pct, lowerIsBetter = true) {
  const flat = Math.abs(pct) < 1;
  const good = lowerIsBetter ? pct < 0 : pct > 0;
  const cls = flat ? 'delta-flat' : good ? 'delta-good' : 'delta-bad';
  const sign = pct > 0 ? '+' : '';
  return el('span', { class: `delta ${cls}`, text: `${sign}${fmt(pct, 1)}%` });
}

function renderPairingCards(a, b, cmp) {
  const wrap = $('pairing-cards');
  wrap.innerHTML = '';

  [['a', a], ['b', b]].forEach(([id, s]) => {
    const head = el('div', { class: 'pcard-head' },
      el('span', { class: 'pcard-title', text: s.label }));
    if (id === 'b') head.appendChild(deltaBadge(cmp.pipeline_p50.pct));

    const stats = [
      ['p50', `${fmt(s.pipeline.p50, 1)} ms`],
      ['p95', `${fmt(s.pipeline.p95, 1)} ms`],
      ['p99', `${fmt(s.pipeline.p99, 1)} ms`],
      ['mean', `${fmt(s.pipeline.mean, 1)} ms`],
      ['jitter σ', `${fmt(s.pipeline.std, 2)} ms`],
      ['max', `${fmt(s.pipeline.max, 1)} ms`],
      ['throughput', `${fmt(s.throughput_fps, 1)} fps`],
      ['frames', `${s.frames_measured}`],
      ['mean det/frame', fmt(s.mean_detections_per_frame, 2)],
    ].map(([k, v]) =>
      el('div', { class: 'pstat' },
        el('span', { class: 'pstat-val', text: v }),
        el('span', { class: 'pstat-key', text: k })));

    wrap.appendChild(
      el('div', { class: `pcard pair-${id}` },
        head,
        el('div', { class: 'pcard-sub', text: s.segmenter }),
        el('div', { class: 'pcard-hero' },
          el('span', { class: 'pcard-hero-val', text: fmt(s.pipeline.p50, 1), style: `color:${PAIR_COLOR[id]}` }),
          el('span', { class: 'pcard-hero-unit', text: 'ms median · pipeline latency' })),
        el('div', { class: 'pcard-grid' }, stats)));
  });
}

function renderRunMeta(data) {
  const wrap = $('run-meta');
  wrap.innerHTML = '';
  const rc = data.run_config;
  const v = data.video;
  const chips = [
    `${v.width}×${v.height} @ ${fmt(v.fps, 1)} fps`,
    `${v.frame_count} frames`,
    `prompts: ${rc.prompts.join(', ')}`,
    `threshold ${rc.threshold}`,
    `stride ${rc.stride}`,
    `${rc.warmup_frames} warm-up excluded`,
    `backend: ${data.backend}`,
  ];
  if (data.lifecycle?.share_detector) chips.push('detector shared across runs');
  if (data.lifecycle?.swap_ms !== undefined) chips.push(`segmenter swap ${fmt(data.lifecycle.swap_ms, 0)} ms`);
  if (data.environment?.gpu) chips.push(data.environment.gpu);
  if (data.environment?.tensorrt) chips.push(`TensorRT ${data.environment.tensorrt}`);
  if (data.environment?.nvpmodel) chips.push(data.environment.nvpmodel);
  chips.forEach((c) => wrap.appendChild(el('span', { class: 'chip', text: c })));

  $('footer-env').textContent = [
    data.environment?.platform, data.environment?.l4t,
    data.environment?.torch ? `torch ${data.environment.torch}` : null,
  ].filter(Boolean).join('  ·  ');
}

function legend(items) {
  return el('div', { class: 'chart-legend' },
    items.map(({ color, label }) =>
      el('span', { class: 'legend-item' },
        el('span', { class: 'legend-swatch', style: `background:${color}` }),
        el('span', { text: label }))));
}

/* Stacked horizontal bars: total height compares, segments show composition. */
function renderStageChart(a, b) {
  const host = $('stage-chart');
  host.innerHTML = '';
  host.appendChild(legend(STAGE_META.map((s) => ({ color: s.color, label: s.label }))));

  const W = 900, rowH = 54, padL = 210, padR = 70, padT = 10, padB = 46;
  const H = padT + rowH * 2 + padB;
  const rows = [['a', a], ['b', b]];
  const totals = rows.map(([, s]) => STAGE_META.reduce((t, m) => t + (s.stages[m.key]?.mean || 0), 0));
  const max = Math.max(...totals) * 1.08 || 1;
  const scale = (v) => (v / max) * (W - padL - padR);

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });

  niceTicks(max, 5).forEach((t) => {
    const x = padL + scale(t);
    svg.appendChild(svgEl('line', {
      x1: x, y1: padT, x2: x, y2: padT + rowH * 2 - 12,
      stroke: '#262d36', 'stroke-width': 1,
    }));
    svg.appendChild(svgText(x, H - padB + 20, `${t.toFixed(t < 10 ? 1 : 0)}`, { 'text-anchor': 'middle' }));
  });
  svg.appendChild(svgText(padL + (W - padL - padR) / 2, H - 8, 'mean milliseconds per frame',
    { 'text-anchor': 'middle', fill: '#6b7683' }));

  rows.forEach(([id, s], r) => {
    const y = padT + r * rowH + 6;
    const barH = 26;

    svg.appendChild(svgEl('circle', { cx: 10, cy: y + barH / 2, r: 5, fill: PAIR_COLOR[id] }));
    svg.appendChild(svgText(24, y + barH / 2 + 4, s.label, {
      fill: '#e6e9ed', 'font-size': 12,
      'font-family': '-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
    }));

    let x = padL;
    STAGE_META.forEach((m) => {
      const value = s.stages[m.key]?.mean || 0;
      const w = scale(value);
      if (w <= 0) return;
      const rect = svgEl('rect', { x, y, width: w, height: barH, fill: m.color });
      rect.appendChild(svgEl('title', {}, document.createTextNode(
        `${m.label}: ${value.toFixed(2)} ms`)));
      svg.appendChild(rect);
      if (w > 40) {
        svg.appendChild(svgText(x + w / 2, y + barH / 2 + 4, value.toFixed(1), {
          'text-anchor': 'middle', fill: '#0d1013', 'font-size': 10.5, 'font-weight': 600,
        }));
      }
      x += w;
    });

    svg.appendChild(svgText(x + 10, y + barH / 2 + 4, `${totals[r].toFixed(1)} ms`, {
      fill: PAIR_COLOR[id], 'font-size': 12, 'font-weight': 600,
    }));
  });

  host.appendChild(svg);
}

function renderStageTable(a, b, cmp) {
  const table = $('stage-table');
  table.innerHTML = '';
  table.appendChild(el('thead', {},
    el('tr', {},
      el('th', { text: 'Stage' }),
      el('th', { text: 'NanoSAM (ms)' }),
      el('th', { text: 'EfficientViT-SAM (ms)' }),
      el('th', { text: 'Δ ms' }),
      el('th', { text: 'Δ %' }))));

  const body = el('tbody');
  STAGE_META.forEach((m) => {
    const d = cmp.stages[m.key] || { a: 0, b: 0, abs: 0, pct: 0 };
    const badge = deltaBadge(d.pct);
    body.appendChild(el('tr', {},
      el('td', { text: m.label }),
      el('td', { text: fmt(d.a, 2) }),
      el('td', { text: fmt(d.b, 2) }),
      el('td', { text: `${d.abs > 0 ? '+' : ''}${fmt(d.abs, 2)}` }),
      el('td', {}, badge)));
  });
  body.appendChild(el('tr', {},
    el('td', { html: '<strong>Pipeline total</strong>' }),
    el('td', { html: `<strong>${fmt(a.pipeline.mean, 2)}</strong>` }),
    el('td', { html: `<strong>${fmt(b.pipeline.mean, 2)}</strong>` }),
    el('td', { html: `<strong>${cmp.pipeline_mean.abs > 0 ? '+' : ''}${fmt(cmp.pipeline_mean.abs, 2)}</strong>` }),
    el('td', {}, deltaBadge(cmp.pipeline_mean.pct))));
  table.appendChild(body);
}

function renderSeriesChart(data) {
  const host = $('series-chart');
  host.innerHTML = '';
  host.appendChild(legend([
    { color: PAIR_COLOR.a, label: 'NanoOWL + NanoSAM' },
    { color: PAIR_COLOR.b, label: 'NanoOWL + EfficientViT-SAM' },
  ]));

  const seriesA = data.series.a || [];
  const seriesB = data.series.b || [];
  if (!seriesA.length && !seriesB.length) return;

  const W = 900, H = 300, padL = 52, padR = 14, padT = 26, padB = 34;
  const all = [...seriesA, ...seriesB];
  const maxY = Math.max(...all.map((d) => d.pipeline_ms)) * 1.1 || 1;
  const maxX = Math.max(seriesA.length, seriesB.length) - 1 || 1;
  const x = (i) => padL + (i / maxX) * (W - padL - padR);
  const y = (v) => H - padB - (v / maxY) * (H - padT - padB);

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });

  niceTicks(maxY, 5).forEach((t) => {
    svg.appendChild(svgEl('line', { x1: padL, y1: y(t), x2: W - padR, y2: y(t), stroke: '#262d36' }));
    svg.appendChild(svgText(padL - 8, y(t) + 4, t.toFixed(t < 10 ? 1 : 0), { 'text-anchor': 'end' }));
  });
  svg.appendChild(svgText(padL, 14, 'ms', { fill: '#6b7683', 'text-anchor': 'end' }));

  // Warm-up band: makes visually obvious which frames are excluded.
  const warmCount = seriesA.filter((d) => d.warmup).length;
  if (warmCount > 0) {
    svg.appendChild(svgEl('rect', {
      x: padL, y: padT, width: x(warmCount - 1) - padL, height: H - padT - padB,
      fill: '#f7aa3c', opacity: 0.07,
    }));
    svg.appendChild(svgText(padL + 6, padT + 14, `warm-up · excluded (${warmCount})`,
      { fill: '#8a7a5f', 'font-size': 10 }));
  }

  [['a', seriesA], ['b', seriesB]].forEach(([id, series]) => {
    if (!series.length) return;
    svg.appendChild(svgEl('polyline', {
      points: series.map((d, i) => `${x(i)},${y(d.pipeline_ms)}`).join(' '),
      fill: 'none', stroke: PAIR_COLOR[id], 'stroke-width': 1.4,
      'stroke-linejoin': 'round', opacity: 0.9,
    }));
  });

  [0, Math.floor(maxX / 2), maxX].forEach((i) =>
    svg.appendChild(svgText(x(i), H - padB + 20, String(i), { 'text-anchor': 'middle' })));
  svg.appendChild(svgText((padL + W - padR) / 2, H - 4, 'processed frame',
    { 'text-anchor': 'middle', fill: '#6b7683' }));

  host.appendChild(svg);
}

function renderHistogram(data, a, b) {
  const host = $('hist-chart');
  host.innerHTML = '';
  host.appendChild(legend([
    { color: PAIR_COLOR.a, label: 'NanoOWL + NanoSAM' },
    { color: PAIR_COLOR.b, label: 'NanoOWL + EfficientViT-SAM' },
  ]));

  const histA = data.histograms.a, histB = data.histograms.b;
  if (!histA.counts.length && !histB.counts.length) return;

  const W = 900, H = 280, padL = 52, padR = 14, padT = 30, padB = 34;
  const minX = Math.min(histA.edges[0] ?? Infinity, histB.edges[0] ?? Infinity);
  const maxX = Math.max(histA.edges.at(-1) ?? 0, histB.edges.at(-1) ?? 0);
  const maxY = Math.max(...histA.counts, ...histB.counts) * 1.1 || 1;
  const x = (v) => padL + ((v - minX) / (maxX - minX || 1)) * (W - padL - padR);
  const y = (v) => H - padB - (v / maxY) * (H - padT - padB);

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });

  niceTicks(maxY, 4).forEach((t) => {
    svg.appendChild(svgEl('line', { x1: padL, y1: y(t), x2: W - padR, y2: y(t), stroke: '#262d36' }));
    svg.appendChild(svgText(padL - 8, y(t) + 4, String(Math.round(t)), { 'text-anchor': 'end' }));
  });
  svg.appendChild(svgText(padL, 14, 'frames', { fill: '#6b7683', 'text-anchor': 'end' }));

  [['a', histA], ['b', histB]].forEach(([id, hist]) => {
    hist.counts.forEach((count, i) => {
      if (!count) return;
      const x0 = x(hist.edges[i]), x1 = x(hist.edges[i + 1]);
      svg.appendChild(svgEl('rect', {
        x: x0, y: y(count), width: Math.max(1, x1 - x0 - 0.5), height: H - padB - y(count),
        fill: PAIR_COLOR[id], opacity: 0.55,
      }));
    });
  });

  // p50/p95 markers give the eye an anchor in the distribution.
  [['a', a], ['b', b]].forEach(([id, s]) => {
    [['p50', s.pipeline.p50], ['p95', s.pipeline.p95]].forEach(([name, value]) => {
      svg.appendChild(svgEl('line', {
        x1: x(value), y1: padT, x2: x(value), y2: H - padB,
        stroke: PAIR_COLOR[id], 'stroke-width': 1.2,
        'stroke-dasharray': name === 'p50' ? '' : '4 3', opacity: 0.85,
      }));
      svg.appendChild(svgText(x(value) + 4, padT + (id === 'a' ? 12 : 26), name,
        { fill: PAIR_COLOR[id], 'font-size': 10 }));
    });
  });

  niceTicks(maxX, 6).filter((t) => t >= minX).forEach((t) =>
    svg.appendChild(svgText(x(t), H - padB + 20, t.toFixed(t < 10 ? 1 : 0), { 'text-anchor': 'middle' })));
  svg.appendChild(svgText((padL + W - padR) / 2, H - 4, 'pipeline latency (ms)',
    { 'text-anchor': 'middle', fill: '#6b7683' }));

  host.appendChild(svg);
}

function renderPercentileTable(a, b) {
  const table = $('percentile-table');
  table.innerHTML = '';
  table.appendChild(el('thead', {},
    el('tr', {},
      el('th', { text: 'Metric' }),
      el('th', { text: 'NanoOWL + NanoSAM' }),
      el('th', { text: 'NanoOWL + EfficientViT-SAM' }))));

  const body = el('tbody');
  const row = (label, av, bv) => body.appendChild(
    el('tr', {}, el('td', { text: label }), el('td', { text: av }), el('td', { text: bv })));
  const section = (label) => body.appendChild(
    el('tr', { class: 'sect' }, el('td', { colspan: 3, text: label })));

  section('Pipeline latency — model stages only (ms)');
  ['min', 'p50', 'p90', 'p95', 'p99', 'max', 'mean', 'std'].forEach((k) =>
    row(k === 'std' ? 'std (jitter)' : k, fmt(a.pipeline[k], 2), fmt(b.pipeline[k], 2)));

  section('End-to-end incl. video decode (ms)');
  ['p50', 'p95', 'mean'].forEach((k) => row(k, fmt(a.e2e[k], 2), fmt(b.e2e[k], 2)));

  section('Throughput');
  row('frames ÷ pipeline time (fps)', fmt(a.throughput_fps, 2), fmt(b.throughput_fps, 2));
  row('mean of 1/latency (fps)', fmt(a.mean_instantaneous_fps, 2), fmt(b.mean_instantaneous_fps, 2));

  section('Run context');
  row('frames measured', a.frames_measured, b.frames_measured);
  row('frames excluded as warm-up', a.frames_warmup, b.frames_warmup);
  row('total detections', a.detections_total, b.detections_total);
  row('mean detections / frame', fmt(a.mean_detections_per_frame, 2), fmt(b.mean_detections_per_frame, 2));
  row('frames with no detections', a.frames_with_no_detections, b.frames_with_no_detections);
  row('model load (ms)', fmt(a.model_load_ms, 0), fmt(b.model_load_ms, 0));
  row('wall clock (s)', fmt(a.wall_clock_s, 1), fmt(b.wall_clock_s, 1));

  table.appendChild(body);
}

function renderVideoSlot(job) {
  const slot = $('video-slot');
  if (job.has_video) {
    if (slot.dataset.job === job.job_id) return;
    slot.dataset.job = job.job_id;
    slot.innerHTML = '';
    const video = el('video', { controls: '', preload: 'metadata', playsinline: '' });
    video.src = `/api/jobs/${job.job_id}/video`;
    slot.appendChild(video);
    slot.appendChild(el('div', { style: 'margin-top:12px' },
      el('a', { class: 'btn btn-ghost btn-sm', href: `/api/jobs/${job.job_id}/video`,
                download: `comparison_${job.job_id}.mp4`, text: 'Download video' })));
  } else if (job.state === 'rendering_video') {
    slot.dataset.job = '';
    slot.innerHTML = '';
    slot.appendChild(el('div', { class: 'video-pending' },
      el('div', { class: 'spinner' }),
      el('span', { text: 'Rendering side-by-side comparison — the measurements above are final.' })));
  } else if (job.state === 'complete_no_video') {
    slot.dataset.job = '';
    slot.innerHTML = '';
    slot.appendChild(el('div', { class: 'note note-warn' , text:
      job.video_error
        ? `The comparison video could not be rendered: ${job.video_error}. The benchmark data above is unaffected.`
        : 'Video rendering was disabled for this run. The benchmark data above is unaffected.' }));
  }
}

/* ── history ─────────────────────────────────────────────────────────── */
async function loadHistory() {
  const res = await fetch('/api/jobs').catch(() => null);
  if (!res || !res.ok) return;
  const { jobs } = await res.json();
  const list = $('history-list');
  list.innerHTML = '';
  if (!jobs.length) {
    list.appendChild(el('p', { class: 'muted small', text: 'No runs yet.' }));
    return;
  }
  jobs.forEach((job) => {
    const ok = job.state === 'complete' || job.state === 'complete_no_video';
    const bad = job.state === 'failed' || job.state === 'cancelled';
    list.appendChild(
      el('div', { class: 'hitem', onclick: () => { attachToJob(job.job_id); renderJob(job); $('history-panel').classList.add('hidden'); $('progress').classList.remove('hidden'); } },
        el('div', {},
          el('div', { class: 'hitem-name', text: job.filename }),
          el('div', { class: 'hitem-meta', text: new Date(job.created_at * 1000).toLocaleString() })),
        el('span', { class: `hstate ${ok ? 'hstate-ok' : bad ? 'hstate-bad' : 'hstate-run'}`, text: job.state })));
  });
}

/* ── boot ────────────────────────────────────────────────────────────── */
async function init() {
  initUpload();
  $('history-toggle').addEventListener('click', () => {
    $('history-panel').classList.toggle('hidden');
    loadHistory();
  });
  $('history-close').addEventListener('click', () => $('history-panel').classList.add('hidden'));

  try {
    const config = await (await fetch('/api/config')).json();
    state.serverConfig = config;
    const badge = $('backend-badge');
    if (config.is_mock) {
      const missing = config.missing_modules || [];
      badge.className = 'badge badge-mock';
      badge.textContent = missing.length
        ? `⚠ mock — missing ${missing.join(', ')}`
        : '⚠ mock backend — synthetic numbers';
      badge.title = (missing.length
        ? `Not importable on this host: ${missing.join(', ')}. Install them, or point `
          + `repo_paths in config.yaml at their git clones.\n\n`
        : '')
        + 'Latencies are synthetic and must not be used as study data.';
    } else {
      badge.className = 'badge badge-live';
      badge.textContent = 'jetson backend';
    }
    const d = config.defaults;
    $('prompts').value = d.prompts.join(', ');
    $('threshold').value = d.threshold;
    $('stride').value = d.stride;
    $('warmup').value = d.warmup_frames;
    $('record-masks').checked = d.record_masks;
  } catch (_) {
    $('backend-badge').textContent = 'server unreachable';
  }

  loadHistory();
}

init();
