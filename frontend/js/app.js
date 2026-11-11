import {
  escapeHtml, renderAnswerHtml, describeError, formatDuration, docLabel, stageBreakdown, evidenceItems,
} from "./lib.js";

const cfg = window.FINRAG_CONFIG;
const $ = (id) => document.getElementById(id);
const els = {
  form: $("ask"), q: $("q"), count: $("q-count"), submit: $("submit"), result: $("result"),
  status: $("status"), statusText: $("status-text"), filings: $("filings"),
};
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let labels = {}; // doc_id -> "Company FY2022"
let inFlight = null; // AbortController of the running request
let cooldownTimer = null;

// ---------------------------------------------------------------------------------------------
// Coverage list + doc labels (static data generated from data/eval/docs.json)
// ---------------------------------------------------------------------------------------------
async function loadFilings() {
  try {
    const res = await fetch("data/filings.json");
    const companies = await res.json();
    for (const c of companies) for (const f of c.filings) labels[f.doc_id] = `${c.company} FY${f.year}`;
    els.filings.innerHTML = companies
      .map((c) => `<li><span class="name">${escapeHtml(c.company)}</span><span class="years">${c.filings.map((f) => `FY${f.year}`).join(" · ")}</span></li>`)
      .join("");
  } catch {
    els.filings.innerHTML = "<li>Coverage list unavailable.</li>";
  }
}

// ---------------------------------------------------------------------------------------------
// Backend status. A free host sleeps when idle: poll /health so the first question finds it awake.
// ---------------------------------------------------------------------------------------------
function setStatus(state, text) {
  els.status.dataset.state = state;
  els.statusText.textContent = text;
}

async function pollHealth() {
  const started = Date.now();
  setStatus("starting", "Connecting");
  while (true) {
    try {
      const res = await fetch(`${cfg.apiBase}/health`, { cache: "no-store" });
      if (res.ok) { setStatus("online", "Online"); return; }
    } catch { /* not up yet */ }
    if (Date.now() - started > cfg.healthPollMaxMs) { setStatus("offline", "Offline"); return; }
    setStatus("starting", "Starting up");
    await new Promise((r) => setTimeout(r, cfg.healthPollMs));
  }
}

// ---------------------------------------------------------------------------------------------
// Form
// ---------------------------------------------------------------------------------------------
function updateCount() {
  const n = els.q.value.length;
  els.count.textContent = `${n} / ${cfg.maxQuestionChars}`;
  els.count.dataset.over = String(n > cfg.maxQuestionChars);
}

function setBusy(busy) {
  els.result.setAttribute("aria-busy", String(busy));
  els.submit.setAttribute("aria-disabled", String(busy || cooldownTimer !== null));
  els.submit.textContent = busy ? "Asking…" : "Ask";
}

function startCooldown(seconds) {
  clearInterval(cooldownTimer);
  let left = seconds;
  const tick = () => {
    if (left <= 0) {
      clearInterval(cooldownTimer);
      cooldownTimer = null;
      els.submit.setAttribute("aria-disabled", "false");
      els.submit.textContent = "Ask";
      return;
    }
    els.submit.setAttribute("aria-disabled", "true");
    els.submit.textContent = `Wait ${left} s`;
    left -= 1;
  };
  tick();
  cooldownTimer = setInterval(tick, 1000);
}

// ---------------------------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------------------------
function renderLoading(waking) {
  els.result.innerHTML = `
    <div class="card" role="status">
      <h3>${waking ? "Still working" : "Searching the filings"}</h3>
      <p class="sub">${waking
        ? "The free backend may be waking up after being idle. The first answer can take up to a minute."
        : "Finding the relevant pages and writing a cited answer."}</p>
      <div class="skeleton" aria-hidden="true"><span></span><span></span><span></span></div>
    </div>`;
}

function renderNotice({ tone, title, message }) {
  els.result.innerHTML = `
    <div class="card notice" data-tone="${escapeHtml(tone)}" role="alert">
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(message)}</p>
    </div>`;
}

function latencyHtml(latency) {
  const rows = stageBreakdown(latency);
  if (!rows.length) return "";
  const alt = rows.map((r) => `${r.label} ${formatDuration(r.ms)}`).join(", ");
  return `
    <div class="card latency">
      <h3>Timing <span class="visually-hidden">(${escapeHtml(formatDuration(latency.total))} in total)</span></h3>
      <p class="sub">${escapeHtml(formatDuration(latency.total))} in total</p>
      <div class="bar" role="img" aria-label="${escapeHtml(alt)}">${rows.map((r) => `<span class="seg" data-k="${r.key}" data-w="${r.pct.toFixed(2)}"></span>`).join("")}</div>
      <ul>${rows.map((r) => `<li data-k="${r.key}"><i aria-hidden="true"></i>${escapeHtml(r.label)} <b>${escapeHtml(formatDuration(r.ms))}</b></li>`).join("")}</ul>
    </div>`;
}

function renderAnswer(result) {
  const idk = result.answer.trim().replace(/\.$/, "").toLowerCase() === "i don't know";
  const cached = typeof result.latency_ms?.cache === "number";
  const items = evidenceItems(result, labels);
  const cites = result.citations || [];

  const badge = idk
    ? `<span class="badge" data-kind="idk">No answer found</span>`
    : `<span class="badge">Answered</span>`;
  const cacheBadge = cached ? `<span class="badge" data-kind="cache">From cache</span>` : "";

  const body = idk
    ? `<div class="answer prose"><p>I don't know.</p><p>The retrieved passages do not contain a reliable answer, or the question is outside the covered filings. Try naming the company and fiscal year, or rephrase.</p></div>`
    : `<div class="answer prose">${renderAnswerHtml(result.answer, result.sources || [], { labels })}</div>`;

  const sources = cites.length
    ? `<div class="sources"><span class="label">Cited:</span>${cites.map((c) =>
        `<button type="button" class="cite" data-doc="${escapeHtml(c.doc_id)}" data-page="${c.page}" aria-label="Show source: ${escapeHtml(docLabel(c.doc_id, labels))}, page ${c.page}">${escapeHtml(docLabel(c.doc_id, labels))} · p.${c.page}</button>`).join("")}</div>`
    : "";

  const evidence = items.length
    ? `<div class="card">
        <h3>Retrieved passages (${items.length})</h3>
        <p class="sub">What the model was given. Cited passages are tagged.</p>
        ${items.map((it) => `
          <details class="passage" id="${escapeHtml(it.id)}"${it.doc ? ` data-doc="${escapeHtml(it.doc)}" data-page="${it.page}"` : ""}>
            <summary><span class="where">${escapeHtml(it.label)}${it.page ? ` · p.${it.page}` : ""}</span>${it.cited ? `<span class="tag">Cited</span>` : ""}</summary>
            <div class="body prose">${renderAnswerHtml(it.text, [])}</div>
          </details>`).join("")}
      </div>`
    : "";

  els.result.innerHTML = `
    <div class="card">
      <div class="meta">${badge}${cacheBadge}<span>${escapeHtml(formatDuration(result.latency_ms?.total))}</span></div>
      ${body}${sources}
    </div>
    ${latencyHtml(result.latency_ms)}
    ${evidence}`;

  // bar widths are set through the CSSOM (allowed under a style-src 'self' policy, unlike inline style attributes)
  els.result.querySelectorAll(".seg").forEach((s) => s.style.setProperty("width", `${s.dataset.w}%`));
}

// ---------------------------------------------------------------------------------------------
// Asking
// ---------------------------------------------------------------------------------------------
async function ask(question) {
  if (inFlight) inFlight.abort();
  const controller = new AbortController();
  inFlight = controller;
  let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, cfg.requestTimeoutMs);
  const wake = setTimeout(() => renderLoading(true), cfg.wakeNoticeMs);

  setBusy(true);
  renderLoading(false);
  els.result.focus({ preventScroll: true });
  els.result.scrollIntoView({ block: "start", behavior: reducedMotion ? "auto" : "smooth" });

  try {
    const res = await fetch(`${cfg.apiBase}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });
    let data = null;
    try { data = await res.json(); } catch { /* non-JSON error body */ }
    if (!res.ok) {
      const err = describeError({ status: res.status, retryAfter: res.headers.get("Retry-After"), detail: data?.detail });
      renderNotice(err);
      if (err.retrySeconds) startCooldown(err.retrySeconds);
      return;
    }
    setStatus("online", "Online");
    renderAnswer(data);
  } catch (e) {
    if (e.name === "AbortError" && !timedOut) return; // superseded by a newer question
    renderNotice(describeError(timedOut ? { timeout: true } : { network: true }));
    if (!timedOut) pollHealth();
  } finally {
    clearTimeout(timeout);
    clearTimeout(wake);
    if (inFlight === controller) { inFlight = null; setBusy(false); }
  }
}

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  if (els.submit.getAttribute("aria-disabled") === "true") return;
  const question = els.q.value.trim();
  if (!question) {
    els.q.setAttribute("aria-invalid", "true");
    renderNotice({ tone: "error", title: "Check your question", message: "Type a question first, for example one of the examples below." });
    els.q.focus();
    return;
  }
  if (question.length > cfg.maxQuestionChars) {
    els.q.setAttribute("aria-invalid", "true");
    renderNotice({ tone: "error", title: "Check your question", message: `The question is too long (limit ${cfg.maxQuestionChars} characters).` });
    els.q.focus();
    return;
  }
  els.q.removeAttribute("aria-invalid");
  ask(question);
});

els.q.addEventListener("input", () => { updateCount(); els.q.removeAttribute("aria-invalid"); });
els.q.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); els.form.requestSubmit(); }
});

document.querySelectorAll(".examples .chip").forEach((b) =>
  b.addEventListener("click", () => {
    els.q.value = b.dataset.q;
    updateCount();
    els.q.removeAttribute("aria-invalid");
    els.q.focus();
  }));

// A citation chip (in the answer or the "Cited" row) opens the matching retrieved passage.
els.result.addEventListener("click", (e) => {
  const chip = e.target.closest("button.cite");
  if (!chip) return;
  const sel = `details.passage[data-doc="${CSS.escape(chip.dataset.doc)}"][data-page="${CSS.escape(chip.dataset.page)}"]`;
  const target = els.result.querySelector(sel);
  if (!target) return;
  target.open = true;
  target.scrollIntoView({ block: "center", behavior: reducedMotion ? "auto" : "smooth" });
  target.querySelector("summary").focus({ preventScroll: true });
  target.dataset.flash = "true";
  setTimeout(() => delete target.dataset.flash, 1800);
});

updateCount();
loadFilings();
pollHealth();
