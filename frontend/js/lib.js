// Pure functions for the FinRag frontend: no DOM, no network, so they run under `node --test`.
//
// The answer text comes from a language model and is UNTRUSTED. Everything that reaches innerHTML goes
// through escapeHtml(); the only markup produced here is a fixed set of tags (p, br, strong, ul, ol, li,
// h4, table parts) and citation <button>s whose attributes are escaped. There is no link syntax.

const PUA_OPEN = "";
const PUA_CLOSE = "";

export function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// ---------------------------------------------------------------------------------------------
// Citations. Mirrors the backend parser (src/generation/grounded_answer.py): the model writes
// [doc p.N], U+3010 corner brackets, U+FF3B square brackets or (...), with stray spaces, page lists,
// ranges and several documents per bracket. Only pages that were actually retrieved become chips.
// ---------------------------------------------------------------------------------------------
const GROUP = /[\[【［〔(]([^\[\]【】［］〔〕()]*)[\]】］〕)]/g;
const RANGE = String.raw`\d+(?:\s*[-‐-—]\s*\d+)?`;
const PAGES = String.raw`pp?\.?\s*${RANGE}(?:\s*[,;&]\s*(?:pp?\.?\s*)?${RANGE})*`;
const SEGMENT = new RegExp(String.raw`([^\s,;]+)\s+(${PAGES})`, "g");
const PAGE_ITEM = /(\d+)(?:\s*[-‐-—]\s*(\d+))?/g;

function pagesBySource(sources) {
  const map = new Map();
  for (const s of sources || []) {
    if (!map.has(s.doc_id)) map.set(s.doc_id, new Set());
    map.get(s.doc_id).add(s.page);
  }
  return map;
}

function chipHtml(doc, page, labels) {
  const name = docLabel(doc, labels);
  return (
    `<button type="button" class="cite" data-doc="${escapeHtml(doc)}" data-page="${page}" ` +
    `aria-label="Show source: ${escapeHtml(name)}, page ${page}">p.${page}</button>`
  );
}

function chipsForGroup(inner, pages, labels) {
  const chips = [];
  for (const seg of inner.matchAll(SEGMENT)) {
    const doc = seg[1];
    const have = pages.get(doc);
    if (!have) continue;
    for (const item of seg[2].matchAll(PAGE_ITEM)) {
      const start = Number(item[1]);
      const end = item[2] ? Number(item[2]) : start;
      for (const p of [...have].filter((x) => x >= start && x <= Math.max(start, end)).sort((a, b) => a - b)) {
        const html = chipHtml(doc, p, labels);
        if (!chips.includes(html)) chips.push(html);
      }
    }
  }
  return chips.length ? chips.join(" ") : null;
}

// ---------------------------------------------------------------------------------------------
// Inline text: citations -> placeholders, escape, bold, <br> in table cells, placeholders -> chips.
// ---------------------------------------------------------------------------------------------
function inline(raw, ctx) {
  const store = [];
  let t = raw.replace(GROUP, (whole, inner) => {
    const chips = chipsForGroup(inner, ctx.pages, ctx.labels);
    if (!chips) return whole;
    store.push(chips);
    return `${PUA_OPEN}${store.length - 1}${PUA_CLOSE}`;
  });
  t = escapeHtml(t);
  t = t.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/&lt;br\s*\/?&gt;/gi, "<br>");
  return t.replace(new RegExp(`${PUA_OPEN}(\\d+)${PUA_CLOSE}`, "g"), (_, i) => store[Number(i)]);
}

// ---------------------------------------------------------------------------------------------
// Block structure: paragraphs, lists, headings, pipe tables.
// ---------------------------------------------------------------------------------------------
const TABLE_SEP = /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/;
const BULLET = /^\s*[-*•]\s+(.*)$/;
const NUMBERED = /^\s*\d+[.)]\s+(.*)$/;
const HEADING = /^\s{0,3}#{1,6}\s+(.*)$/;

function splitRow(line) {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|")) s = s.slice(0, -1);
  return s.split("|").map((c) => c.trim());
}

const isTableStart = (lines, i) => lines[i].trim().startsWith("|") && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1]);
const startsBlock = (lines, i) => BULLET.test(lines[i]) || NUMBERED.test(lines[i]) || HEADING.test(lines[i]) || isTableStart(lines, i);

export function renderAnswerHtml(text, sources = [], opts = {}) {
  const ctx = { pages: pagesBySource(sources), labels: opts.labels || {} };
  // Non-breaking and narrow spaces (the model likes U+202F) are shown as ordinary spaces: they look cramped.
  const lines = String(text ?? "")
    .replace(/[]/g, "")
    .replace(/[    ]/g, " ")
    .replace(/\r\n?/g, "\n")
    .split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    if (!lines[i].trim()) { i++; continue; }

    if (isTableStart(lines, i)) {
      const head = splitRow(lines[i]);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(splitRow(lines[i++]));
      const th = head.map((c) => `<th>${inline(c, ctx)}</th>`).join("");
      const body = rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c, ctx)}</td>`).join("")}</tr>`).join("");
      out.push(`<div class="table-wrap"><table><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table></div>`);
      continue;
    }

    const heading = lines[i].match(HEADING);
    if (heading) { out.push(`<h4>${inline(heading[1], ctx)}</h4>`); i++; continue; }

    const listRe = BULLET.test(lines[i]) ? BULLET : NUMBERED.test(lines[i]) ? NUMBERED : null;
    if (listRe) {
      const tag = listRe === BULLET ? "ul" : "ol";
      const items = [];
      while (i < lines.length && (lines[i].match(listRe))) items.push(`<li>${inline(lines[i++].match(listRe)[1], ctx)}</li>`);
      out.push(`<${tag}>${items.join("")}</${tag}>`);
      continue;
    }

    const para = [];
    while (i < lines.length && lines[i].trim() && (para.length === 0 || !startsBlock(lines, i))) para.push(inline(lines[i++].trim(), ctx));
    out.push(`<p>${para.join("<br>")}</p>`);
  }
  return out.join("");
}

// ---------------------------------------------------------------------------------------------
// Errors: map an API failure to plain language. Server text is never echoed into the message.
// ---------------------------------------------------------------------------------------------
export function parseRetryAfter(value) {
  if (value == null) return null;
  const s = String(value).trim();
  if (!/^\d+$/.test(s)) return null;
  return Math.min(Number(s), 86400);
}

function detailText(detail) {
  if (Array.isArray(detail)) return detail.map((d) => (d && d.msg) || "").join(" ").toLowerCase();
  return typeof detail === "string" ? detail.toLowerCase() : "";
}

export function describeError({ status, retryAfter, detail, network, timeout } = {}) {
  if (network) {
    return { tone: "error", retrySeconds: null, title: "Can't reach the service",
      message: "The backend didn't respond. It may be starting up (free hosting sleeps when idle). Wait a minute and try again." };
  }
  if (timeout) {
    return { tone: "warn", retrySeconds: null, title: "This is taking too long",
      message: "The request timed out. The service may still be waking up; try again in a minute." };
  }
  const d = detailText(detail);
  if (status === 429) {
    const wait = parseRetryAfter(retryAfter) ?? 60;
    return { tone: "warn", retrySeconds: wait, title: "Too many questions",
      message: `This demo limits each visitor to a few questions per minute and per day. Try again in ${wait} s.` };
  }
  if (status === 503) {
    if (d.includes("capacity")) {
      return { tone: "warn", retrySeconds: null, title: "Daily capacity reached",
        message: "This free public demo has used the question budget it can afford today. It resets at 00:00 UTC." };
    }
    if (d.includes("budget") || d.includes("quota")) {
      return { tone: "warn", retrySeconds: null, title: "Model quota used up for today",
        message: "The free language-model quota for today is used up. Try again tomorrow." };
    }
    return { tone: "warn", retrySeconds: null, title: "Temporarily unavailable",
      message: "The language model didn't respond. Try again shortly." };
  }
  if (status === 422) {
    let message = "The question couldn't be processed.";
    if (d.includes("empty")) message = "The question is empty.";
    else if (d.includes("longer") || d.includes("too long") || d.includes("at most")) message = "The question is too long.";
    return { tone: "error", retrySeconds: null, title: "Check your question", message };
  }
  if (status === 401 || status === 403) {
    return { tone: "error", retrySeconds: null, title: "Access restricted", message: "This service isn't open to this request." };
  }
  return { tone: "error", retrySeconds: null, title: "Something went wrong",
    message: "The service returned an unexpected error. Try again in a moment." };
}

// ---------------------------------------------------------------------------------------------
// Formatting and result shaping.
// ---------------------------------------------------------------------------------------------
export function formatDuration(ms) {
  if (typeof ms !== "number" || Number.isNaN(ms)) return "n/a";
  if (ms < 1) return "<1 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const s = ms / 1000;
  return s < 10 ? `${s.toFixed(1)} s` : `${Math.round(s)} s`;
}

export function docLabel(docId, labels = {}) {
  if (labels[docId]) return labels[docId];
  const m = /^(.+)_(\d{4})_10K$/.exec(docId);
  return m ? `${m[1]} FY${m[2]}` : docId;
}

const STAGES = [
  ["cache", "Cache lookup"],
  ["retrieve", "Retrieval"],
  ["rerank", "Rerank"],
  ["llm", "Language model"],
];

export function stageBreakdown(latency) {
  if (!latency || typeof latency.total !== "number" || latency.total <= 0) return [];
  const rows = STAGES.filter(([k]) => typeof latency[k] === "number" && latency[k] > 0)
    .map(([key, label]) => ({ key, label, ms: latency[key] }));
  const used = rows.reduce((a, r) => a + r.ms, 0);
  const other = Math.max(0, latency.total - used);
  if (other > 0.05) rows.push({ key: "other", label: "Overhead", ms: other });
  const denom = rows.reduce((a, r) => a + r.ms, 0);
  return rows.map((r) => ({ ...r, pct: (r.ms / denom) * 100 }));
}

export function evidenceItems(result, labels = {}) {
  const contexts = result?.contexts || [];
  const sources = result?.sources || [];
  const cited = result?.citations || [];
  return contexts.map((text, i) => {
    const s = sources[i];
    if (!s) return { id: `ev-${i}`, doc: null, page: null, label: `Passage ${i + 1}`, text, cited: false };
    return {
      id: `ev-${s.doc_id}-${s.page}-${i}`,
      doc: s.doc_id,
      page: s.page,
      label: docLabel(s.doc_id, labels),
      text,
      cited: cited.some((c) => c.doc_id === s.doc_id && c.page === s.page),
    };
  });
}
