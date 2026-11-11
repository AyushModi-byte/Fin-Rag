import test from "node:test";
import assert from "node:assert/strict";
import {
  escapeHtml, renderAnswerHtml, describeError, parseRetryAfter, formatDuration, docLabel,
  stageBreakdown, evidenceItems,
} from "../js/lib.js";

const D = "VERIZON_2022_10K";
const src = (...pages) => pages.map((page) => ({ doc_id: D, page }));

// ---------- escaping ----------
test("escapeHtml escapes the five dangerous characters", () => {
  assert.equal(escapeHtml(`<a href="x" onclick='y'>&`), "&lt;a href=&quot;x&quot; onclick=&#39;y&#39;&gt;&amp;");
});

// ---------- answer rendering: safety ----------
test("model output can never inject markup or attributes", () => {
  const html = renderAnswerHtml(`<script>alert(1)</script> <img src=x onerror=alert(1)> "><svg/onload=alert(1)>`, []);
  assert.ok(!/<script|<img|<svg/i.test(html), html);
  assert.ok(html.includes("&lt;script&gt;"));
});

test("a document id containing markup is not turned into a chip", () => {
  const html = renderAnswerHtml(`x [<b>${D}</b> p.23]`, src(23));
  assert.ok(!html.includes("<b>"));
  assert.ok(!html.includes("data-doc"));
});

test("links are never produced (no link syntax at all)", () => {
  const html = renderAnswerHtml("[click](javascript:alert(1))", []);
  assert.ok(!/<a\s/i.test(html));
});

// ---------- answer rendering: markdown subset ----------
test("paragraphs, line breaks and bold", () => {
  const html = renderAnswerHtml("First **bold** line\nsecond line\n\nNew paragraph", []);
  assert.equal(html, "<p>First <strong>bold</strong> line<br>second line</p><p>New paragraph</p>");
});

test("bullet and numbered lists", () => {
  assert.equal(renderAnswerHtml("- a\n- **b**\n* c", []), "<ul><li>a</li><li><strong>b</strong></li><li>c</li></ul>");
  assert.equal(renderAnswerHtml("1. a\n2. b", []), "<ol><li>a</li><li>b</li></ol>");
});

test("pipe tables become real tables with a header", () => {
  const html = renderAnswerHtml("| Year | Capex |\n|---|---:|\n| 2022 | $23.1 billion |\n| 2021 | $20.3 billion |", []);
  assert.ok(html.startsWith('<div class="table-wrap"><table>'));
  assert.ok(html.includes("<thead><tr><th>Year</th><th>Capex</th></tr></thead>"));
  assert.ok(html.includes("<td>2022</td><td>$23.1 billion</td>"));
  assert.equal((html.match(/<tr>/g) || []).length, 3);
});

test("a table cell with <br> shows a line break but nothing else survives", () => {
  const html = renderAnswerHtml("| a |\n|---|\n| x<br>y<script> |", []);
  assert.ok(html.includes("x<br>y&lt;script&gt;"));
});

test("headings become small headings, never h1", () => {
  const html = renderAnswerHtml("## Interpretation\ntext", []);
  assert.ok(html.includes("<h4>Interpretation</h4>") && !html.includes("<h1"));
});

test("empty input renders to an empty string", () => {
  assert.equal(renderAnswerHtml("", []), "");
  assert.equal(renderAnswerHtml("   \n\n ", []), "");
});

// ---------- citations ----------
const chip = (page) => `data-doc="${D}" data-page="${page}"`;

test("the standard bracket becomes a chip", () => {
  const html = renderAnswerHtml(`Capex was high [${D} p.23].`, src(23));
  assert.ok(html.includes(chip(23)) && html.includes(">p.23<") && !html.includes("[VERIZON"));
});

for (const [name, text] of [
  ["corner brackets", `x 【${D} p.23】`],
  ["full-width square brackets", `x ［${D} p.23］`],
  ["narrow no-break space", `x [${D} p.23]`],
  ["spaces inside", `x [ ${D} p.23 ]`],
  ["parentheses", `x (${D} p.23)`],
  ["no dot", `x [${D} p23]`],
]) {
  test(`citation variant: ${name}`, () => {
    assert.ok(renderAnswerHtml(text, src(23)).includes(chip(23)));
  });
}

test("a page list in one bracket gives one chip per page", () => {
  const html = renderAnswerHtml(`x [${D} p.23, p.56]`, src(23, 56));
  assert.ok(html.includes(chip(23)) && html.includes(chip(56)));
});

test("page ranges expand only to retrieved pages", () => {
  const html = renderAnswerHtml(`x [${D} p.22‑25]`, src(23, 25, 99));
  assert.ok(html.includes(chip(23)) && html.includes(chip(25)));
  assert.ok(!html.includes(chip(22)) && !html.includes(chip(24)) && !html.includes(chip(99)));
});

test("a huge range does not hang or invent pages", () => {
  const html = renderAnswerHtml(`x [${D} p.1-99999999999]`, src(23));
  assert.ok(html.includes(chip(23)));
});

test("two documents in one bracket", () => {
  const html = renderAnswerHtml(`x [${D} p.23; ADOBE_2022_10K p.40]`, [...src(23), { doc_id: "ADOBE_2022_10K", page: 40 }]);
  assert.ok(html.includes(chip(23)) && html.includes('data-doc="ADOBE_2022_10K" data-page="40"'));
});

test("a citation to a page that was not retrieved stays as plain text", () => {
  const html = renderAnswerHtml(`x [${D} p.99]`, src(23));
  assert.ok(!html.includes("data-doc") && html.includes(`[${D} p.99]`));
});

test("brackets that only look like citations are left alone", () => {
  const html = renderAnswerHtml("See [note p.60] and (p.61) and [1].", src(60, 61));
  assert.ok(!html.includes("data-doc") && html.includes("[note p.60]"));
});

test("citations inside table cells and bold text work", () => {
  const html = renderAnswerHtml(`| a |\n|---|\n| **5** [${D} p.23] |`, src(23));
  assert.ok(html.includes("<strong>5</strong>") && html.includes(chip(23)));
});

test("a chip is a button with an accessible name", () => {
  const html = renderAnswerHtml(`x [${D} p.23]`, src(23), { labels: { [D]: "Verizon FY2022" } });
  assert.ok(html.includes('<button type="button" class="cite"'));
  assert.ok(html.includes('aria-label="Show source: Verizon FY2022, page 23"'));
});

// ---------- errors ----------
test("429 reports the wait from Retry-After", () => {
  const e = describeError({ status: 429, retryAfter: "41", detail: "Rate limit reached (minute); slow down" });
  assert.equal(e.title, "Too many questions");
  assert.equal(e.retrySeconds, 41);
  assert.match(e.message, /41 s/);
});

test("503 daily cap, 503 model quota, other 503", () => {
  assert.equal(describeError({ status: 503, detail: "Daily capacity reached; try again tomorrow" }).title, "Daily capacity reached");
  assert.equal(describeError({ status: 503, detail: "LLM daily budget nearly exhausted; try later" }).title, "Model quota used up for today");
  assert.equal(describeError({ status: 503, detail: "LLM temporarily unavailable; try later" }).title, "Temporarily unavailable");
});

test("422 from FastAPI (array detail) is turned into advice", () => {
  const e = describeError({ status: 422, detail: [{ msg: "Value error, question must not be empty" }] });
  assert.equal(e.title, "Check your question");
  assert.match(e.message, /empty/i);
  assert.equal(describeError({ status: 422, detail: "question longer than 2000 characters" }).title, "Check your question");
});

test("network failure and timeout", () => {
  assert.equal(describeError({ network: true }).title, "Can't reach the service");
  assert.equal(describeError({ timeout: true }).title, "This is taking too long");
});

test("401 and unknown 5xx", () => {
  assert.equal(describeError({ status: 401 }).title, "Access restricted");
  assert.equal(describeError({ status: 500 }).title, "Something went wrong");
});

test("server text is never echoed into the message", () => {
  const e = describeError({ status: 503, detail: "<img src=x onerror=1>" });
  assert.ok(!e.message.includes("<img"));
});

test("parseRetryAfter", () => {
  assert.equal(parseRetryAfter("59"), 59);
  assert.equal(parseRetryAfter(" 7 "), 7);
  assert.equal(parseRetryAfter(null), null);
  assert.equal(parseRetryAfter("soon"), null);
  assert.equal(parseRetryAfter("-3"), null);
  assert.equal(parseRetryAfter("999999"), 86400);
});

// ---------- formatting ----------
test("formatDuration", () => {
  assert.equal(formatDuration(0.4), "<1 ms");
  assert.equal(formatDuration(850.4), "850 ms");
  assert.equal(formatDuration(2096), "2.1 s");
  assert.equal(formatDuration(65000), "65 s");
  assert.equal(formatDuration(undefined), "n/a");
});

test("docLabel uses the lookup, falls back to the id", () => {
  assert.equal(docLabel(D, { [D]: "Verizon FY2022" }), "Verizon FY2022");
  assert.equal(docLabel("AMERICANEXPRESS_2022_10K", {}), "AMERICANEXPRESS FY2022");
  assert.equal(docLabel("weird", {}), "weird");
});

test("stageBreakdown for a normal answer, with the remainder as 'other'", () => {
  const s = stageBreakdown({ retrieve: 300, rerank: 0, llm: 1800, total: 2200 });
  assert.deepEqual(s.map((x) => x.key), ["retrieve", "llm", "other"]);
  assert.equal(s[2].ms, 100);
  assert.ok(Math.abs(s.reduce((a, x) => a + x.pct, 0) - 100) < 1e-6);
});

test("stageBreakdown for a cached answer and for missing data", () => {
  assert.deepEqual(stageBreakdown({ cache: 1.5, total: 2 }).map((x) => x.key), ["cache", "other"]);
  assert.deepEqual(stageBreakdown({}), []);
  assert.deepEqual(stageBreakdown(undefined), []);
});

// ---------- evidence ----------
test("evidenceItems pairs contexts with sources and marks the cited ones", () => {
  const items = evidenceItems(
    { contexts: ["a", "b", "c"], sources: src(10, 23, 23), citations: src(23) },
    { [D]: "Verizon FY2022" },
  );
  assert.deepEqual(items.map((i) => [i.page, i.cited]), [[10, false], [23, true], [23, true]]);
  assert.equal(items[0].label, "Verizon FY2022");
  assert.equal(items[1].id, `ev-${D}-23-1`);
});

test("evidenceItems copes with results that have no sources (older cache entries)", () => {
  const items = evidenceItems({ contexts: ["a", "b"], citations: [] }, {});
  assert.equal(items.length, 2);
  assert.equal(items[0].page, null);
  assert.equal(items[0].label, "Passage 1");
});

// ---------- typography ----------
test("narrow and no-break spaces from the model render as ordinary spaces", () => {
  const html = renderAnswerHtml("In fiscal\u202f2022\u00a0Verizon reported $23.1\u202fbillion", []);
  assert.equal(html, "<p>In fiscal 2022 Verizon reported $23.1 billion</p>");
});

test("citations still work after the spaces are normalised", () => {
  const html = renderAnswerHtml(`x\u202f[${D}\u202fp.23]`, src(23));
  assert.ok(html.startsWith("<p>x ") && html.includes(chip(23)));
});
