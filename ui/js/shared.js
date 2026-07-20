/* shared.js — utilities, data loading, and nav shell for every page.
   Loaded before any page script. Exposes globals: docs, DOSSIERS, UAP. */
"use strict";

const ROOT_REL = "..";   // ui/ sits under disclosure/
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

let docs = [];
let DOSSIERS = [];

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
}

function highlight(text, terms) {
  if (!text) return "";
  let html = escapeHtml(text);
  terms.forEach(t => {
    if (t.length < 2) return;
    const re = new RegExp("(" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "gi");
    html = html.replace(re, "<mark>$1</mark>");
  });
  return html;
}

function snippet(text, terms, len = 240) {
  if (!text) return "";
  let idx = -1;
  for (const t of terms) {
    if (t.length < 2) continue;
    const i = text.toLowerCase().indexOf(t.toLowerCase());
    if (i >= 0 && (idx === -1 || i < idx)) idx = i;
  }
  if (idx === -1) return text.slice(0, len);
  const start = Math.max(0, idx - 60);
  return (start > 0 ? "…" : "") + text.slice(start, start + len) + "…";
}

// Map a source image path (raw/...) to its pre-generated thumbnail path.
function thumbForPath(src) {
  if (!src) return "";
  if (!/\.(jpg|jpeg|png|gif|webp|jp2|tiff?)$/i.test(src)) return "";
  if (!src.startsWith("raw/")) return "";
  return "raw/thumbs/" + src.slice(4).replace(/\.(jpg|jpeg|png|gif|webp|jp2|tiff?)$/i, ".jpg");
}

function badge(t) {
  const cls = t === "PDF" ? "pdf" : t === "VID" ? "vid" : t === "AUD" ? "aud" : "img";
  return `<span class="badge ${cls}">${t || "?"}</span>`;
}

function fillFilter(sel, values) {
  const el = $(sel);
  if (!el) return;
  values.sort();
  for (const v of values) if (v) {
    const o = document.createElement("option");
    o.value = v; o.textContent = v; el.appendChild(o);
  }
}

function recordYear(d) {
  const s = (d.incident_date || "").trim();
  if (!s) return null;
  const m = s.match(/(\d{1,2})\/(\d{1,2})\/(\d{2,4})/);
  if (!m) return null;
  let y = parseInt(m[3], 10);
  if (y < 100) y += y < 30 ? 2000 : 1900;
  return y;
}

const SHAPE_PATTERNS = [
  ["tic-tac",   /\btic[\s-]?tacs?\b/i],
  ["triangle",  /\btriangular|triangle(?:-shaped)?\b/i],
  ["disc",      /\b(disc|disk|saucer|flying saucer|disc-shaped|disk-shaped)\b/i],
  ["sphere",    /\b(spheres?|spherical|orbs?|sphere-shaped|round)\b/i],
  ["cigar",     /\bcigar(?:-shaped)?\b/i],
  ["oval",      /\boval(?:-shaped)?|elliptical\b/i],
  ["cylinder",  /\bcylind(?:rical|er)\b/i],
  ["chevron",   /\bchevron(?:-shaped)?|boomerang\b/i],
  ["light",     /\b(bright lights?|glowing objects?|illuminated)\b/i],
];
function detectShape(d) {
  const corpus = ((d.blurb || "") + " " + (d.text || "").slice(0, 8000)).toLowerCase();
  for (const [name, re] of SHAPE_PATTERNS) {
    if (re.test(corpus)) return name;
  }
  return "";
}

// Dossier-hit relevance: hits with relevant===false were rejected by the AI
// false-positive filter; unclassified hits degrade gracefully to relevant.
function isRelevantHit(h) { return h.relevant !== false; }
function hasRelevantHit(d) {
  const hits = d.dossier_hits || {};
  for (const list of Object.values(hits)) for (const h of list) if (isRelevantHit(h)) return true;
  return false;
}

function dossierHitsRowHTML(d) {
  const hits = d.dossier_hits || {};
  const ids = Object.keys(hits);
  if (!ids.length) return "";
  const parts = [];
  for (const did of ids) {
    const dos = (DOSSIERS || []).find(x => x.id === did);
    const emoji = (dos && dos.emoji) || "🏷";
    for (const h of hits[did]) {
      if (!isRelevantHit(h)) continue;
      const label = escapeHtml(h.kw);
      const summary = h.summary ? `${h.summary}\n\n` : "";
      const tip = escapeHtml(`${dos ? dos.label : did} · pattern ${h.pat}\n\n${summary}…${h.ctx}…`);
      parts.push(`<span class="dossier-hit" title="${tip}">${emoji} ${label}</span>`);
    }
  }
  if (!parts.length) return "";
  return `<div class="dossier-hits">${parts.join("")}</div>`;
}

function markDossierHits(html, hits) {
  if (!hits) return html;
  for (const [did, list] of Object.entries(hits)) {
    list.forEach((h, i) => {
      try {
        const re = new RegExp("(" + h.pat + ")", "i");
        let replaced = false;
        const cls = isRelevantHit(h) ? "dossier-match" : "dossier-match dossier-match-rejected";
        html = html.replace(re, m => {
          if (replaced) return m;
          replaced = true;
          return `<mark class="${cls}" id="dh-${did}-${i}">${m}</mark>`;
        });
      } catch (_) { /* pattern doesn't compile in JS regex */ }
    });
  }
  return html;
}

const RELEASE_LABELS = {
  release_1: "Release 1 · war.gov · 8 May 2026",
  release_2: "Release 2 · war.gov · 22 May 2026",
  release_3: "Release 3 · war.gov · 12 Jun 2026",
  release_4: "Release 4 · war.gov · 10 Jul 2026",
};
function releaseLabel(d) { return RELEASE_LABELS[d.release] || ""; }

function docById(id) { return docs.find(d => d.id === id) || null; }

// ── Data loading ─────────────────────────────────────────────────────
// Every page starts here. Loads search-index.json + augments docs, then
// any optional extras the page asked for (graceful no-ops when absent).
const EXTRA_FILES = {
  incidents:   ["incidents.json",        j => { window.INCIDENTS = j.incidents || j; }],
  correlations:["correlations.json",     j => { window.CORRELATIONS = j; }],
  dedup:       ["dedup_clusters.json",   j => { window.DEDUP_CLUSTERS = j; }],
  communities: ["graph_communities.json",j => { window.GRAPH_COMMUNITIES = j; }],
  graphLayout: ["graph-layout.json",     j => { window.GRAPH_LAYOUT = j; }],
  topics:      ["topics.json",           j => { window.TOPICS = j; }],
  today:       ["today_index.json",      j => { window.TODAY_INDEX = j; }],
};

async function loadCoreData(extras = []) {
  const res = await fetch("search-index.json");
  docs = await res.json();

  for (const d of docs) {
    const fresh = [];
    for (const [did, list] of Object.entries(d.dossier_hits || {})) {
      if (list.some(isRelevantHit)) fresh.push(did);
    }
    d.dossiers = fresh;
    if (!("craft_shape" in d) || d.craft_shape == null) d.craft_shape = detectShape(d);
  }

  const jobs = extras
    .filter(k => EXTRA_FILES[k])
    .map(k => fetch(EXTRA_FILES[k][0])
      .then(r => r.ok ? r.json() : null)
      .then(j => { if (j) EXTRA_FILES[k][1](j); }));
  // Dossiers are needed by most pages; always try.
  jobs.push(fetch("dossiers.json").then(r => r.ok ? r.json() : null).then(j => { if (j) DOSSIERS = j; }));
  await Promise.allSettled(jobs);
  return docs;
}

// ── URL params (cross-page state) ────────────────────────────────────
function getParam(name) {
  return new URLSearchParams(location.search).get(name) || "";
}
function recordUrl(id) { return `record.html?id=${encodeURIComponent(id)}`; }
function searchUrl(q)  { return q ? `search.html?q=${encodeURIComponent(q)}` : "search.html"; }

// ── Nav shell ────────────────────────────────────────────────────────
const NAV_PAGES = [
  ["search.html",   "search",   "🔎 Search"],
  ["map.html",      "map",      "🗺 Map"],
  ["timeline.html", "timeline", "📅 Timeline"],
  ["graph.html",    "graph",    "🕸 Graph"],
  ["ask.html",      "ask",      "⌘ Ask"],
];

function renderNav(active) {
  const nav = document.createElement("nav");
  nav.className = "topnav";
  nav.innerHTML = `
    <a class="topnav-brand" href="index.html">UFO/UAP <span>Disclosure Archive</span></a>
    <div class="topnav-links">
      ${NAV_PAGES.map(([href, key, label]) =>
        `<a href="${href}" class="${key === active ? "active" : ""}">${label}</a>`).join("")}
    </div>
    <a class="topnav-coffee" href="https://buymeacoffee.com/uapdisclosuremirror" target="_blank" rel="noopener" title="Help keep this archive online">☕</a>`;
  document.body.prepend(nav);
  // ⌘K / Ctrl-K jumps to the Ask page from anywhere.
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k" && active !== "ask") {
      e.preventDefault();
      location.href = "ask.html";
    }
  });
}
