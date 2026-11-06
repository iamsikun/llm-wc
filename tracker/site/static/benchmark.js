/* Benchmark report page. Loads data/<key>/<edition>.json and renders every panel. */
(function () {
  "use strict";
  const { el, css, esc, fmt, line, onResize, Tip, rulerFigure, rpmFigure, rpmLegend } = window.Site;
  const B = window.BENCH;
  const ROOT = "../";
  const $ = (id) => document.getElementById(id);
  const state = { doc: null, edition: 0, level: "80", history: null, cache: {} };

  const rankSpan = (a, b) => (a === b ? `${a}` : `${a}–${b}`);
  const pct = (v) => `${fmt(v, 1)}%`;

  async function load(i) {
    const e = B.editions[i];
    if (!state.cache[e.json]) {
      const res = await fetch(`${ROOT}${e.json}?v=${B.v}`);
      if (!res.ok) throw new Error(`Could not load ${e.json}`);
      state.cache[e.json] = await res.json();
    }
    state.edition = i;
    state.doc = state.cache[e.json];
    render();
  }

  function tierRows(level) { return state.doc.tiers.filter((t) => String(t.rank_mass) === level); }

  function renderHead() {
    const d = state.doc, e = B.editions[state.edition], top = d.models[0], s = d.summary;
    const t1 = tierRows("80")[0];
    const adjusted = top.raw_score_pct - top.adjustment_pp;
    const tie = top.rank_start !== top.rank_end ? ` (tied at ranks ${rankSpan(top.rank_start, top.rank_end)})` : "";
    const leaders = d.models.filter((m) => m.rank_start === 1);
    const who = leaders.length > 1 ? `${leaders.length} systems tie for #1 at` : `The top system, <strong>${esc(top.model)}</strong>, scores`;
    $("headline").innerHTML =
      `${who} <strong>${pct(top.raw_score_pct)}</strong>. Adjusted for the winner's curse, that is <strong>${pct(adjusted)}</strong>. ` +
      `${leaders.length > 1 ? "They rank" : "It ranks"} anywhere from <strong>#${Math.min(...leaders.map((m) => m.rank_low_80))} to ` +
      `#${Math.max(...leaders.map((m) => m.rank_high_80))}</strong> in 80% of ` +
      `leaderboards rebuilt from resampled tasks, and the tasks cannot order the top <strong>${t1.size} of ${d.n_models}</strong> systems.`;
    const up = e.upstream || {};
    const rev = /^[0-9a-f]{7,}$/.test(String(up.revision || "")) ? String(up.revision).slice(0, 7) : "";
    $("edition-note").innerHTML = (up.url ? `<a href="${esc(up.url)}">${rev ? `Revision ${esc(rev)}` : "Source"}</a>` : rev ? `Revision ${esc(rev)}` : "") +
      (B.checked ? `${up.url || rev ? " · " : ""}source last checked ${esc(B.checked)}` : "");
    const stats = [
      ["Inflation at #1", fmt(s.position_1_adjustment_pp), "pp", tie ? `Tied systems share the average over their positions: ${fmt(s.leader_adjustment_pp)} pp` : "Estimated, for the top position"],
      ["Top tier", t1.size, `of ${d.n_models}`, `Systems the tasks cannot order at the top · ${s.tiers_80} tiers in all`],
      ["Top system ranks #1 in", fmt(100 * s.leader_p_rank_1, 0), "%", "of rebuilt leaderboards"],
      ["Tasks · systems", d.n_tasks.toLocaleString(), `· ${d.n_models}`, `Each rebuilt leaderboard uses ${d.params.m.toLocaleString()} resampled tasks`],
    ];
    $("stats").innerHTML = stats.map(([label, value, unit, note]) =>
      `<div class="stat"><div class="label">${label}</div><div class="value">${value}<span class="unit">${unit}</span></div><div class="note">${note}</div></div>`).join("");
  }

  function intervalSvg(lo, hi, obs, j, tier1) {
    const w = 160, h = 18, pad = 5;
    const x = (k) => pad + (j <= 1 ? 0.5 : (k - 1) / (j - 1)) * (w - 2 * pad);
    const color = tier1 ? css("--accent") : css("--text-2");
    return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">` +
      `<line x1="${pad}" x2="${w - pad}" y1="9" y2="9" stroke="${css("--grid")}" stroke-width="2" stroke-linecap="round"/>` +
      `<line x1="${x(lo)}" x2="${x(hi)}" y1="9" y2="9" stroke="${color}" stroke-width="6" stroke-linecap="round" opacity="${tier1 ? 1 : 0.55}"/>` +
      `<circle cx="${x(obs)}" cy="9" r="3.6" fill="${css("--text")}" stroke="${css("--surface")}" stroke-width="1.5"/></svg>`;
  }

  function renderTable() {
    const d = state.doc, lv = state.level;
    const q = $("lb-search").value.trim().toLowerCase();
    const only1 = $("lb-tier1").checked;
    const limit = $("lb-rows").value === "all" ? Infinity : Number($("lb-rows").value);
    let shown = 0, prevTier = null;
    const rows = [];
    const visible = d.models.filter((m) => (!only1 || m[`tier_${lv}`] === 1) &&
      (!q || m.model.toLowerCase().includes(q) || String(m.model_id).toLowerCase().includes(q)));
    const shownRows = q ? visible : visible.slice(0, limit);
    // Interval axis spans the ranks the shown rows can reach, so short intervals stay legible.
    const axisMax = Math.min(d.n_models, Math.max(5, ...shownRows.map((m) => m[`rank_high_${lv}`])));
    $("lb-axis").textContent = `ranks 1–${axisMax}`;
    d.models.forEach((m) => {
      const tier = m[`tier_${lv}`];
      if (only1 && tier !== 1) return;
      if (q && !m.model.toLowerCase().includes(q) && !String(m.model_id).toLowerCase().includes(q)) return;
      if (shown >= limit && !q) return;
      shown += 1;
      const lo = m[`rank_low_${lv}`], hi = m[`rank_high_${lv}`];
      const adj = m.adjustment_low_pp === m.adjustment_high_pp ? fmt(m.adjustment_pp)
        : `${fmt(m.adjustment_pp)}<span class="sub">${fmt(m.adjustment_low_pp)} to ${fmt(m.adjustment_high_pp)}</span>`;
      const cls = [tier % 2 === 0 ? "tier-even" : "", tier !== prevTier && prevTier !== null && !q ? "tier-start" : "", tier === 1 ? "t1" : ""].join(" ");
      prevTier = tier;
      rows.push(`<tr class="${cls}">` +
        `<td class="num">${esc(m.rank)}</td>` +
        `<td class="model-cell"><span class="model" title="${esc(m.model_id)}">${esc(m.model)}</span></td>` +
        `<td class="num">${fmt(m.raw_score_pct, 1)}</td>` +
        `<td class="num"><strong>${fmt(m.raw_score_pct - m.adjustment_pp, 1)}</strong></td>` +
        `<td class="num adj">${adj}</td>` +
        `<td class="interval-cell" title="Ranks ${rankSpan(lo, hi)}">${intervalSvg(lo, hi, m.rank_start, axisMax, tier === 1)}<span class="sub">${rankSpan(lo, hi)}</span></td>` +
        `<td class="num"><span class="tier-badge${tier === 1 ? " t1" : ""}">T${tier}</span></td>` +
        `<td class="num">${fmt(100 * m.p_top_5, 0)}%</td></tr>`);
    });
    $("lb-body").innerHTML = rows.join("") || `<tr><td colspan="8" class="muted">No models match.</td></tr>`;
    const tiers = tierRows(lv);
    $("lb-count").textContent = `Showing ${shown} of ${d.n_models} models · ${tiers.length} tiers at ${lv}% rank mass` +
      (shown < d.n_models && !q && !only1 ? " · choose “All” to see the rest" : "");
  }

  // --- Rank probability matrix --------------------------------------------
  function rpmCount() {
    const raw = $("rpm-size").value, j = state.doc.n_models;
    if (raw === "tier1") return Math.max(2, Math.min(j, tierRows(state.level)[0].size));
    return raw === "all" ? j : Math.min(Number(raw), j);
  }
  function renderRpm() {
    const n = rpmCount();
    const out = rpmFigure($("rpm-figure"), state.doc, { level: state.level, rows: n, cols: Math.max(20, n) });
    rpmLegend($("rpm-legend"), out.values);
  }

  function renderPosition() {
    const d = state.doc;
    const xs = d.position_bias.map((p) => String(p.position));
    line($("pos-chart"), xs, [
      { name: "m-out-of-N", values: d.position_bias.map((p) => p.adjustment_pp), color: css("--accent") },
      { name: "Ordinary", values: d.position_bias.map((p) => p.ordinary_adjustment_pp), color: css("--series-2") },
    ], { yUnit: " pp", labels: xs.map((x) => `Position ${x}`), height: 250 });
  }

  async function renderHistory() {
    if (!B.history) return;
    if (!state.history) state.history = await (await fetch(`${ROOT}data/${B.key}/history.json?v=${B.v}`)).json();
    const h = state.history;
    const xs = h.map((r) => r.date.slice(0, 7));
    const labels = h.map((r) => r.date);
    const extra = (i) => `<br>Leader: ${esc(h[i].leader_model)}`;
    const time = h.map((r) => Date.parse(r.date + "T00:00:00Z"));
    const accent = css("--accent");
    line($("hist-adj"), xs, [{ name: "Rank-1 adjustment", values: h.map((r) => r.position_1_adjustment_pp), color: accent, area: true }], { yUnit: " pp", labels, extra, time, height: 190 });
    line($("hist-tier"), xs, [{ name: "Tier 1 size", values: h.map((r) => r.tier_1_size_80), color: accent, area: true }], { labels, digits: 0, digits0: true, time, height: 190 });
    line($("hist-j"), xs, [{ name: "Models", values: h.map((r) => r.n_models), color: accent, area: true }], { labels, digits: 0, digits0: true, time, height: 190 });
  }

  function renderMeta() {
    const d = state.doc, e = B.editions[state.edition], up = e.upstream || {};
    const items = [
      ["Source data", esc(e.date)],
      ["Snapshot", `<code>${esc(d.snapshot_id)}</code>`],
      ["Upstream revision", up.revision ? (up.url ? `<a href="${esc(up.url)}"><code>${esc(String(up.revision).slice(0, 12))}</code></a>` : `<code>${esc(String(up.revision).slice(0, 12))}</code>`) : "—"],
      ["Matrix", `${d.n_tasks.toLocaleString()} tasks × ${d.n_models} models`],
      ["Bootstrap", `<math><mi>m</mi><mo>=</mo><mo stretchy="false">⌊</mo><msup><mi>N</mi><mn>${d.params.gamma}</mn></msup><mo stretchy="false">⌋</mo><mo>=</mo><mn>${d.params.m.toLocaleString()}</mn></math>, ` +
        `<math><mi>B</mi><mo>=</mo><mn>${d.params.n_bootstraps.toLocaleString()}</mn></math>, seed ${d.params.seed}`],
    ];
    if (d.generated_at) items.push(["Analyzed", esc(d.generated_at.slice(0, 10))]);
    $("edition-meta").innerHTML = items.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
    $("downloads").innerHTML = [["JSON report", e.json], ["Leaderboard CSV", e.csv], ["RPM CSV", e.rpm_csv], ["Tiers CSV", e.tiers_csv]]
      .map(([label, href]) => `<a class="btn small" href="${ROOT}${href}" download>${label}</a>`).join("");
    const ex = d.excluded || [];
    $("excluded").innerHTML = ex.length ? `<details style="margin-top:1rem"><summary class="small">${ex.length} model${ex.length > 1 ? "s" : ""} excluded from the matrix</summary><ul class="list-tight small" style="margin-top:0.5rem">${ex.slice(0, 200).map((x) => `<li><code>${esc(x.model_id || x.model)}</code>: ${esc(x.reason || x.error || "incomplete or invalid scores")}</li>`).join("")}</ul></details>` : "";
  }

  // Compare two systems: share of rebuilt leaderboards in which A ranks above B.
  async function renderCompare() {
    const e = B.editions[state.edition], panel = $("compare-panel");
    if (!e.pairwise) { panel.hidden = true; return; }
    panel.hidden = false;
    if (!state.cache[e.pairwise]) state.cache[e.pairwise] = await (await fetch(`${ROOT}${e.pairwise}?v=${B.v}`)).json();
    const pw = state.cache[e.pairwise], a = $("cmp-a"), b = $("cmp-b");
    if (a.dataset.for !== e.pairwise) {
      const opts = pw.models.map((m, i) => `<option value="${i}">${i + 1}. ${esc(m)}</option>`).join("");
      a.innerHTML = opts; b.innerHTML = opts; a.value = "0"; b.value = String(Math.min(1, pw.models.length - 1));
      a.dataset.for = e.pairwise;
    }
    const i = Number(a.value), j = Number(b.value), p = pw.p_above[i][j];
    $("cmp-out").innerHTML = i === j ? "Pick two different systems." :
      `<strong>${esc(pw.models[i])}</strong> ranks above <strong>${esc(pw.models[j])}</strong> in ` +
      `<span class="cmp-p">${fmt(100 * p, 0)}%</span> of rebuilt leaderboards.`;
    $("cmp-bar").style.width = `${100 * p}%`;
  }
  ["cmp-a", "cmp-b"].forEach((id) => $(id).addEventListener("change", renderCompare));

  function render() {
    renderHead(); rulerFigure($("key-figure"), state.doc, { level: state.level, animate: !state.drawn }); state.drawn = true;
    renderTable(); renderCompare(); renderRpm(); renderPosition(); renderMeta(); renderHistory();
  }

  document.querySelectorAll("[data-edition]").forEach((btn) => btn.addEventListener("click", () => {
    document.querySelectorAll("[data-edition]").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    load(Number(btn.dataset.edition));
  }));
  $("lb-level").addEventListener("change", (e) => { state.level = e.target.value; rulerFigure($("key-figure"), state.doc, { level: state.level }); renderTable(); renderRpm(); });
  ["lb-search", "lb-rows", "lb-tier1"].forEach((id) => $(id).addEventListener(id === "lb-search" ? "input" : "change", renderTable));
  $("rpm-size").addEventListener("change", renderRpm);
  onResize(() => { if (state.doc) { rulerFigure($("key-figure"), state.doc, { level: state.level }); renderRpm(); renderPosition(); renderHistory(); renderTable(); } });

  load(0).catch((err) => { $("headline").textContent = `${err.message}. Serve the site over HTTP (python -m llm_wc.tracker serve).`; });
})();
