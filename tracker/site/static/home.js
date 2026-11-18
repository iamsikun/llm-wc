/* Home page: the key figure for the featured leaderboard, the same figure for a picked one,
   and two cross-leaderboard bar charts. No tuning controls. */
(function () {
  "use strict";
  const { fmt, esc, hbar, onResize, rulerFigure, rpmFigure, rpmLegend, el, css, niceTicks, Tip, disambiguate } = window.Site;
  const $ = (id) => document.getElementById(id);
  const byKey = Object.fromEntries(window.SUMMARY.map((b) => [b.key, b]));
  const latest = (b) => b.editions.live || b.editions.reference;
  const cache = {};
  const docs = {};

  async function doc(key) {
    const e = latest(byKey[key]);
    if (!cache[e.json]) cache[e.json] = fetch(`${e.json}?v=${window.BUILD}`).then((r) => r.json()).then(disambiguate);
    return cache[e.json];
  }

  // "source data of <date>", linked to the analyzed upstream revision in a new tab.
  function source(d, date) {
    const url = d.upstream && d.upstream.url, text = `source data of ${esc(date)}`;
    return url ? `<a href="${esc(url)}" target="_blank" rel="noopener" data-track="upstream/${esc(d.key)}">${text} ↗</a>` : text;
  }

  function caption(b, d, k, num) {
    const e = latest(b);
    return `<span class="fig-num">Figure ${num}.</span> ` +
      "○ observed score, ● adjusted for the rank-induced bias. Each rank range holds the rank with 80% probability, " +
      "estimated by bootstrap. " +
      `Top ${k} of ${d.n_models} agents on ${d.n_tasks.toLocaleString()} tasks · ${source(d, e.date)}.`;
  }

  function stats(d) {
    const top = d.models[0], adj = top.raw_score_pct - top.adjustment_pp;
    const tier1 = d.models.filter((m) => m.tier_80 === 1).length;
    const range = top.rank_low_80 === top.rank_high_80 ? `#${top.rank_low_80}` : `#${top.rank_low_80}–${top.rank_high_80}`;
    return [
      [`${fmt(top.raw_score_pct, 1)}%<span class="u"> → ${fmt(adj, 1)}%</span>`, "Top score, observed → adjusted"],
      [range, "Rank range of the top agent"],
      [`${tier1}<span class="u"> of ${d.n_models}</span>`, "Agents in the top tier"],
    ].map(([v, l]) => `<div class="big-stat"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
  }

  async function drawHero() {
    const key = window.FEATURED, d = docs.hero = await doc(key);
    $("hero-caption").innerHTML = caption(byKey[key], d, rulerFigure($("hero-figure"), d, { animate: true }), 1);
    drawRpm();
  }

  function drawRpm() {
    const d = docs.hero, svg = $("rpm-figure");
    if (!d || !svg) return;
    const narrow = (svg.clientWidth || 900) < 640;
    const out = rpmFigure(svg, d, { cols: narrow ? 10 : 20 });
    rpmLegend($("rpm-legend"), out.values, out.pooled);
    $("rpm-caption").innerHTML = `<span class="fig-num">Figure 2.</span> Top ${out.k} of ${d.n_models} agents` +
      (out.pooled ? `, ranks past #${out.cols} pooled in the last column` : "") +
      ` · probabilities estimated by bootstrap · ${source(d, latest(byKey[window.FEATURED]).date)}.`;
  }

  async function drawPick() {
    const key = $("pick").value, d = docs.pick = await doc(key);
    $("pick-link").href = `benchmarks/${key}.html`;
    $("pick-stats").innerHTML = stats(d);
    // No reveal here: the picker redraws often, and the hero figure has already shown the motion once.
    $("pick-title").textContent = `RULER (Rank Uncertainty and Leaderboard Evaluation Report) for ${byKey[key].label}`;
    $("pick-caption").innerHTML = caption(byKey[key], d, rulerFigure($("pick-figure"), d), $("pick-caption").dataset.fig);
  }

  function drawOverview() {
    const rows = window.SUMMARY.map((b) => ({ b, e: latest(b) }));
    const tip = (b, e, line) => `<b>${esc(b.label)}</b><br>${line}<br>${e.n_tasks.toLocaleString()} tasks · ${e.n_models} agents · ${e.date}`;
    hbar($("overview-inflation"), rows.map(({ b, e }) => ({
      label: b.short_label, value: e.position_1_adjustment, href: `benchmarks/${b.key}.html`,
      tip: tip(b, e, `Inflation at #1 <b>${fmt(e.position_1_adjustment)} pp</b>`),
    })), { unit: " pp" });
    hbar($("overview-tier"), rows.map(({ b, e }) => ({
      label: b.short_label, value: e.tier_1_size_80, href: `benchmarks/${b.key}.html`,
      tip: tip(b, e, `Top tier <b>${e.tier_1_size_80} of ${e.n_models}</b> agents`),
    })), { digits: 0 });
  }

  // Story: a timeline of each #1's reign, with a marker at every handover.
  function drawStory() {
    const S = window.STORY, svg = $("story-chart");
    if (!S || !svg) return;
    svg.innerHTML = "";
    const width = svg.clientWidth || 900, narrow = width < 560;
    const height = 104, m = { l: 8, r: 8 }, laneY = 50, laneH = 30;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`); svg.setAttribute("height", height);
    const P = (d) => Date.parse(d + "T00:00:00Z");
    const t0 = P(S.first), t1 = P(S.last);
    const x = (t) => m.l + (t - t0) / (t1 - t0 || 1) * (width - m.l - m.r);
    const reigns = [{ start: S.first, names: S.events.length ? S.events[0].old_names : [] }];
    S.events.forEach((ev) => reigns.push({ start: ev.date, names: ev.new_names, ev }));
    reigns.forEach((r, i) => { r.end = i + 1 < reigns.length ? reigns[i + 1].start : S.last; });
    const accent = css("--accent");
    reigns.forEach((r, i) => {
      const x0 = x(P(r.start)), x1 = Math.max(x0 + 3, x(P(r.end)));
      const g = el("g", {}, svg);
      el("rect", { x: x0 + 1, y: laneY - laneH / 2, width: x1 - x0 - 2, height: laneH, rx: 5, fill: accent, opacity: i % 2 ? 0.4 : 1 }, g);
      const label = r.names.join(" + ");
      const short = label.length > 26 ? label.slice(0, 25) + "…" : label;
      if (x1 - x0 > short.length * 6.6 + 14) {
        const t = el("text", { x: x0 + 8, y: laneY + 4, class: i % 2 ? "story-label alt" : "story-label" }, g);
        t.textContent = short;
      }
      g.addEventListener("mousemove", (e) => Tip.show(`<b>#1: ${esc(label)}</b><br>${esc(r.start)} to ${esc(r.end)}`, e.clientX, e.clientY));
      g.addEventListener("mouseleave", Tip.hide);
    });
    reigns.slice(1).forEach((r) => {
      const cx = x(P(r.start)), ev = r.ev;
      const g = el("g", { tabindex: 0 }, svg);
      el("line", { x1: cx, x2: cx, y1: laneY - laneH / 2 - 12, y2: laneY + laneH / 2, stroke: css("--text"), "stroke-width": 1 }, g);
      el("circle", { cx, cy: laneY - laneH / 2 - 14, r: 10, fill: "transparent" }, g);
      el("circle", { cx, cy: laneY - laneH / 2 - 14, r: 5.5, fill: ev.departed ? css("--surface") : css("--text"),
                     stroke: css("--text"), "stroke-width": 1.5 }, g);
      const msg = ev.departed ? "The previous #1 left the board (renamed or withdrawn). Not counted." :
        ev.unresolved ? "The previous #1 was still in the new top tier: the leaderboard could not separate them." :
          "The new #1 was separated from the previous #1.";
      g.addEventListener("mousemove", (e) => Tip.show(`<b>${esc(ev.date)}</b><br>New #1: ${esc(ev.new_names.join(" + "))}<br>Previous #1: ${esc(ev.old_names.join(" + "))}<br>${msg}`, e.clientX, e.clientY));
      g.addEventListener("mouseleave", Tip.hide);
    });
    const mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const k = narrow ? 2 : 4;
    for (let j = 0; j <= k; j++) {
      const d = new Date(t0 + (t1 - t0) * j / k);
      const t = el("text", { x: m.l + (j / k) * (width - m.l - m.r), y: laneY + laneH / 2 + 22, "text-anchor": j === 0 ? "start" : j === k ? "end" : "middle", class: "fig-axis" }, svg);
      t.textContent = `${mon[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
    }
  }

  $("pick").addEventListener("change", drawPick);
  drawHero(); drawPick(); drawOverview(); drawStory();
  onResize(() => {
    if (docs.hero) { rulerFigure($("hero-figure"), docs.hero); drawRpm(); }
    if (docs.pick) rulerFigure($("pick-figure"), docs.pick);
    drawOverview(); drawStory();
  });
})();
