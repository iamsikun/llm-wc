/* Shared helpers: theme, menu, tooltip, formatting, sortable tables, small SVG charts. */
(function () {
  "use strict";
  const root = document.documentElement;

  // Theme: cycles to the opposite of what is shown; stored per viewer.
  const themeBtn = document.querySelector(".theme-toggle");
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const dark = root.getAttribute("data-theme") === "dark" ||
        (!root.hasAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
      const next = dark ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) { /* storage may be unavailable */ }
      document.dispatchEvent(new CustomEvent("themechange"));
    });
  }
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => document.dispatchEvent(new CustomEvent("themechange")));

  const menuBtn = document.querySelector(".menu-toggle");
  if (menuBtn) {
    menuBtn.addEventListener("click", () => {
      const nav = document.getElementById("nav");
      const open = nav.classList.toggle("open");
      menuBtn.setAttribute("aria-expanded", String(open));
    });
  }

  const tip = document.querySelector(".tooltip");
  const Tip = {
    show(html, x, y) {
      if (!tip) return;
      tip.innerHTML = html;
      tip.classList.add("show");
      const r = tip.getBoundingClientRect();
      let left = x + 14, top = y + 14;
      if (left + r.width > innerWidth - 8) left = x - r.width - 14;
      if (top + r.height > innerHeight - 8) top = y - r.height - 14;
      tip.style.left = Math.max(8, left) + "px";
      tip.style.top = Math.max(8, top) + "px";
    },
    hide() { if (tip) tip.classList.remove("show"); },
  };

  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const minus = (x) => String(x).replace(/^-/, "−");
  const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : minus(Number(v).toFixed(d));
  const signed = (v, d = 2) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
  const css = (name) => getComputedStyle(root).getPropertyValue(name).trim();

  // Sortable tables: <th data-sort="num|text">; rows carry data-v-<col> or cell text.
  function sortable(table) {
    const heads = table.querySelectorAll("th[data-sort]");
    heads.forEach((th, idx) => {
      th.tabIndex = 0;
      const run = () => {
        const tbody = table.tBodies[0];
        const col = Array.from(th.parentNode.children).indexOf(th);
        const asc = th.getAttribute("aria-sort") !== "ascending";
        heads.forEach((h) => h.removeAttribute("aria-sort"));
        th.setAttribute("aria-sort", asc ? "ascending" : "descending");
        const num = th.dataset.sort === "num";
        const rows = Array.from(tbody.rows);
        rows.sort((a, b) => {
          const ca = a.cells[col], cb = b.cells[col];
          let va = ca.dataset.v ?? ca.textContent.trim(), vb = cb.dataset.v ?? cb.textContent.trim();
          if (num) { va = parseFloat(va); vb = parseFloat(vb); if (isNaN(va)) va = -Infinity; if (isNaN(vb)) vb = -Infinity; }
          return (va < vb ? -1 : va > vb ? 1 : 0) * (asc ? 1 : -1);
        });
        rows.forEach((r) => tbody.appendChild(r));
      };
      th.addEventListener("click", run);
      th.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); run(); } });
    });
  }
  document.querySelectorAll("table.sortable").forEach(sortable);

  const NS = "http://www.w3.org/2000/svg";
  function el(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }
  // Fit a label into `room` px by cutting from the middle, so that names sharing a long prefix
  // ("Sonar Foundation Agent + Claude 4.5 Opus" / "… Sonnet") keep the part that tells them apart.
  function fitText(node, full, room) {
    node.textContent = full;
    if (!node.getComputedTextLength || node.getComputedTextLength() <= room) return;
    const fits = (s) => { node.textContent = s; return node.getComputedTextLength() <= room; };
    // Prefer dropping whole words from the middle; keep as many words as fit, balanced head/tail.
    const words = full.split(/\s+/);
    for (let n = words.length - 1; n >= 2; n--) {
      for (let h = Math.ceil(n / 2); h >= 1 && n - h <= words.length - 1; h--) {
        if (fits(`${words.slice(0, h).join(" ")} … ${words.slice(words.length - (n - h)).join(" ")}`)) return;
      }
    }
    // A single long token (e.g. "Lingxi-v1.5_claude-4-sonnet-20250514"): cut characters from the middle.
    for (let n = full.length - 1; n > 4; n--) {
      const head = Math.ceil(n * 0.55), tail = n - head;
      node.textContent = full.slice(0, head).replace(/\s+$/, "") + "…" + full.slice(full.length - tail).replace(/^\s+/, "");
      if (node.getComputedTextLength() <= room) return;
    }
  }

  // Fit a whole figure's labels at once. After fitting each on its own, labels that collide
  // ("SWE-agent…Sonnet" twice) are refit around the pieces where their full names differ.
  const pieces = (str) => str.split(/(?<=[\s_\-\/+])/);
  function fitGroup(items, room) {
    items.forEach((it) => fitText(it.node, it.full, room));
    if (!items.length || !items[0].node.getComputedTextLength) return;
    const groups = new Map();
    items.forEach((it) => { const k = it.node.textContent; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(it); });
    groups.forEach((group) => {
      if (group.length < 2) return;
      const T = group.map((it) => pieces(it.full)), minLen = Math.min(...T.map((t) => t.length));
      let p = 0;
      while (p < minLen && T.every((t) => t[p] === T[0][p])) p++;
      let q = 0;
      while (q < minLen - p && T.every((t) => t[t.length - 1 - q] === T[0][T[0].length - 1 - q])) q++;
      group.forEach((it, gi) => {
        const t = T[gi], mid = t.slice(p, t.length - q).join(""), node = it.node;
        const fits = (x) => { node.textContent = x; return node.getComputedTextLength() <= room; };
        // Most shared context first; the differing middle is always kept.
        for (let n = p + q; n >= 0; n--) {
          for (let h = Math.min(p, n); h >= Math.max(0, n - q); h--) {
            const head = t.slice(0, h).join(""), tail = t.slice(t.length - (n - h)).join("");
            const x = (h < p ? `${head.trimEnd()}…` : head) + mid + (n - h < q ? `…${tail.trimStart()}` : tail);
            if (fits(x)) return;
          }
        }
        for (let k = mid.length - 1; k > 1; k--) if (fits(`…${mid.slice(0, k).trimEnd()}…`)) return;
      });
    });
  }

  // When two agents share a display name (e.g. "GPT-5.2" for gpt-5-2 and gpt-5-2-2), the first keeps the
  // name and each later one gets its id appended, so the top of the board stays clean.
  function disambiguate(doc) {
    if (!doc || doc._named) return doc;
    const seen = {};
    doc.models.forEach((m) => {
      if (seen[m.model] && m.model_id && m.model_id !== m.model) m.model = `${m.model} (${m.model_id})`;
      else seen[m.model] = true;
    });
    doc._named = true;
    return doc;
  }

  function niceTicks(lo, hi, count = 4) {
    if (hi === lo) hi = lo + 1;
    const span = hi - lo, step0 = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || mag * 10;
    const ticks = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) ticks.push(+t.toFixed(10));
    return ticks;
  }

  // Horizontal bars, one series. items: [{label, value, href, tip}]
  function hbar(svg, items, opts = {}) {
    svg.innerHTML = "";
    const width = svg.clientWidth || 640, rowH = 30, labelW = opts.labelWidth || (width < 520 ? 128 : 170);
    const top = 6, right = 54, height = top + items.length * rowH + 26;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("height", height);
    const max = Math.max(...items.map((d) => d.value), 0) * 1.05 || 1;
    const x = (v) => labelW + (width - labelW - right) * (v / max);
    const grid = el("g", { class: "grid" }, svg);
    niceTicks(0, max, width < 520 ? 3 : 5).forEach((t) => {
      el("line", { x1: x(t), x2: x(t), y1: top, y2: top + items.length * rowH }, grid);
      const lab = el("text", { x: x(t), y: top + items.length * rowH + 18, "text-anchor": "middle" }, svg);
      lab.textContent = minus(t) + (opts.unit || "");
    });
    items.forEach((d, i) => {
      const y = top + i * rowH, bh = Math.min(16, rowH - 10), by = y + (rowH - bh) / 2;
      const g = el("g", { class: "bar-row", tabindex: 0, role: "img", "aria-label": `${d.label}: ${fmt(d.value)}${opts.unit || ""}` }, svg);
      el("rect", { x: 0, y, width, height: rowH, fill: "transparent" }, g);
      const lab = el("text", { x: labelW - 10, y: y + rowH / 2 + 4, "text-anchor": "end" }, g);
      fitText(lab, d.label, labelW - 12);
      const w = Math.max(2, x(d.value) - labelW);
      if (d.value > 0) el("path", { d: `M${labelW},${by} h${w - 4} a4,4 0 0 1 4,4 v${bh - 8} a4,4 0 0 1 -4,4 h${-(w - 4)} z`, fill: d.color || css("--accent") }, g);
      const val = el("text", { x: Math.max(labelW, x(d.value)) + 6, y: y + rowH / 2 + 4, class: "val" }, g);
      val.textContent = fmt(d.value, opts.digits ?? 2);
      const move = (e) => Tip.show(d.tip || `<b>${esc(d.label)}</b><br>${fmt(d.value)}${opts.unit || ""}`, e.clientX, e.clientY);
      g.addEventListener("mousemove", move);
      g.addEventListener("mouseleave", Tip.hide);
      if (d.href) {
        g.style.cursor = "pointer";
        g.addEventListener("click", () => { location.href = d.href; });
        g.addEventListener("keydown", (e) => { if (e.key === "Enter") location.href = d.href; });
      }
    });
    el("line", { x1: labelW, x2: labelW, y1: top, y2: top + items.length * rowH, stroke: css("--border-strong") }, svg);
  }

  // Line chart, 1-2 series on one y-axis, with crosshair tooltip.
  // series: [{name, values:[...], color}], xs: labels (same length), opts: {yUnit, xLabel, digits, step}
  function line(svg, xs, series, opts = {}) {
    svg.innerHTML = "";
    const width = svg.clientWidth || 640, height = opts.height || 240;
    const m = { l: 64, r: 16, t: 12, b: 30 };
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("height", height);
    const all = series.flatMap((s) => s.values.filter((v) => v !== null && !Number.isNaN(v)));
    let lo = Math.min(0, ...all), hi = Math.max(...all);
    if (opts.yMin !== undefined) lo = opts.yMin;
    hi = hi + (hi - lo) * 0.08;
    // Snap the domain outward to tick multiples so the lowest and highest ticks are drawn.
    const probe = niceTicks(lo, hi, 4), step = probe.length > 1 ? probe[1] - probe[0] : 1;
    lo = Math.floor(lo / step + 1e-9) * step;
    hi = Math.ceil(hi / step - 1e-9) * step;
    const n = xs.length;
    const T = opts.time, t0 = T ? Math.min(...T) : 0, t1 = T ? Math.max(...T) : 1;
    const x = T ? (i) => m.l + (t1 === t0 ? 0.5 : (T[i] - t0) / (t1 - t0)) * (width - m.l - m.r)
      : (i) => m.l + (n <= 1 ? 0.5 : i / (n - 1)) * (width - m.l - m.r);
    const y = (v) => m.t + (1 - (v - lo) / (hi - lo || 1)) * (height - m.t - m.b);
    const grid = el("g", { class: "grid" }, svg);
    niceTicks(lo, hi, 4).forEach((t) => {
      el("line", { x1: m.l, x2: width - m.r, y1: y(t), y2: y(t) }, grid);
      const lab = el("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end" }, svg);
      lab.textContent = minus(opts.digits0 ? t.toFixed(0) : t) + (opts.yUnit || "");
    });
    if (T) {
      const k = width < 360 ? 2 : 3, mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      for (let j = 0; j <= k; j++) {
        const t = t0 + (t1 - t0) * j / k, d = new Date(t);
        const lab = el("text", { x: m.l + (j / k) * (width - m.l - m.r), y: height - 8, "text-anchor": j === 0 ? "start" : j === k ? "end" : "middle" }, svg);
        lab.textContent = `${mon[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
      }
    }
    const every = T ? Infinity : Math.max(1, Math.ceil(n / (width < 520 ? 4 : 8)));
    if (!T) xs.forEach((lab, i) => {
      if (i % every !== 0 && i !== n - 1) return;
      if (i !== n - 1 && n - 1 - i < every * 0.6) return;
      const t = el("text", { x: x(i), y: height - 8, "text-anchor": i === 0 ? "start" : i === n - 1 ? "end" : "middle" }, svg);
      t.textContent = lab;
    });
    series.forEach((s) => {
      const pts = s.values.map((v, i) => (v === null ? null : [x(i), y(v)])).filter(Boolean);
      if (!pts.length) return;
      // opts.step: each value holds until the next point (snapshot data), drawn as a step-after line.
      const seg = (p, i) => (i === 0 ? `M${p[0]},${p[1]}` : opts.step ? `H${p[0]} V${p[1]}` : `L${p[0]},${p[1]}`);
      const d = pts.map(seg).join(" ");
      if (s.area) {
        el("path", { d: `${d} V${y(lo)} H${pts[0][0]} Z`, fill: s.color, opacity: 0.1 }, svg);
      }
      el("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round", ...(s.dash ? { "stroke-dasharray": "5 4" } : {}) }, svg);
      if (n <= 40) pts.forEach((p) => el("circle", { cx: p[0], cy: p[1], r: 3.2, fill: s.color, stroke: css("--surface"), "stroke-width": 1.5 }, svg));
      const last = pts[pts.length - 1];
      el("circle", { cx: last[0], cy: last[1], r: 4.5, fill: s.color, stroke: css("--surface"), "stroke-width": 2 }, svg);
    });
    const cross = el("line", { y1: m.t, y2: height - m.b, stroke: css("--border-strong"), "stroke-width": 1, visibility: "hidden" }, svg);
    const hit = el("rect", { x: m.l, y: m.t, width: width - m.l - m.r, height: height - m.t - m.b, fill: "transparent" }, svg);
    hit.addEventListener("mousemove", (e) => {
      const box = svg.getBoundingClientRect();
      const px = (e.clientX - box.left) * (width / box.width);
      let i = Math.max(0, Math.min(n - 1, Math.round(((px - m.l) / (width - m.l - m.r)) * (n - 1))));
      if (T) { let best = Infinity; for (let j = 0; j < n; j++) { const dx = Math.abs(x(j) - px); if (dx < best) { best = dx; i = j; } } }
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
      const rows = series.map((s) => s.values[i] === null ? "" : `<br><span style="color:${s.color}">●</span> ${esc(s.name)}: <b>${fmt(s.values[i], opts.digits ?? 2)}${opts.yUnit || ""}</b>`).join("");
      Tip.show(`<b>${esc(opts.labels ? opts.labels[i] : xs[i])}</b>${rows}${opts.extra ? opts.extra(i) : ""}`, e.clientX, e.clientY);
    });
    hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); Tip.hide(); });
  }

  // Vertical bars from zero, one per category (positive in the accent, negative muted), with an x-axis title.
  // opts: {yUnit, digits, xTitle, tip(i) -> html, height}
  function columns(svg, xs, values, opts = {}) {
    svg.innerHTML = "";
    const width = svg.clientWidth || 640, height = opts.height || 260;
    const m = { l: 64, r: 16, t: 12, b: 46 };
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("height", height);
    let lo = Math.min(0, ...values), hi = Math.max(0, ...values);
    const probe = niceTicks(lo, hi, 4), step = probe.length > 1 ? probe[1] - probe[0] : 1;
    lo = Math.floor(lo / step + 1e-9) * step;
    hi = Math.ceil(hi / step - 1e-9) * step;
    const n = xs.length, slot = (width - m.l - m.r) / n, gap = slot >= 6 ? 1 : 0;
    const x = (i) => m.l + i * slot;
    const y = (v) => m.t + (1 - (v - lo) / (hi - lo || 1)) * (height - m.t - m.b);
    const grid = el("g", { class: "grid" }, svg);
    niceTicks(lo, hi, 4).forEach((t) => {
      el("line", { x1: m.l, x2: width - m.r, y1: y(t), y2: y(t) }, grid);
      const lab = el("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end" }, svg);
      lab.textContent = minus(t) + (opts.yUnit || "");
    });
    const every = Math.max(1, Math.ceil(n / (width < 520 ? 4 : 8)));
    xs.forEach((lab, i) => {
      if (i % every !== 0 && i !== n - 1) return;
      if (i !== n - 1 && n - 1 - i < every * 0.6) return;
      const t = el("text", { x: x(i) + slot / 2, y: height - 26, "text-anchor": "middle" }, svg);
      t.textContent = lab;
    });
    if (opts.xTitle) {
      const t = el("text", { x: m.l + (width - m.l - m.r) / 2, y: height - 6, "text-anchor": "middle", class: "fig-axis" }, svg);
      t.textContent = opts.xTitle;
    }
    const accent = css("--accent"), muted = css("--muted");
    const bars = values.map((v, i) => el("rect", { x: x(i) + gap / 2, y: Math.min(y(v), y(0)), width: Math.max(1, slot - gap),
      height: Math.max(0.5, Math.abs(y(v) - y(0))), fill: v >= 0 ? accent : muted, opacity: v >= 0 ? 1 : 0.6 }, svg));
    el("line", { x1: m.l, x2: width - m.r, y1: y(0), y2: y(0), stroke: css("--border-strong") }, svg);
    const hit = el("rect", { x: m.l, y: m.t, width: width - m.l - m.r, height: height - m.t - m.b, fill: "transparent" }, svg);
    let on = -1;
    hit.addEventListener("mousemove", (e) => {
      const box = svg.getBoundingClientRect();
      const i = Math.max(0, Math.min(n - 1, Math.floor(((e.clientX - box.left) * (width / box.width) - m.l) / slot)));
      if (on !== i) { if (on >= 0) bars[on].removeAttribute("stroke"); bars[i].setAttribute("stroke", css("--text")); on = i; }
      Tip.show(opts.tip ? opts.tip(i) : `<b>${esc(xs[i])}</b><br>${fmt(values[i], opts.digits ?? 2)}${opts.yUnit || ""}`, e.clientX, e.clientY);
    });
    hit.addEventListener("mouseleave", () => { if (on >= 0) bars[on].removeAttribute("stroke"); on = -1; Tip.hide(); });
  }

  function onResize(fn) {
    let last = innerWidth, t;
    addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => { if (innerWidth !== last) { last = innerWidth; fn(); } }, 120); });
    document.addEventListener("themechange", fn);
  }

  // The key figure: for the top rows of a leaderboard, (1) raw -> adjusted score and
  // (2) the rank range each model falls in with 80% probability, with the top tier marked.
  function rulerFigure(svg, doc, opts = {}) {
    svg.innerHTML = "";
    const lv = opts.level || "80";
    const width = svg.clientWidth || 900, narrow = width < 640;
    const tier1 = doc.models.filter((m) => m[`tier_${lv}`] === 1).length;
    const k = Math.min(doc.n_models, opts.rows || (narrow ? Math.max(6, Math.min(10, tier1 + 2)) : Math.max(8, Math.min(18, tier1 + 3))));
    const rows = doc.models.slice(0, k);
    const labelW = narrow ? 0 : Math.min(330, Math.round(width * 0.31));
    const gap = narrow ? 18 : 36;
    const panelW = (width - labelW - gap) / 2;
    const sx0 = labelW, rx0 = labelW + panelW + gap;
    const headH = narrow ? 34 : 38, rowH = narrow ? 40 : 27, nameH = narrow ? 13 : 0, tierGap = 30;
    const showTier = tier1 < k;
    const yRow = (i) => headH + i * rowH + (showTier && i >= tier1 ? tierGap : 0);
    const bodyEnd = yRow(k - 1) + rowH;
    const showNote = tier1 && !showTier;
    const ink = css("--text"), muted = css("--muted"), accent = css("--accent"), surface = css("--surface");

    // Headers carry the two messages.
    // The legend for the marks is in the caption beneath the figure.
    const head = (x, title) => {
      const t = el("text", { x: x + panelW / 2, y: 18, "text-anchor": "middle", class: "fig-title" }, svg);
      t.textContent = title;
    };
    head(sx0, "Score adjustment");
    head(rx0, "Rank interval");
    // Arrowhead for the observed -> adjusted link; the id is per-figure since a page can hold several.
    const arrowId = `adj-arrow-${svg.id}`;
    const marker = el("marker", { id: arrowId, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 8, markerHeight: 8, markerUnits: "userSpaceOnUse", orient: "auto" }, el("defs", {}, svg));
    el("path", { d: "M0,0 L10,5 L0,10 z", fill: accent }, marker);

    // Top tier band.
    if (tier1) {
      el("rect", { x: 0, y: yRow(0) - 2, width, height: Math.min(tier1, k) * rowH + 4, rx: 8, fill: css("--accent-wash") }, svg);
    }
    if (showTier) {
      const y = yRow(tier1) - tierGap / 2;
      const t = el("text", { x: width / 2, y: y + 4, "text-anchor": "middle", class: "fig-note" }, svg);
      t.textContent = "▲ Top tier";
      const tw = t.getComputedTextLength ? t.getComputedTextLength() : 100;
      el("line", { x1: 0, x2: width / 2 - tw / 2 - 10, y1: y, y2: y, stroke: css("--border-strong"), "stroke-dasharray": "3 4" }, svg);
      el("line", { x1: width / 2 + tw / 2 + 10, x2: width, y1: y, y2: y, stroke: css("--border-strong"), "stroke-dasharray": "3 4" }, svg);
    }
    // When the whole figure is inside the top tier, say so beneath it, wrapped to the figure's width.
    let noteLines = 0;
    if (showNote) {
      const t = el("text", { x: 0, y: bodyEnd + 40, class: "fig-note" }, svg);
      let span = el("tspan", { x: 0, dy: 0 }, t), text = "";
      noteLines = 1;
      `Shaded: the top tier, ${tier1} of ${doc.n_models} agents the benchmark cannot differentiate.`.split(" ").forEach((w) => {
        const trial = text ? `${text} ${w}` : w;
        span.textContent = trial;
        if (text && span.getComputedTextLength && span.getComputedTextLength() > width) {
          span.textContent = text;
          span = el("tspan", { x: 0, dy: 16 }, t);
          span.textContent = text = w;
          noteLines += 1;
        } else text = trial;
      });
    }
    const height = bodyEnd + (showNote ? 46 + 16 * (noteLines - 1) : 30);
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("height", height);

    // Scales.
    const vals = rows.flatMap((m) => [m.raw_score_pct, m.raw_score_pct - m.adjustment_pp]);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = Math.max(0.4, (hi - lo) * 0.08);
    lo -= pad; hi += pad;
    const sx = (v) => sx0 + 6 + (v - lo) / (hi - lo) * (panelW - 12);
    const rmax = Math.max(k, ...rows.map((m) => m[`rank_high_${lv}`]));
    const rx = (r) => rx0 + 6 + (rmax === 1 ? 0.5 : (r - 1) / (rmax - 1)) * (panelW - 12);
    const axisY = bodyEnd + 4;
    niceTicks(lo, hi, narrow ? 2 : 4).filter((t) => sx(t) > sx0 + 14 && sx(t) < sx0 + panelW - 18).forEach((t) => {
      el("line", { x1: sx(t), x2: sx(t), y1: headH - 6, y2: axisY, stroke: css("--grid") }, svg);
      const lab = el("text", { x: sx(t), y: axisY + 14, "text-anchor": narrow && sx(t) < 24 ? "start" : "middle", class: "fig-axis" }, svg);
      lab.textContent = `${t}%`;
    });
    const rticks = [1, ...niceTicks(1, rmax, narrow ? 2 : 4).filter((t) => t > 1 && Number.isInteger(t))];
    rticks.forEach((t) => {
      el("line", { x1: rx(t), x2: rx(t), y1: headH - 6, y2: axisY, stroke: css("--grid") }, svg);
      const lab = el("text", { x: rx(t), y: axisY + 14, "text-anchor": narrow && rx(t) > width - 16 ? "end" : "middle", class: "fig-axis" }, svg);
      lab.textContent = t === 1 ? "#1" : `#${t}`;
    });

    const anims = [], labels = [];
    const rankW = 44; // the name starts right after the rank, so the pair reads as one label
    rows.forEach((m, i) => {
      const y = yRow(i) + nameH + (rowH - nameH) / 2;
      const g = el("g", { class: "fig-row" }, svg);
      el("title", {}, g).textContent = `${m.rank}. ${m.model}`;
      el("rect", { x: 0, y: yRow(i), width, height: rowH, fill: "transparent" }, g);
      const name = el("text", narrow ? { x: 6, y: yRow(i) + 12, class: "fig-name" } : { x: rankW, y: y + 4, class: "fig-name" }, g);
      labels.push({ node: name, full: narrow ? `${m.rank}  ${m.model}` : m.model });
      if (!narrow) {
        const r = el("text", { x: 2, y: y + 4, class: "fig-rank" }, g);
        r.textContent = m.rank;
      }
      const raw = m.raw_score_pct, adj = raw - m.adjustment_pp;
      // The link ends just short of the adjusted dot, with an arrowhead pointing at it, when there is room.
      const span = sx(raw) - sx(adj), arrow = Math.abs(span) > 16;
      const linkEnd = arrow ? sx(adj) + Math.sign(span) * 6 : sx(adj);
      const link = el("line", { x1: sx(raw), x2: linkEnd, y1: y, y2: y, stroke: accent, "stroke-width": 2, "stroke-linecap": "round", ...(arrow ? { "marker-end": `url(#${arrowId})` } : {}) }, g);
      el("circle", { cx: sx(raw), cy: y, r: 4.5, fill: surface, stroke: ink, "stroke-width": 1.6 }, g);
      const dot = el("circle", { cx: sx(adj), cy: y, r: 5, fill: accent, stroke: surface, "stroke-width": 1.5 }, g);
      anims.push([link, "x2", sx(raw), linkEnd], [dot, "cx", sx(raw), sx(adj)]);
      const l = m[`rank_low_${lv}`], h = m[`rank_high_${lv}`], t1 = m[`tier_${lv}`] === 1;
      el("line", { x1: rx0 + 6, x2: rx0 + panelW - 6, y1: y, y2: y, stroke: css("--grid"), "stroke-width": 1 }, g);
      const bar = el("line", { x1: rx(l), x2: rx(h), y1: y, y2: y, stroke: t1 ? accent : muted, "stroke-width": 8, "stroke-linecap": "round", opacity: t1 ? 1 : 0.55 }, g);
      anims.push([bar, "x1", rx(m.rank_start), rx(l)], [bar, "x2", rx(m.rank_start), rx(h)]);
      el("circle", { cx: rx(m.rank_start), cy: y, r: 3.8, fill: ink, stroke: surface, "stroke-width": 1.5 }, g);
      if (i === 0 && !narrow) {
        const down = adj <= raw;
        const dlt = el("text", { x: (sx(raw) + sx(adj)) / 2, y: y - 9, "text-anchor": "middle", class: "fig-delta" }, g);
        dlt.textContent = `${adj <= raw ? "−" : "+"}${fmt(Math.abs(raw - adj), 1)} pp`;
        const a = el("text", { x: sx(adj) + (down ? -9 : 9), y: y + 4, "text-anchor": down ? "end" : "start", class: "fig-val" }, g);
        a.textContent = `${fmt(adj, 1)}%`;
        const b = el("text", { x: sx(raw) + (down ? 9 : -9), y: y + 4, "text-anchor": down ? "start" : "end", class: "fig-val muted-val" }, g);
        b.textContent = `${fmt(raw, 1)}%`;
        const c = el("text", { x: rx(h) + 10, y: y + 4, class: "fig-val" }, g);
        c.textContent = l === h ? `#${l}` : `#${l}–${h}`;
      }
      // Score adjustment first, then ranks, then the tier, in the order of the figure.
      const tier = m[`tier_${lv}`];
      const move = (e) => Tip.show(`<b>${esc(m.model)}</b><br>` +
        `Score: ${fmt(raw, 1)}% → adjusted <b>${fmt(adj, 1)}%</b> (${adj <= raw ? "−" : "+"}${fmt(Math.abs(raw - adj), 1)} pp)<br>` +
        `Rank: observed ${esc(m.rank)} · range <b>${l === h ? `#${l}` : `#${l}–${h}`}</b><br>` +
        `Tier: ${tier === 1 ? "top tier" : `tier ${tier}`}`, e.clientX, e.clientY);
      g.addEventListener("mousemove", move);
      g.addEventListener("mouseleave", Tip.hide);
    });
    // Fit by measured width so names never run into the plot, and never collide.
    fitGroup(labels, narrow ? width - 12 : labelW - rankW - 16);
    // One-time reveal on first load: adjusted dots slide from the observed score, rank bars grow from
    // the observed rank. It explains the figure once; redraws (resize, picker, level) never animate.
    if (opts.animate && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
      const labels = svg.querySelectorAll(".fig-val, .fig-delta");
      labels.forEach((t) => t.setAttribute("opacity", 0));
      anims.forEach(([node, attr, from]) => node.setAttribute(attr, from));
      const t0 = performance.now(), dur = 600;
      const step = (now) => {
        const t = Math.min(1, Math.max(0, (now - t0) / dur)), e = 1 - Math.pow(1 - t, 4);
        anims.forEach(([node, attr, from, to]) => node.setAttribute(attr, from + (to - from) * e));
        if (t < 1) requestAnimationFrame(step);
        else labels.forEach((x) => x.setAttribute("opacity", 1));
      };
      requestAnimationFrame(step);
    }
    return k;
  }

  // Sequential blue ramp for probabilities (light: pale -> deep; dark: deep -> pale), sqrt-spaced
  // so that small but nonzero shares stay visible.
  function rampStops() {
    const t = root.getAttribute("data-theme");
    const dark = t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    return dark ? ["#1d1d1c", "#184f95", "#2a78d6", "#6da7ec", "#cde2fb"] : ["#f5f5f5", "#b7d3f6", "#5598e7", "#256abf", "#0d366b"];
  }
  const hexRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  function rampColor(stops, v) {
    const t = Math.sqrt(Math.max(0, Math.min(1, v))) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(t)), f = t - i;
    const a = hexRgb(stops[i]), b = hexRgb(stops[i + 1]);
    return a.map((c, k) => Math.round(c + (b[k] - c) * f));
  }
  const rgb = (c) => `rgb(${c.join(",")})`;
  const lum = (c) => (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;

  // Rank probability matrix: one row per system (observed order), one column per rank; each cell is
  // the bootstrap probability that the system takes that rank. Ranks past `cols` are
  // pooled into one last column, so every row still sums to 100%.
  function rpmFigure(svg, doc, opts = {}) {
    svg.innerHTML = "";
    const lv = opts.level || "80";
    const width = svg.clientWidth || 900, narrow = width < 640;
    const J = doc.n_models;
    const tier1 = doc.models.filter((m) => m[`tier_${lv}`] === 1).length;
    const k = Math.min(J, opts.rows || (narrow ? Math.max(6, Math.min(8, tier1 + 2)) : Math.max(8, Math.min(15, tier1 + 3))));
    const rows = doc.models.slice(0, k);
    const C = Math.min(J, opts.cols || Math.max(k, ...rows.map((m) => m.rank_high_99)));
    const pooled = C < J, nCols = C + (pooled ? 1 : 0);
    const labelW = narrow ? Math.round(Math.min(130, width * 0.36)) : Math.min(290, Math.round(width * 0.3));
    // The pooled column sits apart from the rank columns, and lighter, since it sums many ranks.
    const poolGap = pooled ? 10 : 0;
    const cellW = (width - labelW - 4 - poolGap) / nCols;
    const colX = (c) => labelW + c * cellW + (pooled && c === C ? poolGap : 0);
    const cellH = k > 40 ? Math.max(5, Math.min(cellW, 14)) : narrow ? 24 : 26;
    const gap = Math.min(cellW, cellH) >= 8 ? 2 : 0;
    const headH = (opts.title ? 58 : 0) + 22, tierGap = tier1 < k && cellH >= 14 ? 26 : 0;
    const yRow = (i) => headH + i * cellH + (tierGap && i >= tier1 ? tierGap : 0);
    const height = yRow(k - 1) + cellH + 6;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("height", height);
    const stops = rampStops(), ink = css("--text");
    if (opts.title) {
      const t = el("text", { x: labelW, y: 18, class: "fig-title" }, svg);
      t.textContent = opts.title;
      const u = el("text", { x: labelW, y: 38, class: "fig-sub" }, svg);
      u.textContent = opts.sub || "";
    }
    // Column labels: #1, then evenly spaced ranks, and the pooled column.
    const every = Math.max(1, Math.ceil(30 / cellW));
    for (let c = 0; c < nCols; c++) {
      const last = pooled && c === C;
      if (!last && c !== 0 && ((c + 1) % every !== 0 || (pooled && (C - c) * cellW < 34))) continue;
      const t = el("text", last ? { x: colX(c) + cellW, y: headH - 8, "text-anchor": "end", class: "fig-axis" }
        : { x: colX(c) + 0.5 * cellW, y: headH - 8, "text-anchor": "middle", class: "fig-axis" }, svg);
      t.textContent = last ? `#${C + 1}+` : c === 0 ? "#1" : `${c + 1}`;
    }
    if (tierGap) {
      const y = yRow(tier1) - tierGap / 2;
      const t = el("text", { x: width / 2, y: y + 4, "text-anchor": "middle", class: "fig-note" }, svg);
      t.textContent = "▲ Top tier";
      const tw = t.getComputedTextLength ? t.getComputedTextLength() : 100;
      el("line", { x1: 0, x2: width / 2 - tw / 2 - 10, y1: y, y2: y, stroke: css("--border-strong"), "stroke-dasharray": "3 4" }, svg);
      el("line", { x1: width / 2 + tw / 2 + 10, x2: width, y1: y, y2: y, stroke: css("--border-strong"), "stroke-dasharray": "3 4" }, svg);
    }
    const bgRgb = hexRgb(css("--bg"));
    const showVals = cellW >= 22 && cellH >= 18;
    const cells = el("g", {}, svg), labels = [];
    const nameLeft = !narrow && cellH >= 14 ? 44 : 0;
    rows.forEach((m, i) => {
      const y = yRow(i);
      if (cellH >= 14 || i % Math.ceil(14 / cellH) === 0) {
        const name = el("text", { x: nameLeft, y: y + cellH / 2 + 4, class: "fig-name" }, svg);
        labels.push({ node: name, full: m.model });
        if (!narrow && cellH >= 14) {
          const r = el("text", { x: 2, y: y + cellH / 2 + 4, class: "fig-rank" }, svg);
          r.textContent = m.rank;
        }
      }
      const row = doc.rpm[i];
      for (let c = 0; c < nCols; c++) {
        const pool = pooled && c === C;
        const v = pool ? row.slice(C).reduce((a, b) => a + b, 0) : row[c];
        const base = rampColor(stops, v);
        const col = pool ? base.map((ch, j) => Math.round(ch + (bgRgb[j] - ch) * 0.5)) : base;
        const x = colX(c);
        el("rect", { x: x + gap / 2, y: y + gap / 2, width: Math.max(1, cellW - gap), height: Math.max(1, cellH - gap),
                     rx: gap ? 3 : 0, fill: rgb(col), "data-r": i, "data-c": c, "data-v": v }, cells);
        if (showVals && v >= 0.095) {
          const t = el("text", { x: x + cellW / 2, y: y + cellH / 2 + 4, "text-anchor": "middle", class: "rpm-cell-val",
                                 fill: lum(col) < 0.5 ? "#ffffff" : "#171717" }, cells);
          t.textContent = Math.round(100 * v);
        }
      }
      // Observed rank; tied systems share the span of ranks they occupy.
      const obs = m.rank_start - 1, span = Math.min(m.rank_end, C) - m.rank_start + 1;
      if (obs < C && cellW >= 8) {
        el("rect", { x: colX(obs) + 0.75, y: y + 0.75, width: span * cellW - 1.5, height: cellH - 1.5, rx: 4,
                     fill: "none", stroke: ink, "stroke-width": 1.5, "pointer-events": "none" }, svg);
      }
    });
    fitGroup(labels, labelW - nameLeft - 12);
    labels.forEach(({ node, full }) => { el("title", {}, node).textContent = full; });
    cells.addEventListener("mousemove", (e) => {
      const r = e.target.dataset && e.target.dataset.r;
      if (r === undefined) return Tip.hide();
      const m = rows[Number(r)], c = Number(e.target.dataset.c), v = Number(e.target.dataset.v);
      const where = pooled && c === C ? `rank #${C + 1} or lower` : `rank #${c + 1}`;
      Tip.show(`<b>${esc(m.model)}</b><br>Probability of ${where}: <b>${(100 * v).toFixed(1)}%</b><br>` +
        `Observed rank ${esc(m.rank)} · rank range #${m[`rank_low_${lv}`]}–${m[`rank_high_${lv}`]}`, e.clientX, e.clientY);
    });
    cells.addEventListener("mouseleave", Tip.hide);
    return { k, cols: C, pooled, values: showVals };
  }

  // HTML legend for rpmFigure: the color ramp and the observed-rank outline.
  function rpmLegend(node, values = true, pooled = false) {
    const stops = rampStops();
    const grad = [0, 0.01, 0.05, 0.15, 0.3, 0.5, 0.75, 1].map((v) => rgb(rampColor(stops, v))).join(",");
    node.innerHTML = `<span class="ramp">0%<span class="bar" style="background:linear-gradient(90deg,${grad})"></span>100%</span>` +
      `<span><span class="obs"></span>observed rank</span><span>${values ? "Numbers are probabilities in %. " : ""}${matchMedia("(hover: none)").matches ? "Tap" : "Hover"} a cell for its value.${pooled ? " The last column, set apart and fainter, pools all lower ranks." : ""}</span>`;
  }

  // Save an on-page SVG as a PNG: inline the computed text styles, paint the surface, scale 2x.
  function downloadSvg(svg, filename) {
    const clone = svg.cloneNode(true);
    const src = svg.querySelectorAll("text"), dst = clone.querySelectorAll("text");
    src.forEach((t, i) => {
      const cs = getComputedStyle(t);
      dst[i].setAttribute("style", `font-family:${cs.fontFamily};font-size:${cs.fontSize};font-weight:${cs.fontWeight};fill:${cs.fill}`);
    });
    const vb = svg.viewBox.baseVal, w = vb.width, h = vb.height, pad = 24;
    clone.setAttribute("xmlns", NS);
    clone.setAttribute("width", w); clone.setAttribute("height", h);
    const data = new XMLSerializer().serializeToString(clone);
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas"), s = 2;
      c.width = (w + 2 * pad) * s; c.height = (h + 2 * pad) * s;
      const ctx = c.getContext("2d");
      ctx.fillStyle = css("--surface"); ctx.fillRect(0, 0, c.width, c.height);
      ctx.drawImage(img, pad * s, pad * s, w * s, h * s);
      c.toBlob((blob) => {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob); a.download = filename; a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 1000);
      });
    };
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(data);
  }
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-png]");
    if (btn) downloadSvg(document.getElementById(btn.dataset.png), `${btn.dataset.name || "figure"}.png`);
  });

  // Copy buttons: <button data-copy="#id"> copies that element's text.
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-copy]");
    if (!btn) return;
    const src = document.querySelector(btn.dataset.copy);
    if (!src || !navigator.clipboard) return;
    navigator.clipboard.writeText(src.textContent.trim()).then(() => {
      const was = btn.textContent;
      btn.textContent = "Copied";
      setTimeout(() => { btn.textContent = was; }, 1400);
    }).catch(() => {});
  });

  window.Site = { Tip, esc, fmt, signed, css, sortable, el, niceTicks, fitText, fitGroup, disambiguate, hbar, line, columns, onResize, rulerFigure, rpmFigure, rpmLegend, downloadSvg };
})();

// Click counting: links with data-track="<event>" are sent to GoatCounter as events when it is
// configured (site.goatcounter). One delegated listener also covers links added by scripts.
document.addEventListener("click", (e) => {
  const a = e.target.closest && e.target.closest("[data-track]");
  const gc = window.goatcounter;
  if (!a || !gc || typeof gc.count !== "function") return;
  gc.count({ path: a.dataset.track, title: a.textContent.trim().slice(0, 80), event: true });
});
