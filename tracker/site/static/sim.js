/* Methods page: equally good models, noisy winner. */
(function () {
  "use strict";
  const { el, css, signed, onResize } = window.Site;
  const P = 0.6;
  const state = { j: 12, n: 200, s: 0, draws: 0, sum: 0, last: null, timer: null };
  const $ = (id) => document.getElementById(id);
  if (!$("sim")) return;

  function gauss() { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function binomMean(p, n) {
    if (n <= 120) { let k = 0; for (let i = 0; i < n; i++) if (Math.random() < p) k++; return k / n; }
    const k = Math.round(n * p + Math.sqrt(n * p * (1 - p)) * gauss());
    return Math.min(n, Math.max(0, k)) / n;
  }
  const truths = () => Array.from({ length: state.j }, (_, i) => P + (state.j === 1 ? 0 : (i / (state.j - 1) - 0.5) * state.s / 100));
  function draw() {
    const t = truths(), obs = t.map((p) => binomMean(p, state.n));
    let w = 0;
    obs.forEach((v, i) => { if (v > obs[w]) w = i; });
    const infl = 100 * (obs[w] - t[w]);
    state.draws += 1; state.sum += infl; state.last = { t, obs, w, infl };
  }
  function render() {
    const svg = $("sim-strip"), width = svg.clientWidth || 420, height = 92, m = 10;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`); svg.setAttribute("height", height);
    svg.innerHTML = "";
    const sd = Math.sqrt(P * (1 - P) / state.n);
    const lo = P - state.s / 200 - 4 * sd, hi = P + state.s / 200 + 4 * sd;
    const x = (v) => m + (v - lo) / (hi - lo) * (width - 2 * m);
    const axisY = 64;
    el("line", { x1: m, x2: width - m, y1: axisY, y2: axisY, stroke: css("--border-strong") }, svg);
    [lo + (hi - lo) * 0.1, P, hi - (hi - lo) * 0.1].forEach((v) => {
      const tx = el("text", { x: x(v), y: axisY + 18, "text-anchor": "middle" }, svg);
      tx.textContent = `${(100 * v).toFixed(0)}%` + (v === P && state.s === 0 ? " true accuracy" : "");
    });
    if (state.s === 0) el("line", { x1: x(P), x2: x(P), y1: 8, y2: axisY, stroke: css("--text-2"), "stroke-width": 1.5, "stroke-dasharray": "3 3" }, svg);
    const L = state.last;
    if (L) {
      L.obs.forEach((v, i) => {
        if (i !== L.w) el("circle", { cx: x(v), cy: axisY - 14 - (i % 3) * 9, r: 4.5, fill: css("--accent"), opacity: 0.55, stroke: css("--surface"), "stroke-width": 2 }, svg);
      });
      const wx = x(L.obs[L.w]);
      if (state.s > 0) el("line", { x1: x(L.t[L.w]), x2: wx, y1: 26, y2: 26, stroke: css("--series-2"), "stroke-width": 2 }, svg);
      el("circle", { cx: wx, cy: 26, r: 6, fill: css("--series-2"), stroke: css("--surface"), "stroke-width": 2 }, svg);
      const wl = el("text", { x: wx, y: 12, "text-anchor": wx > width - 60 ? "end" : "middle", class: "val" }, svg);
      wl.textContent = "winner";
    }
    $("sim-last").textContent = L ? `${signed(L.infl, 1)} pp` : "—";
    $("sim-avg").textContent = state.draws ? `${signed(state.sum / state.draws, 2)} pp` : "—";
    $("sim-count").textContent = state.draws.toLocaleString();
  }
  function reset() { state.draws = 0; state.sum = 0; state.last = null; if (state.timer) { clearInterval(state.timer); state.timer = null; } render(); }
  function run(k) {
    if (state.timer) clearInterval(state.timer);
    let left = k;
    state.timer = setInterval(() => {
      const batch = Math.min(left, state.n > 800 ? 4 : 10);
      for (let i = 0; i < batch; i++) draw();
      left -= batch; render();
      if (left <= 0) { clearInterval(state.timer); state.timer = null; }
    }, 30);
  }
  function bind(id, key, label) {
    const input = $(id);
    input.addEventListener("input", () => {
      state[key] = Number(input.value);
      document.querySelectorAll(`[data-out="${key}"]`).forEach((o) => { o.textContent = label(state[key]); });
      reset(); draw(); render();
    });
  }
  bind("sim-j", "j", (v) => v);
  bind("sim-n", "n", (v) => v.toLocaleString());
  bind("sim-s", "s", (v) => `${v} pp`);
  $("sim-run").addEventListener("click", () => run(300));
  $("sim-reset").addEventListener("click", reset);
  draw(); render();
  onResize(render);
})();
