/* Task-pool sizing calculator: N >= (delta / eps)^2 with delta = sqrt(N) * b_N(1). */
(function () {
  "use strict";
  const { fmt, esc } = window.Site;
  const sel = document.getElementById("size-bench"), eps = document.getElementById("size-eps");
  const rows = window.SUMMARY.map((b) => ({ b, e: b.editions.live || b.editions.reference })).filter((r) => r.e);
  sel.innerHTML = rows.map((r, i) => `<option value="${i}"${r.b.key === "swebench_verified" ? " selected" : ""}>${esc(r.b.label)}</option>`).join("");
  const plan = (delta, target) => Math.ceil((delta / target) ** 2);
  function render() {
    const { b, e } = rows[Number(sel.value)];
    const target = Math.max(0.01, Number(eps.value) || 0.5);
    const delta = e.delta_pp, n = plan(delta, target);
    document.getElementById("size-n").textContent = n.toLocaleString();
    document.getElementById("size-n-note").innerHTML = `tasks for <math><mi>ε</mi><mo>=</mo><mn>${fmt(target)}</mn></math> pp, ` +
      `with <math><mover accent="true"><mi>δ</mi><mo>ˆ</mo></mover><mo>≈</mo><mn>${fmt(delta, 1)}</mn></math> pp·<math><msqrt><mtext>task</mtext></msqrt></math>`;
    document.getElementById("size-now").innerHTML = `${fmt(e.position_1_adjustment)}<span class="unit">pp</span>`;
    document.getElementById("size-now-note").innerHTML = `at <math><mi>N</mi><mo>=</mo><mn>${e.n_tasks.toLocaleString()}</mn></math> (source data of ${esc(e.date)})`;
    document.getElementById("size-table").innerHTML = [2, 1, 0.5, 0.25, 0.1].map((t) => {
      const k = plan(delta, t);
      return `<tr><td class="num">${fmt(t)}</td><td class="num">${k.toLocaleString()}</td><td class="num">${fmt(k / e.n_tasks, 1)}×</td></tr>`;
    }).join("");
  }
  sel.addEventListener("change", render);
  eps.addEventListener("input", render);
  render();
})();
