/* Two planning calculators from one rule, b_N(1) ~ delta / sqrt(N), with delta = sqrt(N) * b_N(1) today:
   tasks needed for a target inflation, N >= (delta / eps)^2, and the lead a new #1 needs, delta / sqrt(N). */
(function () {
  "use strict";
  const { fmt, esc } = window.Site;
  const sel = document.getElementById("size-bench"), eps = document.getElementById("size-eps");
  const nIn = document.getElementById("margin-n");
  const $ = (id) => document.getElementById(id);
  const rows = window.SUMMARY.map((b) => ({ b, e: b.editions.live || b.editions.reference })).filter((r) => r.e);
  sel.innerHTML = rows.map((r, i) => `<option value="${i}"${r.b.key === "swebench_verified" ? " selected" : ""}>${esc(r.b.label)}</option>`).join("");
  const plan = (delta, target) => Math.ceil((delta / target) ** 2);
  const lead = (delta, n) => delta / Math.sqrt(n);
  const mi = (s) => `<math>${s}</math>`;
  const current = () => rows[Number(sel.value)];

  function basis() {
    const { e } = current();
    $("size-basis").innerHTML = `Today: inflation at #1 of ${fmt(e.position_1_adjustment)} pp with ` +
      `${mi(`<mi>N</mi><mo>=</mo><mn>${e.n_tasks.toLocaleString()}</mn>`)} tasks (source data of ${esc(e.date)}), so ` +
      `${mi(`<mover accent="true"><mi>δ</mi><mo>ˆ</mo></mover><mo>≈</mo><mn>${fmt(e.delta_pp, 1)}</mn>`)} pp·${mi("<msqrt><mtext>task</mtext></msqrt>")}.`;
  }

  function sizing() {
    const { e } = current();
    const target = Math.max(0.01, Number(eps.value) || 0.5);
    const n = plan(e.delta_pp, target);
    $("size-n").textContent = n.toLocaleString();
    $("size-n-note").innerHTML = `tasks for ${mi(`<mi>ε</mi><mo>=</mo><mn>${fmt(target)}</mn>`)} pp`;
    $("size-x").innerHTML = `${fmt(n / e.n_tasks, 1)}<span class="unit">×</span>`;
    $("size-x-note").textContent = `today's ${e.n_tasks.toLocaleString()} tasks`;
    $("size-table").innerHTML = [2, 1, 0.5, 0.25, 0.1].map((t) => {
      const k = plan(e.delta_pp, t);
      return `<tr><td class="num">${fmt(t)}</td><td class="num">${k.toLocaleString()}</td><td class="num">${fmt(k / e.n_tasks, 1)}×</td></tr>`;
    }).join("");
  }

  function margin() {
    const { e } = current();
    const n = Math.max(10, Math.round(Number(nIn.value) || e.n_tasks));
    $("margin-lead").innerHTML = `${fmt(lead(e.delta_pp, n))}<span class="unit"> pp</span>`;
    $("margin-lead-note").innerHTML = `over the previous best, with ${mi(`<mi>N</mi><mo>=</mo><mn>${n.toLocaleString()}</mn>`)} tasks`;
    const beat = e.leader_raw + lead(e.delta_pp, e.n_tasks);
    $("margin-beat").innerHTML = `${fmt(beat, 1)}<span class="unit">%</span>`;
    $("margin-beat-note").textContent = `today's top score of ${fmt(e.leader_raw, 1)}% plus ${fmt(lead(e.delta_pp, e.n_tasks))} pp, on today's ${e.n_tasks.toLocaleString()} tasks`;
    $("margin-table").innerHTML = [0.5, 1, 2, 4, 10].map((x) => {
      const k = Math.round(e.n_tasks * x);
      const today = x === 1 ? ' class="is-today"' : "";
      return `<tr${today}><td class="num">${k.toLocaleString()}</td><td class="num">${fmt(x, 1)}×</td><td class="num">${fmt(lead(e.delta_pp, k))}</td></tr>`;
    }).join("");
  }

  function render() { basis(); sizing(); margin(); }
  sel.addEventListener("change", () => { nIn.value = current().e.n_tasks; render(); });
  eps.addEventListener("input", sizing);
  nIn.addEventListener("input", margin);
  nIn.value = current().e.n_tasks;
  render();
})();
