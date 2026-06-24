/* Atlas — Animated analysis pipeline */
window.AtlasPipeline = function (root, { onComplete } = {}) {
  const nodes = [...root.querySelectorAll('.pipeline-node')];
  const bar = root.querySelector('[data-pipeline-bar]');
  const pct = root.querySelector('[data-pipeline-pct]');
  let i = 0;
  function step() {
    if (i >= nodes.length) { onComplete && onComplete(); return; }
    if (i > 0) nodes[i-1].classList.remove('active'), nodes[i-1].classList.add('done');
    nodes[i].classList.add('active');
    const p = Math.round(((i + 1) / nodes.length) * 100);
    if (bar) bar.style.width = p + '%';
    if (pct) pct.textContent = p + '%';
    i++;
    setTimeout(step, 1200);
  }
  step();
};
