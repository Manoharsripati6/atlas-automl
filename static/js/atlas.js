/* Atlas — shared client runtime */
(function () {
  const Atlas = window.Atlas = window.Atlas || {};

  // ---------- Theme ----------
  const root = document.documentElement;
  const saved = localStorage.getItem('atlas-theme') || 'dark';
  root.setAttribute('data-theme', saved);
  Atlas.toggleTheme = () => {
    const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    localStorage.setItem('atlas-theme', next);
  };

  // ---------- API layer ----------
  const json = async (r) => { if (!r.ok) throw new Error(r.statusText); return r.json().catch(() => ({})); };
  Atlas.api = {
    login:  (data)  => fetch('/login',  { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(data) }).then(json),
    signup: (data)  => fetch('/signup', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(data) }).then(json),
    upload: (file, onProgress) => {
      return new Promise((resolve, reject) => {
        const fd = new FormData(); fd.append('file', file);
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload');
        xhr.upload.onprogress = (e) => onProgress && onProgress(Math.round(e.loaded / e.total * 100));
        xhr.onload = () => xhr.status < 400 ? resolve(JSON.parse(xhr.responseText || '{}')) : reject(xhr.statusText);
        xhr.onerror = () => reject('network');
        xhr.send(fd);
      });
    },
    analyze:  ()    => fetch('/analyze', { method: 'POST' }).then(json),
    eda:      ()    => fetch('/eda').then(json),
    insights: ()    => fetch('/insights').then(json),
    viz:      ()    => fetch('/visualizations').then(json),
    ml:       ()    => fetch('/ml-results').then(json),
    report:   ()    => fetch('/report').then(json),
    profile:  (data)=> data
      ? fetch('/profile', { method: 'PUT', headers: {'Content-Type':'application/json'}, body: JSON.stringify(data) }).then(json)
      : fetch('/profile').then(json),
    billing:  ()    => fetch('/billing').then(json),
  };

  // ---------- Counter ----------
  Atlas.countUp = (el, to, dur = 1200) => {
    const start = performance.now();
    const from = 0;
    const step = (t) => {
      const p = Math.min(1, (t - start) / dur);
      const v = Math.floor(from + (to - from) * (1 - Math.pow(1 - p, 3)));
      el.textContent = v.toLocaleString();
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };

  // ---------- Typing effect ----------
  Atlas.type = (el, text, speed = 30) => {
    el.textContent = ''; let i = 0;
    const tick = () => {
      if (i <= text.length) { el.textContent = text.slice(0, i++); setTimeout(tick, speed); }
    };
    tick();
  };

  // ---------- Init on load ----------
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-theme-toggle]').forEach(b => b.addEventListener('click', Atlas.toggleTheme));
    document.querySelectorAll('[data-count]').forEach(el => Atlas.countUp(el, +el.dataset.count));
    document.querySelectorAll('[data-type]').forEach(el => Atlas.type(el, el.dataset.type));
    // Fade-up scroll trigger
    if (window.gsap && window.ScrollTrigger) {
      gsap.registerPlugin(ScrollTrigger);
      gsap.utils.toArray('[data-reveal]').forEach((el) => {
        gsap.from(el, {
          y: 24, opacity: 0, duration: .8, ease: 'power3.out',
          scrollTrigger: { trigger: el, start: 'top 85%' }
        });
      });
    }
  });
})();
