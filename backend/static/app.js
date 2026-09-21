// SEEPSENSE dashboard behaviour: data fetch/render, scroll reveal,
// cursor-reactive hero blobs, and magnetic tilt on node cards.
// Every animation here only touches transform/opacity (never layout
// properties), per the project's performance guardrail.

const REFRESH_MS = 15000;
let lastUpdatedAt = null;

// ---------- cursor-following glow on stat tiles ----------
function initTileGlow() {
  document.querySelectorAll(".stat-tile").forEach((tile) => {
    tile.addEventListener("mousemove", (e) => {
      const rect = tile.getBoundingClientRect();
      tile.style.setProperty("--mx", `${e.clientX - rect.left}px`);
      tile.style.setProperty("--my", `${e.clientY - rect.top}px`);
    });
  });
}

// ---------- button ripple feedback ----------
function initRipples() {
  document.querySelectorAll(".btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const rect = btn.getBoundingClientRect();
      const size = Math.max(rect.width, rect.height) * 1.4;
      const ripple = document.createElement("span");
      ripple.className = "ripple";
      ripple.style.width = ripple.style.height = `${size}px`;
      ripple.style.left = `${e.clientX - rect.left - size / 2}px`;
      ripple.style.top = `${e.clientY - rect.top - size / 2}px`;
      btn.appendChild(ripple);
      ripple.addEventListener("animationend", () => ripple.remove());
    });
  });
}

// ---------- animated number count-up ----------
function animateNumber(el, target, { decimals = 0, duration = 700, suffix = "" } = {}) {
  const start = Number(el.dataset.value || 0);
  if (start === target) return;
  el.dataset.value = target;
  const startTime = performance.now();

  function tick(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // ease-out-cubic
    const value = start + (target - start) * eased;
    el.textContent = `${value.toFixed(decimals)}${suffix}`;
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ---------- "updated Xs ago" ticker ----------
function tickLastUpdated() {
  const el = document.getElementById("last-updated");
  if (!el || lastUpdatedAt === null) return;
  const seconds = Math.round((Date.now() - lastUpdatedAt) / 1000);
  el.textContent = seconds < 2 ? "updated just now" : `updated ${seconds}s ago`;
}

// ---------- per-node sparkline ----------
async function renderNodeSparkline(container, nodeId) {
  try {
    const res = await fetch(`/readings?node_id=${encodeURIComponent(nodeId)}&limit=200`);
    if (!res.ok) return;
    const data = await res.json();
    const values = data.readings.map((r) => r.soil_raw).slice(-20);
    if (values.length < 2) return;

    const w = 240, h = 34;
    const min = Math.min(...values), max = Math.max(...values);
    const span = max - min || 1;
    const points = values.map((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = h - ((v - min) / span) * h;
      return [x, y];
    });
    const linePath = points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
    const fillPath = `${linePath} L${w},${h} L0,${h} Z`;
    // Each card gets its own gradient id: SVG <defs> ids must be unique
    // per-document, and several cards render this markup on the same page.
    const gradientId = `sparkGradient-${nodeId}`;

    container.innerHTML = `
      <svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
        <defs>
          <linearGradient id="${gradientId}" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="var(--water)" stop-opacity="0.35" />
            <stop offset="100%" stop-color="var(--water)" stop-opacity="0" />
          </linearGradient>
        </defs>
        <path class="spark-fill" style="fill: url(#${gradientId})" d="${fillPath}"></path>
        <path class="spark-line" d="${linePath}"></path>
      </svg>
    `;
  } catch (err) {
    // Sparkline is a nice-to-have; a fetch failure here shouldn't break the card.
  }
}

// ---------- scroll-triggered reveals ----------
function initScrollReveal() {
  const targets = document.querySelectorAll(".reveal");
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      }
    },
    { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
  );
  targets.forEach((el) => observer.observe(el));
}

// ---------- hero cursor parallax ----------
function initHeroParallax() {
  const field = document.getElementById("hero-field");
  const hero = document.querySelector(".hero");
  const blobA = field.querySelector(".blob-a");
  const blobB = field.querySelector(".blob-b");
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  hero.addEventListener("mousemove", (e) => {
    const rect = hero.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / rect.width - 0.5; // -0.5..0.5
    const ny = (e.clientY - rect.top) / rect.height - 0.5;
    blobA.style.transform = `translate3d(${nx * 40}px, ${ny * 40}px, 0)`;
    blobB.style.transform = `translate3d(${nx * -30}px, ${ny * -30}px, 0)`;
  });
  hero.addEventListener("mouseleave", () => {
    blobA.style.transform = "translate3d(0,0,0)";
    blobB.style.transform = "translate3d(0,0,0)";
  });
}

// ---------- magnetic tilt on cards ----------
function attachTilt(card) {
  const maxTilt = 6; // degrees
  card.addEventListener("mousemove", (e) => {
    const rect = card.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width - 0.5;
    const py = (e.clientY - rect.top) / rect.height - 0.5;
    card.style.transform = `perspective(600px) rotateX(${(-py * maxTilt).toFixed(2)}deg) rotateY(${(px * maxTilt).toFixed(2)}deg) translateZ(0)`;
  });
  card.addEventListener("mouseleave", () => {
    card.style.transform = "perspective(600px) rotateX(0deg) rotateY(0deg)";
  });
}

// ---------- helpers ----------
function riskClass(prob) {
  if (prob === null || prob === undefined) return "risk-unknown";
  if (prob >= 0.75) return "risk-high";
  if (prob >= 0.4) return "risk-medium";
  return "risk-low";
}
function riskLabel(prob) {
  if (prob === null || prob === undefined) return "no data";
  return `${Math.round(prob * 100)}% risk`;
}
function fmt(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
}

// ---------- render: node cards ----------
function renderNodes(nodes) {
  const grid = document.getElementById("node-grid");
  grid.innerHTML = "";

  nodes.forEach((node) => {
    const card = document.createElement("div");
    card.className = "node-card";

    const reading = node.latest_reading;
    const metricsHtml = reading
      ? `
        <div class="node-metrics">
          <div><span class="node-metric-label">soil_raw</span><span class="node-metric-value">${fmt(reading.soil_raw, 0)}</span></div>
          <div><span class="node-metric-label">temp</span><span class="node-metric-value">${fmt(reading.temp_c)}°C</span></div>
          <div><span class="node-metric-label">humidity</span><span class="node-metric-value">${fmt(reading.humidity_pct)}%</span></div>
          <div><span class="node-metric-label">pressure</span><span class="node-metric-value">${fmt(reading.pressure_hpa)}hPa</span></div>
        </div>`
      : `<p class="node-empty">No readings ingested yet.</p>`;

    card.innerHTML = `
      <div class="node-card-head">
        <div>
          <div class="node-id">${node.node_id}</div>
          <div class="node-desc">${node.description ?? ""}</div>
        </div>
        <span class="risk-badge ${riskClass(node.leak_probability)}">${riskLabel(node.leak_probability)}</span>
      </div>
      ${metricsHtml}
      <div class="node-sparkline" id="spark-${node.node_id}"></div>
    `;
    attachTilt(card);
    grid.appendChild(card);

    if (reading) {
      renderNodeSparkline(card.querySelector(`#spark-${node.node_id}`), node.node_id);
    }
  });
}

// ---------- render: model performance ----------
function renderModel(model) {
  const grid = document.getElementById("model-grid");
  if (!model) return; // keep the static "no model yet" note

  grid.innerHTML = "";
  const names = Object.keys(model).filter((k) => k !== "best_model");

  names.forEach((name) => {
    const m = model[name];
    const isBest = model.best_model === name;
    const card = document.createElement("div");
    card.className = "model-card" + (isBest ? " best" : "");

    const bars = [
      ["precision", m.precision],
      ["recall", m.recall],
      ["f1", m.f1],
      ["roc_auc", m.roc_auc],
    ];

    card.innerHTML = `
      <div class="model-card-title">
        <span>${name.replace("_", " ")}</span>
        ${isBest ? '<span class="best-tag">selected</span>' : ""}
      </div>
      ${bars
        .map(
          ([label, value]) => `
        <div class="bar-row">
          <div class="bar-row-label"><span>${label}</span><span>${fmt(value, 3)}</span></div>
          <div class="bar-track"><div class="bar-fill" data-width="${(value || 0) * 100}"></div></div>
        </div>`
        )
        .join("")}
      ${
        "rain_false_positive_rate" in m
          ? `<div class="rain-fp"><span>rain false-positive rate</span><span>${(m.rain_false_positive_rate * 100).toFixed(1)}%</span></div>`
          : ""
      }
    `;
    grid.appendChild(card);
  });

  // Animate bar widths on next frame so the CSS transition actually plays.
  requestAnimationFrame(() => {
    grid.querySelectorAll(".bar-fill").forEach((el) => {
      el.style.width = `${el.dataset.width}%`;
    });
  });
}

// ---------- render: LeakDB real-data validation ----------
function renderLeakdb(leakdb) {
  const grid = document.getElementById("leakdb-grid");
  if (!leakdb) return; // keep the static "not run yet" note

  const bars = [
    ["precision", leakdb.precision],
    ["recall", leakdb.recall],
    ["f1", leakdb.f1],
    ["roc_auc", leakdb.roc_auc],
  ];

  grid.innerHTML = `
    <div class="model-card">
      <div class="model-card-title">
        <span>RandomForest on ${leakdb.dataset.split(" (")[0]}</span>
        <span class="best-tag">real data</span>
      </div>
      ${bars
        .map(
          ([label, value]) => `
        <div class="bar-row">
          <div class="bar-row-label"><span>${label}</span><span>${fmt(value, 3)}</span></div>
          <div class="bar-track"><div class="bar-fill" data-width="${(value || 0) * 100}"></div></div>
        </div>`
        )
        .join("")}
      <div class="rain-fp">
        <span>trained on scenarios ${leakdb.train_scenarios.join(", ")}</span>
        <span>tested on ${leakdb.test_scenarios.join(", ")}</span>
      </div>
    </div>
  `;
  requestAnimationFrame(() => {
    grid.querySelectorAll(".bar-fill").forEach((el) => (el.style.width = `${el.dataset.width}%`));
  });
}

// ---------- risk ring ----------
const RING_CIRCUMFERENCE = 327; // 2 * PI * r, r=52 (matches the SVG in index.html)

function updateRiskRing(prob) {
  const ring = document.getElementById("risk-ring");
  const fraction = prob === null || prob === undefined ? 0 : prob;
  const offset = RING_CIRCUMFERENCE * (1 - fraction);
  ring.style.strokeDashoffset = String(offset);
  ring.style.stroke = fraction >= 0.75 ? "var(--clay)" : fraction >= 0.4 ? "var(--gold)" : "var(--leaf)";
}

// ---------- sidebar scrollspy ----------
function initScrollspy() {
  const links = Array.from(document.querySelectorAll(".sidenav-link"));
  const sections = links
    .map((link) => document.getElementById(link.getAttribute("href").slice(1)))
    .filter(Boolean);

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const id = entry.target.id;
        links.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === `#${id}`));
      }
    },
    { rootMargin: "-20% 0px -70% 0px" }
  );
  sections.forEach((section) => observer.observe(section));
}

// ---------- main data load ----------
async function loadDashboard() {
  const statusPill = document.getElementById("status-pill");
  const statusText = document.getElementById("status-text");
  try {
    const res = await fetch("/api/dashboard");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    statusPill.classList.add("live");
    statusText.textContent = "live";

    animateNumber(document.getElementById("stat-readings"), data.reading_count);
    animateNumber(document.getElementById("stat-nodes"), data.nodes.length);
    document.getElementById("stat-model").textContent = data.model ? data.model.best_model.replace("_", " ") : "not trained";

    const probs = data.nodes.map((n) => n.leak_probability).filter((p) => p !== null && p !== undefined);
    const peakRisk = probs.length ? Math.max(...probs) : null;
    const riskEl = document.getElementById("stat-risk");
    if (peakRisk !== null) {
      animateNumber(riskEl, Math.round(peakRisk * 100), { suffix: "%" });
    } else {
      riskEl.textContent = "—";
    }
    updateRiskRing(peakRisk);

    renderNodes(data.nodes);
    renderModel(data.model);
    renderLeakdb(data.leakdb);

    lastUpdatedAt = Date.now();
    tickLastUpdated();
  } catch (err) {
    statusPill.classList.remove("live");
    statusText.textContent = "offline";
    console.error("dashboard load failed", err);
  }
}

// ---------- boot ----------
document.addEventListener("DOMContentLoaded", () => {
  initScrollReveal();
  initHeroParallax();
  initScrollspy();
  initTileGlow();
  initRipples();
  loadDashboard();
  setInterval(loadDashboard, REFRESH_MS);
  setInterval(tickLastUpdated, 1000);
});
