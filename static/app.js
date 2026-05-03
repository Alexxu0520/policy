// ── Tab switching ──────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(p => p.classList.add("hidden"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.remove("hidden");
  });
});

// ── File input / drag-drop ─────────────────────────────────────────────
const fileInput  = document.getElementById("fileInput");
const dropZone   = document.getElementById("dropZone");
const fileChosen = document.getElementById("fileChosen");
const fileNameEl = document.getElementById("fileName");
const auditBtn   = document.getElementById("auditBtn");

function setFile(file) {
  if (!file) return;
  fileNameEl.textContent = file.name;
  fileChosen.style.display = "flex";
  dropZone.style.display = "none";
  auditBtn.disabled = false;
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

document.getElementById("clearFile").addEventListener("click", () => {
  fileInput.value = "";
  fileChosen.style.display = "none";
  dropZone.style.display = "";
  auditBtn.disabled = true;
});

["dragover","dragenter"].forEach(e => dropZone.addEventListener(e, ev => {
  ev.preventDefault(); dropZone.classList.add("drag-over");
}));
["dragleave","drop"].forEach(e => dropZone.addEventListener(e, ev => {
  ev.preventDefault(); dropZone.classList.remove("drag-over");
}));
dropZone.addEventListener("drop", ev => {
  const f = ev.dataTransfer.files[0];
  if (f) { fileInput.files = ev.dataTransfer.files; setFile(f); }
});

// ── UI helpers ─────────────────────────────────────────────────────────
const statusBar  = document.getElementById("statusBar");
const statusText = document.getElementById("statusText");

function setStatus(msg) {
  statusBar.style.display = msg ? "flex" : "none";
  statusText.textContent = msg;
}

function fmt(n) {
  if (n === null || n === undefined) return "—";
  return Number(n).toFixed(3);
}

// ── Risk badge ─────────────────────────────────────────────────────────
function applyRisk(level, filename, chunkCount, subQueryCount) {
  const banner = document.getElementById("riskBanner");
  const badge  = document.getElementById("riskBadge");
  const file   = document.getElementById("riskFile");
  const meta   = document.getElementById("riskMeta");

  const map = {
    high: { cls: "high", label: "HIGH 高风险" },
    medium: { cls: "med", label: "MEDIUM 中风险" },
    low:  { cls: "low",  label: "LOW 低风险" },
  };
  const key = (level || "").toLowerCase();
  const entry = map[key] || { cls: "", label: level || "未知" };

  badge.className = "risk-badge " + entry.cls;
  badge.textContent = entry.label;
  file.textContent = filename || "";

  // Stats pills
  meta.innerHTML = [
    { val: chunkCount,     label: "命中政策数" },
    { val: subQueryCount,  label: "子查询数" },
  ].map(s => `
    <div class="risk-stat">
      <div class="risk-stat-val">${s.val}</div>
      <div class="risk-stat-label">${s.label}</div>
    </div>
  `).join("");

  // Banner left border colour
  const colours = { high: "#dc2626", med: "#d97706", low: "#16a34a" };
  banner.style.borderLeft = `4px solid ${colours[entry.cls] || "#e2e8f0"}`;
}

// ── Parse Qwen structured output into sections ─────────────────────────
const SECTIONS = [
  { label: "总体风险等级", icon: "🚦", re: /^\s*1[\.、\s]/m },
  { label: "主要违规点",   icon: "⚠",  re: /^\s*2[\.、\s]/m },
  { label: "命中政策 ID",  icon: "📋", re: /^\s*3[\.、\s]/m },
  { label: "修改建议",     icon: "💡", re: /^\s*4[\.、\s]/m },
  { label: "合规改写示例", icon: "✏",  re: /^\s*5[\.、\s]/m },
];

function parseAnswer(text) {
  const positions = [];
  for (const s of SECTIONS) {
    const m = s.re.exec(text);
    if (m) positions.push({ ...s, idx: m.index });
  }
  positions.sort((a, b) => a.idx - b.idx);

  const results = [];
  for (let i = 0; i < positions.length; i++) {
    const start = positions[i].idx;
    const end   = i + 1 < positions.length ? positions[i + 1].idx : text.length;
    results.push({ ...positions[i], content: text.slice(start, end).trim() });
  }
  return results.length ? results : [{ label: "审核结果", icon: "📄", content: text }];
}

function extractRiskLevel(text) {
  const m = text.match(/[Hh]igh|高风险|[Mm]edium|中风险|[Ll]ow|低风险/);
  if (!m) return null;
  const v = m[0].toLowerCase();
  if (v.includes("high") || v.includes("高")) return "high";
  if (v.includes("medium") || v.includes("中")) return "medium";
  if (v.includes("low") || v.includes("低")) return "low";
  return null;
}

// ── Render results ─────────────────────────────────────────────────────
function render(data, filename) {
  document.getElementById("resultsWrap").style.display = "block";

  // Parse risk level
  const firstSection = data.answer || "";
  const risk = extractRiskLevel(firstSection);
  const subQueries = data.sub_queries || [];
  const chunks = data.chunks || [];

  applyRisk(risk, filename, chunks.length, subQueries.length);

  // Audit sections
  const sectionsEl = document.getElementById("auditSections");
  sectionsEl.innerHTML = "";
  parseAnswer(data.answer || "").forEach(s => {
    const div = document.createElement("div");
    div.className = "audit-section";
    div.innerHTML = `
      <div class="section-heading">
        <span class="section-icon">${s.icon}</span>${s.label}
      </div>
      <div class="section-body">${escHtml(s.content)}</div>
    `;
    sectionsEl.appendChild(div);
  });

  // Sub-queries
  const sqCard = document.getElementById("subQueryCard");
  const sqList = document.getElementById("subQueries");
  sqList.innerHTML = "";
  if (subQueries.length) {
    sqCard.style.display = "";
    subQueries.forEach(q => {
      const li = document.createElement("li");
      li.textContent = q;
      sqList.appendChild(li);
    });
  } else {
    sqCard.style.display = "none";
  }

  // Policy chunks
  const chunksEl = document.getElementById("chunks");
  chunksEl.innerHTML = "";
  const list = document.createElement("div");
  list.className = "chunks-list";

  chunks.forEach((c, i) => {
    const m = c.metadata || {};
    const sev = (m.severity || "").toLowerCase();
    const sevLabel = sev === "high" ? "高危" : sev === "medium" ? "中危" : sev;

    list.innerHTML += `
      <div class="chunk-card">
        <div class="chunk-card-head">
          <span class="policy-id">${escHtml(m.policy_id || "")}</span>
          <span class="chunk-title">${escHtml(m.title || "")}</span>
          <span class="sev-tag ${sev === "high" ? "high" : sev === "medium" ? "med" : ""}">${sevLabel}</span>
        </div>
        <div class="chunk-scores">
          <span class="score-pill">综合 <b>${fmt(c.score)}</b></span>
          <span class="score-pill">Reranker <b>${fmt(c.reranker_score)}</b></span>
          <span class="score-pill">Dense <b>${fmt(c.dense_score)}</b></span>
          <span class="score-pill">BM25 <b>${fmt(c.bm25_score)}</b></span>
        </div>
        <div class="chunk-body">${escHtml(c.text || "")}</div>
        <div class="chunk-category">${escHtml(m.category || "")}</div>
      </div>
    `;
  });
  chunksEl.appendChild(list);

  // Scroll to results
  document.getElementById("resultsWrap").scrollIntoView({ behavior: "smooth", block: "start" });
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ── API calls ──────────────────────────────────────────────────────────
auditBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  setStatus("审核中，首次加载模型可能需要较久…");
  auditBtn.disabled = true;
  try {
    const res  = await fetch("/api/audit", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "审核失败");
    render(data, file.name);
    setStatus("");
  } catch (err) {
    setStatus("错误：" + err.message);
  } finally {
    auditBtn.disabled = false;
  }
});

document.getElementById("auditTextBtn").addEventListener("click", async () => {
  const text = document.getElementById("textInput").value.trim();
  if (!text) { alert("请输入待审核文本"); return; }
  const btn = document.getElementById("auditTextBtn");
  setStatus("审核中…");
  btn.disabled = true;
  try {
    const res  = await fetch("/api/audit-text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "审核失败");
    render(data, "文本输入");
    setStatus("");
  } catch (err) {
    setStatus("错误：" + err.message);
  } finally {
    btn.disabled = false;
  }
});
