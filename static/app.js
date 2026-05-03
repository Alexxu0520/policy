const statusEl = document.getElementById("status");
const answerEl = document.getElementById("answer");
const chunksEl = document.getElementById("chunks");

function renderResult(data) {
  answerEl.textContent = data.answer || "No answer";
  chunksEl.innerHTML = "";
  (data.chunks || []).forEach((chunk) => {
    const div = document.createElement("div");
    div.className = "chunk";
    const m = chunk.metadata || {};
    div.innerHTML = `
      <div class="meta"><b>${m.policy_id || ""}</b> | ${m.category || ""} | ${m.severity || ""} | ${m.title || ""}</div>
      <div>${chunk.text || ""}</div>
    `;
    chunksEl.appendChild(div);
  });
}

document.getElementById("auditBtn").addEventListener("click", async () => {
  const file = document.getElementById("fileInput").files[0];
  if (!file) { alert("请选择文件"); return; }
  const form = new FormData();
  form.append("file", file);
  statusEl.textContent = "审核中，首次加载 Qwen 可能需要较久...";
  answerEl.textContent = "Loading...";
  try {
    const res = await fetch("/api/audit", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Audit failed");
    renderResult(data);
    statusEl.textContent = "完成";
  } catch (err) {
    statusEl.textContent = "错误: " + err.message;
  }
});

document.getElementById("auditTextBtn").addEventListener("click", async () => {
  const text = document.getElementById("textInput").value;
  statusEl.textContent = "审核中...";
  answerEl.textContent = "Loading...";
  try {
    const res = await fetch("/api/audit-text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Audit failed");
    renderResult(data);
    statusEl.textContent = "完成";
  } catch (err) {
    statusEl.textContent = "错误: " + err.message;
  }
});
