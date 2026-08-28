// Shared helpers used across every page.

async function api(path, options = {}) {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `Request failed (${res.status})`);
  }
  return data;
}

function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function fmtDate(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString([], { year: "numeric", month: "short", day: "numeric" });
}

function statusBadge(status) {
  const map = { Present: "badge-present", Late: "badge-late", Absent: "badge-absent" };
  return `<span class="badge ${map[status] || "badge-unknown"}">${status}</span>`;
}

async function refreshEngineStatus() {
  const pill = document.getElementById("engine-status");
  if (!pill) return;
  try {
    const health = await api("/health");
    if (health.model_trained) {
      pill.className = "status-pill status-ok";
      pill.textContent = `engine ready · ${health.enrolled_identities} enrolled`;
    } else {
      pill.className = "status-pill status-unknown";
      pill.textContent = "no employees trained yet";
    }
  } catch (e) {
    pill.className = "status-pill status-bad";
    pill.textContent = "engine unreachable";
  }
}

document.addEventListener("DOMContentLoaded", refreshEngineStatus);
