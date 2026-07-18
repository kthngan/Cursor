const DAYS = [
  "monday",
  "tuesday",
  "wednesday",
  "thursday",
  "friday",
  "saturday",
];
const DAY_LABELS = {
  monday: "Mon",
  tuesday: "Tue",
  wednesday: "Wed",
  thursday: "Thu",
  friday: "Fri",
  saturday: "Sat",
};
const DAY_NAMES = {
  monday: "Monday",
  tuesday: "Tuesday",
  wednesday: "Wednesday",
  thursday: "Thursday",
  friday: "Friday",
  saturday: "Saturday",
};
const PERIODS = ["morning", "afternoon"];
const PERIOD_LABELS = { morning: "AM", afternoon: "PM" };
const CAREGIVER_PRESETS = ["Por por", "Mama"];
const STORAGE_KEY = "lewisScheduleState";
const TOKEN_KEY = "lewisScheduleToken";
const TEMPLATE_VERSION_KEY = "lewisScheduleTemplateVersion";
const CURRENT_TEMPLATE_VERSION = "2026-07-14-v2-no-sunday";

const gate = document.getElementById("gate");
const app = document.getElementById("app");
const tokenInput = document.getElementById("token-input");
const connectBtn = document.getElementById("connect-btn");
const weekLabel = document.getElementById("week-label");
const scheduleGrid = document.getElementById("schedule-grid");
const screenshotInput = document.getElementById("screenshot-input");
const importDialog = document.getElementById("import-dialog");
const importPreview = document.getElementById("import-preview");
const importImage = document.getElementById("import-image");
const importStatus = document.getElementById("import-status");
const importMessage = document.getElementById("import-message");
const importQuestions = document.getElementById("import-questions");
const importProposal = document.getElementById("import-proposal");
const importApplyBtn = document.getElementById("import-apply-btn");
const importCancelBtn = document.getElementById("import-cancel-btn");
const importCloseBtn = document.getElementById("import-close-btn");
const exportDialog = document.getElementById("export-dialog");
const exportText = document.getElementById("export-text");
const exportCloseBtn = document.getElementById("export-close-btn");
const copyExportBtn = document.getElementById("copy-export-btn");
const shareExportBtn = document.getElementById("share-export-btn");
const toast = document.getElementById("toast");

let token = localStorage.getItem(TOKEN_KEY) || "";
let state = loadState();
let selectedSlotKey = null;
let importThreadId = null;
let pendingPatch = null;
let pendingImage = null;

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      return JSON.parse(raw);
    }
  } catch (_) {
    /* use default */
  }
  return {
    week_start: mondayIso(new Date()),
    caregivers: [],
    activities: [],
    slots: emptySlots(),
  };
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function mondayIso(date) {
  const d = new Date(date);
  const day = d.getDay();
  const diff = day === 0 ? -6 : 1 - day;
  d.setDate(d.getDate() + diff);
  return d.toISOString().slice(0, 10);
}

function addDays(iso, days) {
  const d = new Date(iso + "T12:00:00");
  d.setDate(d.getDate() + days);
  return d.toISOString().slice(0, 10);
}

function emptySlots() {
  const slots = [];
  for (const day of DAYS) {
    for (const period of PERIODS) {
      slots.push({ day, period, activity: "", caregiver: "" });
    }
  }
  return slots;
}

function slotKey(day, period) {
  return `${day}:${period}`;
}

function findSlot(day, period) {
  return state.slots.find((s) => s.day === day && s.period === period);
}

function authHeaders() {
  const headers = { "Content-Type": "application/json" };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

async function apiFetch(path, options = {}) {
  const url = token ? `${path}${path.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}` : path;
  const res = await fetch(url, {
    ...options,
    headers: { ...authHeaders(), ...(options.headers || {}) },
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.json();
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 2200);
}

function showApp() {
  gate.classList.add("hidden");
  app.classList.remove("hidden");
  render();
}

async function connect() {
  token = tokenInput.value.trim();
  if (!token) {
    return;
  }
  try {
    await apiFetch("/api/health");
    localStorage.setItem(TOKEN_KEY, token);
    showApp();
    await ensureTemplateLoaded();
  } catch (_) {
    alert("Invalid token or server unavailable.");
  }
}

function formatWeekLabel() {
  const start = new Date(state.week_start + "T12:00:00");
  const end = new Date(start);
  end.setDate(end.getDate() + (DAYS.length - 1));
  const fmt = new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short" });
  weekLabel.textContent = `Week of ${fmt.format(start)} – ${fmt.format(end)}`;
}

function normalizeCaregiver(value) {
  const trimmed = (value || "").trim();
  if (trimmed === "Mah mah") {
    return "Mama";
  }
  return trimmed;
}

function isPresetCaregiver(value) {
  return CAREGIVER_PRESETS.includes(normalizeCaregiver(value));
}

function createCaregiverControl(slot) {
  const wrap = document.createElement("div");
  wrap.className = "caregiver-control chip caregiver";

  const select = document.createElement("select");
  select.className = "caregiver-select";
  for (const name of [...CAREGIVER_PRESETS, "Other"]) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  }

  const otherInput = document.createElement("input");
  otherInput.type = "text";
  otherInput.className = "caregiver-other hidden";
  otherInput.placeholder = "Other name";

  const current = normalizeCaregiver(slot.caregiver);
  if (isPresetCaregiver(current)) {
    select.value = current;
  } else if (current) {
    select.value = "Other";
    otherInput.value = current;
    otherInput.classList.remove("hidden");
  } else {
    select.value = "Por por";
    slot.caregiver = "Por por";
  }

  function syncCaregiver() {
    if (select.value === "Other") {
      otherInput.classList.remove("hidden");
      slot.caregiver = otherInput.value.trim();
    } else {
      otherInput.classList.add("hidden");
      slot.caregiver = select.value;
    }
    saveState();
  }

  select.addEventListener("change", syncCaregiver);
  otherInput.addEventListener("input", syncCaregiver);
  for (const el of [select, otherInput, wrap]) {
    el.addEventListener("click", (event) => event.stopPropagation());
  }

  wrap.append(select, otherInput);
  return wrap;
}

function dateForSlot(dayIndex, period) {
  const start = new Date(state.week_start + "T12:00:00");
  start.setDate(start.getDate() + dayIndex);
  return start.toISOString().slice(0, 10);
}

function isToday(day, period) {
  const dayIndex = DAYS.indexOf(day);
  const slotDate = dateForSlot(dayIndex, period);
  const today = new Date().toISOString().slice(0, 10);
  return slotDate === today;
}

function swapSlots(sourceDay, sourcePeriod, targetDay, targetPeriod) {
  const source = findSlot(sourceDay, sourcePeriod);
  const target = findSlot(targetDay, targetPeriod);
  if (!source || !target) {
    return;
  }
  const tempActivity = source.activity;
  const tempCaregiver = source.caregiver;
  source.activity = target.activity;
  source.caregiver = target.caregiver;
  target.activity = tempActivity;
  target.caregiver = tempCaregiver;
  saveState();
  render();
}

function applyTemplate(template) {
  if (template.week_start) {
    state.week_start = template.week_start;
  }
  state.caregivers = template.caregivers || [];
  state.activities = template.activities || [];
  state.slots = template.slots.map((slot) => ({
    day: slot.day,
    period: slot.period,
    activity: slot.activity || "",
    caregiver: slot.caregiver || "",
  }));
  saveState();
  render();
}

async function loadConfirmedTemplate(showNotice = true) {
  const template = await apiFetch("/api/template");
  applyTemplate(template);
  localStorage.setItem(TEMPLATE_VERSION_KEY, CURRENT_TEMPLATE_VERSION);
  if (showNotice) {
    showToast("This week's schedule loaded");
  }
}

async function ensureTemplateLoaded() {
  if (localStorage.getItem(TEMPLATE_VERSION_KEY) === CURRENT_TEMPLATE_VERSION) {
    return;
  }
  try {
    await loadConfirmedTemplate(false);
    showToast("Confirmed schedule for this week loaded");
  } catch (err) {
    console.warn("Could not auto-load template", err);
  }
}

function applyPatch(patch) {
  for (const entry of patch) {
    const slot = findSlot(entry.day, entry.period);
    if (!slot) {
      continue;
    }
    if (entry.activity !== null && entry.activity !== undefined) {
      slot.activity = entry.activity;
      if (!state.activities.includes(entry.activity)) {
        state.activities.push(entry.activity);
      }
    }
    if (entry.caregiver !== null && entry.caregiver !== undefined) {
      slot.caregiver = entry.caregiver;
      if (!state.caregivers.includes(entry.caregiver)) {
        state.caregivers.push(entry.caregiver);
      }
    }
  }
  saveState();
  render();
}

function render() {
  formatWeekLabel();
  scheduleGrid.innerHTML = "";

  for (const day of DAYS) {
    for (const period of PERIODS) {
      const slot = findSlot(day, period);
      const row = document.createElement("article");
      row.className = "slot-row";
      row.dataset.day = day;
      row.dataset.period = period;
      row.draggable = true;

      if (isToday(day, period)) {
        row.classList.add("today");
      }

      const dayLabel = document.createElement("div");
      dayLabel.className = "day-label";
      dayLabel.textContent = period === "morning" ? DAY_LABELS[day] : "";

      const periodLabel = document.createElement("div");
      periodLabel.className = "period-label";
      periodLabel.textContent = PERIOD_LABELS[period];

      const activityChip = document.createElement("div");
      activityChip.className = "chip activity";
      activityChip.textContent = slot.activity || "Activity";
      activityChip.contentEditable = "true";
      activityChip.spellcheck = false;
      activityChip.addEventListener("blur", () => {
        slot.activity = activityChip.textContent.trim();
        saveState();
      });

      const caregiverChip = createCaregiverControl(slot);

      row.append(dayLabel, periodLabel, activityChip, caregiverChip);
      bindDragAndTap(row);
      scheduleGrid.appendChild(row);
    }
  }
}

function bindDragAndTap(row) {
  row.addEventListener("dragstart", (event) => {
    row.classList.add("dragging");
    event.dataTransfer.setData(
      "text/plain",
      JSON.stringify({ day: row.dataset.day, period: row.dataset.period })
    );
  });

  row.addEventListener("dragend", () => {
    row.classList.remove("dragging");
  });

  row.addEventListener("dragover", (event) => {
    event.preventDefault();
  });

  row.addEventListener("drop", (event) => {
    event.preventDefault();
    try {
      const source = JSON.parse(event.dataTransfer.getData("text/plain"));
      swapSlots(source.day, source.period, row.dataset.day, row.dataset.period);
    } catch (_) {
      /* ignore invalid drop */
    }
  });

  row.addEventListener("click", () => {
    const key = slotKey(row.dataset.day, row.dataset.period);
    if (!selectedSlotKey) {
      selectedSlotKey = key;
      row.classList.add("selected");
      showToast("Tap another row to swap");
      return;
    }
    if (selectedSlotKey === key) {
      selectedSlotKey = null;
      row.classList.remove("selected");
      return;
    }
    const [sourceDay, sourcePeriod] = selectedSlotKey.split(":");
    document
      .querySelectorAll(".slot-row.selected")
      .forEach((el) => el.classList.remove("selected"));
    selectedSlotKey = null;
    swapSlots(sourceDay, sourcePeriod, row.dataset.day, row.dataset.period);
  });
}

function isRegularActivity(activity) {
  const value = (activity || "").trim().toLowerCase();
  return !value || value === "regular day" || value === "free";
}

function formatExportWeekTitle() {
  const start = new Date(state.week_start + "T12:00:00");
  const fmt = new Intl.DateTimeFormat(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
  return fmt.format(start);
}

function buildExportText() {
  const lines = [`📅 Lewis — Week of ${formatExportWeekTitle()}`, ""];

  for (const day of DAYS) {
    lines.push(DAY_NAMES[day].toUpperCase());
    lines.push("────────────────");

    for (const period of PERIODS) {
      const slot = findSlot(day, period);
      const activity = (slot.activity || "").trim();
      const caregiver = (slot.caregiver || "—").trim();
      const periodLabel = PERIOD_LABELS[period];

      if (isRegularActivity(activity)) {
        lines.push(`   ${periodLabel}  ${caregiver}`);
      } else {
        lines.push(`★  ${periodLabel}  ${activity} · ${caregiver}`);
      }
    }

    lines.push("");
  }

  lines.push("—");
  lines.push("Generated from Ngan's AI");
  return lines.join("\n");
}

function resetImportDialog() {
  importThreadId = null;
  pendingPatch = null;
  importMessage.textContent = "";
  importQuestions.innerHTML = "";
  importProposal.classList.add("hidden");
  importProposal.innerHTML = "";
  importApplyBtn.classList.add("hidden");
  importStatus.textContent = "Composer will read partial updates and ask questions if needed.";
}

function renderAgentResponse(payload) {
  const agent = payload.agent;
  importMessage.textContent = agent.message || "";
  importQuestions.innerHTML = "";
  importProposal.classList.add("hidden");
  importApplyBtn.classList.add("hidden");
  pendingPatch = null;

  if (agent.mode === "questions" && agent.questions?.length) {
    for (const question of agent.questions) {
      const card = document.createElement("div");
      card.className = "question-card";
      const text = document.createElement("p");
      text.textContent = question.text;
      card.appendChild(text);

      const choices = document.createElement("div");
      choices.className = "choice-row";
      for (const choice of question.choices || []) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = choice;
        btn.addEventListener("click", () => continueImport(choice));
        choices.appendChild(btn);
      }
      card.appendChild(choices);
      importQuestions.appendChild(card);
    }
    return;
  }

  if (agent.mode === "proposal" && agent.patch?.length) {
    pendingPatch = agent.patch;
    importProposal.classList.remove("hidden");
    const list = document.createElement("ul");
    for (const entry of agent.patch) {
      const slot = findSlot(entry.day, entry.period);
      const before = `${slot.activity || "—"} / ${slot.caregiver || "—"}`;
      const afterActivity =
        entry.activity === null || entry.activity === undefined
          ? slot.activity || "—"
          : entry.activity;
      const afterCaregiver =
        entry.caregiver === null || entry.caregiver === undefined
          ? slot.caregiver || "—"
          : entry.caregiver;
      const item = document.createElement("li");
      item.textContent = `${DAY_LABELS[entry.day]} ${PERIOD_LABELS[entry.period]}: ${before} → ${afterActivity} / ${afterCaregiver}`;
      list.appendChild(item);
    }
    importProposal.appendChild(list);
    importApplyBtn.classList.remove("hidden");
    return;
  }

  if (agent.mode === "noop") {
    const card = document.createElement("div");
    card.className = "question-card";
    const input = document.createElement("input");
    input.placeholder = "Describe the change in a sentence";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "Send";
    btn.addEventListener("click", () => {
      if (input.value.trim()) {
        continueImport(input.value.trim());
      }
    });
    card.append(input, btn);
    importQuestions.appendChild(card);
  }
}

async function startImportFromFile(file) {
  resetImportDialog();
  importDialog.showModal();
  importPreview.classList.remove("hidden");
  importImage.src = URL.createObjectURL(file);
  importStatus.textContent = "Sending screenshot to Composer…";

  const base64 = await fileToBase64(file);
  pendingImage = { base64, mime_type: file.type || "image/jpeg" };

  try {
    const payload = await apiFetch("/schedule/import/start", {
      method: "POST",
      body: JSON.stringify({
        week_start: state.week_start,
        schedule: state,
        image_base64: base64,
        mime_type: pendingImage.mime_type,
      }),
    });
    importThreadId = payload.thread_id;
    importStatus.textContent = "Composer response";
    renderAgentResponse(payload);
  } catch (err) {
    importStatus.textContent = "Import failed";
    importMessage.textContent = String(err.message || err);
  }
}

async function continueImport(userMessage) {
  if (!importThreadId) {
    return;
  }
  importStatus.textContent = "Waiting for Composer…";
  importQuestions.innerHTML = "";
  try {
    const payload = await apiFetch("/schedule/import/continue", {
      method: "POST",
      body: JSON.stringify({
        thread_id: importThreadId,
        user_message: userMessage,
        schedule: state,
      }),
    });
    importStatus.textContent = "Composer response";
    renderAgentResponse(payload);
  } catch (err) {
    importStatus.textContent = "Import failed";
    importMessage.textContent = String(err.message || err);
  }
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || "");
      resolve(result.split(",")[1] || result);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

connectBtn.addEventListener("click", connect);
tokenInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    connect();
  }
});

document.getElementById("prev-week-btn").addEventListener("click", () => {
  state.week_start = addDays(state.week_start, -7);
  saveState();
  render();
});

document.getElementById("next-week-btn").addEventListener("click", () => {
  state.week_start = addDays(state.week_start, 7);
  saveState();
  render();
});

document.getElementById("reset-template-btn").addEventListener("click", async () => {
  if (!confirm("Reload the confirmed schedule for this week? This replaces your current edits.")) {
    return;
  }
  try {
    await loadConfirmedTemplate();
  } catch (err) {
    alert(String(err.message || err));
  }
});

document.getElementById("import-btn").addEventListener("click", () => {
  screenshotInput.click();
});

screenshotInput.addEventListener("change", () => {
  const file = screenshotInput.files?.[0];
  screenshotInput.value = "";
  if (file) {
    startImportFromFile(file);
  }
});

importApplyBtn.addEventListener("click", () => {
  if (pendingPatch) {
    applyPatch(pendingPatch);
    showToast("Schedule updated");
  }
  if (importThreadId) {
    apiFetch(`/schedule/import/${importThreadId}`, { method: "DELETE" }).catch(() => {});
  }
  importDialog.close();
  resetImportDialog();
});

importCancelBtn.addEventListener("click", () => {
  if (importThreadId) {
    apiFetch(`/schedule/import/${importThreadId}`, { method: "DELETE" }).catch(() => {});
  }
  importDialog.close();
  resetImportDialog();
});

importCloseBtn.addEventListener("click", () => importCancelBtn.click());
exportCloseBtn.addEventListener("click", () => exportDialog.close());

document.getElementById("export-btn").addEventListener("click", () => {
  exportText.value = buildExportText();
  exportDialog.showModal();
});

copyExportBtn.addEventListener("click", async () => {
  await navigator.clipboard.writeText(exportText.value);
  showToast("Copied — paste into WhatsApp");
});

shareExportBtn.addEventListener("click", async () => {
  const text = exportText.value;
  if (navigator.share) {
    try {
      await navigator.share({ text, title: "Lewis Schedule" });
      return;
    } catch (_) {
      /* fall through to copy */
    }
  }
  await navigator.clipboard.writeText(text);
  showToast("Copied — paste into WhatsApp");
});

if (token) {
  apiFetch("/api/health")
    .then(async () => {
      showApp();
      await ensureTemplateLoaded();
    })
    .catch(() => localStorage.removeItem(TOKEN_KEY));
}

if (!state.slots?.length) {
  state.slots = emptySlots();
}
