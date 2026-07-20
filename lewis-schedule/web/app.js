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
const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatSendBtn = document.getElementById("chat-send-btn");
const chatClearBtn = document.getElementById("chat-clear-btn");
const chatPatch = document.getElementById("chat-patch");

let token = localStorage.getItem(TOKEN_KEY) || "";
let state = defaultState();
let selectedSlotKey = null;
let importThreadId = null;
let pendingPatch = null;
let pendingImage = null;
let chatThreadId = null;
let chatPendingPatch = null;
let chatBusy = false;

function defaultState(weekStart = mondayIso(new Date())) {
  return {
    week_start: weekStart,
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
      slots.push({ day, period, activity: "", caregiver: "", time: "" });
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
    await loadCurrentWeek();
    resetChat();
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

function createTimeControl(slot) {
  const wrap = document.createElement("div");
  wrap.className = "chip time-control";

  const select = document.createElement("select");
  select.className = "time-select";
  select.setAttribute("aria-label", "Activity time");

  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = "Time";
  select.appendChild(blank);

  const options =
    slot.period === "morning"
      ? [
          "07:00",
          "07:30",
          "08:00",
          "08:30",
          "09:00",
          "09:30",
          "10:00",
          "10:30",
          "11:00",
          "11:30",
        ]
      : [
          "12:00",
          "12:30",
          "13:00",
          "13:30",
          "14:00",
          "14:30",
          "15:00",
          "15:30",
          "16:00",
          "16:30",
          "17:00",
          "17:30",
          "18:00",
        ];

  for (const value of options) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = formatTimeLabel(value);
    select.appendChild(option);
  }

  if (slot.time && !options.includes(slot.time)) {
    const custom = document.createElement("option");
    custom.value = slot.time;
    custom.textContent = formatTimeLabel(slot.time);
    select.appendChild(custom);
  }

  select.value = slot.time || "";
  select.addEventListener("change", () => {
    slot.time = select.value;
    saveState();
  });
  select.addEventListener("click", (event) => event.stopPropagation());
  wrap.addEventListener("click", (event) => event.stopPropagation());
  wrap.appendChild(select);
  return wrap;
}

function formatTimeLabel(value) {
  const [hourText, minuteText] = String(value).split(":");
  const hour = Number(hourText);
  if (Number.isNaN(hour)) {
    return value;
  }
  const suffix = hour >= 12 ? "pm" : "am";
  const hour12 = hour % 12 || 12;
  return `${hour12}:${minuteText || "00"}${suffix}`;
}

function hasActivity(value) {
  const trimmed = (value || "").trim();
  return Boolean(trimmed) && trimmed.toLowerCase() !== "activity";
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
  const tempTime = source.time;
  source.activity = target.activity;
  source.caregiver = target.caregiver;
  source.time = target.time || "";
  target.activity = tempActivity;
  target.caregiver = tempCaregiver;
  target.time = tempTime || "";
  saveState();
  render();
}

async function fetchScheduleWeek(weekStart) {
  const url = `/api/schedule?week_start=${encodeURIComponent(weekStart)}`;
  const fullUrl = token
    ? `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}`
    : url;
  const res = await fetch(fullUrl, { headers: authHeaders() });
  if (res.status === 404) {
    return null;
  }
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.json();
}

async function loadWeekFromBackend(weekStart, showNotice = false) {
  state.week_start = weekStart;
  try {
    const saved = await fetchScheduleWeek(weekStart);
    if (saved) {
      applySchedule(saved, weekStart);
      if (showNotice) {
        showToast("Schedule loaded");
      }
      return;
    }
    state.caregivers = [];
    state.activities = [];
    state.slots = emptySlots();
    state.week_start = weekStart;
    saveState();
    render();
  } catch (err) {
    console.warn("Could not load schedule from backend", err);
    state.caregivers = [];
    state.activities = [];
    state.slots = emptySlots();
    state.week_start = weekStart;
    saveState();
    render();
  }
}

async function saveScheduleToBackend() {
  const payload = {
    week_start: state.week_start,
    caregivers: state.caregivers,
    activities: state.activities,
    slots: state.slots,
  };
  await apiFetch("/api/schedule", {
    method: "PUT",
    body: JSON.stringify(payload),
  });
  saveState();
  showToast("Schedule saved");
}

function applySchedule(schedule, weekStart) {
  state.week_start = weekStart;
  state.caregivers = schedule.caregivers || [];
  state.activities = schedule.activities || [];
  state.slots = (schedule.slots || emptySlots()).map((slot) => ({
    day: slot.day,
    period: slot.period,
    activity: slot.activity || "",
    caregiver: slot.caregiver || "",
    time: slot.time || "",
  }));
  if (!state.slots.length) {
    state.slots = emptySlots();
  }
  saveState();
  render();
}

function applyTemplate(template) {
  const weekStart = state.week_start;
  applySchedule(template, weekStart);
}

async function loadConfirmedTemplate(showNotice = true) {
  const template = await apiFetch("/api/template");
  applyTemplate(template);
  if (showNotice) {
    showToast("This week's schedule loaded");
  }
}

async function loadCurrentWeek() {
  await loadWeekFromBackend(mondayIso(new Date()));
}

function applyPatch(patch) {
  for (const entry of patch) {
    const slot = findSlot(entry.day, entry.period);
    if (!slot) {
      continue;
    }
    if (entry.activity !== null && entry.activity !== undefined) {
      slot.activity = entry.activity;
      if (entry.activity && !state.activities.includes(entry.activity)) {
        state.activities.push(entry.activity);
      }
      if (!hasActivity(slot.activity)) {
        slot.time = "";
      }
    }
    if (entry.caregiver !== null && entry.caregiver !== undefined) {
      slot.caregiver = entry.caregiver;
      if (entry.caregiver && !state.caregivers.includes(entry.caregiver)) {
        state.caregivers.push(entry.caregiver);
      }
    }
    if (entry.time !== null && entry.time !== undefined) {
      slot.time = hasActivity(slot.activity) ? entry.time : "";
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
      activityChip.textContent = hasActivity(slot.activity) ? slot.activity : "";
      activityChip.dataset.placeholder = "Activity";
      if (!hasActivity(slot.activity)) {
        activityChip.classList.add("empty");
      }
      activityChip.contentEditable = "true";
      activityChip.spellcheck = false;
      activityChip.addEventListener("focus", () => {
        activityChip.classList.remove("empty");
      });
      activityChip.addEventListener("blur", () => {
        const value = activityChip.textContent.trim();
        slot.activity = value;
        if (!hasActivity(value)) {
          slot.activity = "";
          slot.time = "";
          activityChip.textContent = "";
          activityChip.classList.add("empty");
        }
        saveState();
        render();
      });

      const caregiverChip = createCaregiverControl(slot);
      row.append(dayLabel, periodLabel, activityChip, caregiverChip);

      if (hasActivity(slot.activity)) {
        row.classList.add("has-activity");
        row.appendChild(createTimeControl(slot));
      }

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
      const time = (slot.time || "").trim();
      const periodLabel = PERIOD_LABELS[period];
      const timeLabel = time ? ` ${formatTimeLabel(time)}` : "";

      if (isRegularActivity(activity)) {
        lines.push(`   ${periodLabel}  ${caregiver}`);
      } else {
        lines.push(`★  ${periodLabel}${timeLabel}  ${activity} · ${caregiver}`);
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
    const answers = {};
    const questions = agent.questions;

    for (const question of questions) {
      const card = document.createElement("div");
      card.className = "question-card";
      card.dataset.questionId = question.id || question.text;

      const text = document.createElement("p");
      text.textContent = question.text;
      card.appendChild(text);

      const choices = document.createElement("div");
      choices.className = "choice-row";
      for (const choice of question.choices || []) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "choice-btn";
        btn.textContent = choice;
        btn.addEventListener("click", () => {
          answers[question.id || question.text] = {
            question: question.text,
            answer: choice,
          };
          choices.querySelectorAll(".choice-btn").forEach((el) => {
            el.classList.toggle("selected", el === btn);
          });
          updateSendEnabled();
        });
        choices.appendChild(btn);
      }
      card.appendChild(choices);
      importQuestions.appendChild(card);
    }

    const sendRow = document.createElement("div");
    sendRow.className = "question-send-row";
    const sendBtn = document.createElement("button");
    sendBtn.type = "button";
    sendBtn.className = "primary";
    sendBtn.textContent = "Send answers";
    sendBtn.disabled = true;
    sendBtn.addEventListener("click", () => {
      if (Object.keys(answers).length < questions.length) {
        return;
      }
      const lines = questions.map((question) => {
        const key = question.id || question.text;
        const picked = answers[key];
        return `${question.text} → ${picked.answer}`;
      });
      continueImport(lines.join("\n"));
    });
    sendRow.appendChild(sendBtn);
    importQuestions.appendChild(sendRow);

    function updateSendEnabled() {
      sendBtn.disabled = Object.keys(answers).length < questions.length;
    }
    return;
  }

  if (agent.mode === "proposal" && agent.patch?.length) {
    pendingPatch = agent.patch;
    importProposal.classList.remove("hidden");
    const list = document.createElement("ul");
    for (const entry of agent.patch) {
      const slot = findSlot(entry.day, entry.period);
      const before = `${slot.activity || "—"} / ${slot.caregiver || "—"}${slot.time ? ` @ ${formatTimeLabel(slot.time)}` : ""}`;
      const afterActivity =
        entry.activity === null || entry.activity === undefined
          ? slot.activity || "—"
          : entry.activity;
      const afterCaregiver =
        entry.caregiver === null || entry.caregiver === undefined
          ? slot.caregiver || "—"
          : entry.caregiver;
      const afterTime =
        entry.time === null || entry.time === undefined
          ? slot.time || ""
          : entry.time;
      const item = document.createElement("li");
      item.textContent = `${DAY_LABELS[entry.day]} ${PERIOD_LABELS[entry.period]}: ${before} → ${afterActivity} / ${afterCaregiver}${afterTime ? ` @ ${formatTimeLabel(afterTime)}` : ""}`;
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
  importStatus.textContent = "AI working its magic…";

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

function appendChatBubble(role, text) {
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${role}`;
  bubble.textContent = text;
  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function clearChatPatch() {
  chatPendingPatch = null;
  chatPatch.classList.add("hidden");
  chatPatch.innerHTML = "";
}

function renderChatPatch(patch) {
  clearChatPatch();
  if (!patch?.length) {
    return;
  }
  chatPendingPatch = patch;
  chatPatch.classList.remove("hidden");
  const title = document.createElement("strong");
  title.textContent = "Proposed changes";
  const list = document.createElement("ul");
  for (const entry of patch) {
    const slot = findSlot(entry.day, entry.period);
    const before = `${slot?.activity || "—"} / ${slot?.caregiver || "—"}${slot?.time ? ` @ ${formatTimeLabel(slot.time)}` : ""}`;
    const afterActivity =
      entry.activity === null || entry.activity === undefined
        ? slot?.activity || "—"
        : entry.activity;
    const afterCaregiver =
      entry.caregiver === null || entry.caregiver === undefined
        ? slot?.caregiver || "—"
        : entry.caregiver;
    const afterTime =
      entry.time === null || entry.time === undefined
        ? slot?.time || ""
        : entry.time;
    const item = document.createElement("li");
    item.textContent = `${DAY_LABELS[entry.day]} ${PERIOD_LABELS[entry.period]}: ${before} → ${afterActivity} / ${afterCaregiver}${afterTime ? ` @ ${formatTimeLabel(afterTime)}` : ""}`;
    list.appendChild(item);
  }
  const applyBtn = document.createElement("button");
  applyBtn.type = "button";
  applyBtn.className = "primary";
  applyBtn.textContent = "Apply changes";
  applyBtn.addEventListener("click", () => {
    if (chatPendingPatch) {
      applyPatch(chatPendingPatch);
      showToast("Schedule updated");
      clearChatPatch();
      appendChatBubble("system", "Changes applied to the grid. Tap Save to keep them on the server.");
    }
  });
  chatPatch.append(title, list, applyBtn);
}

async function sendChatMessage(message) {
  if (chatBusy || !message.trim()) {
    return;
  }
  chatBusy = true;
  chatSendBtn.disabled = true;
  chatInput.disabled = true;
  appendChatBubble("user", message.trim());
  appendChatBubble("system", "AI working its magic…");
  const thinking = chatMessages.lastElementChild;

  try {
    const payload = await apiFetch("/schedule/chat", {
      method: "POST",
      body: JSON.stringify({
        message: message.trim(),
        schedule: state,
        thread_id: chatThreadId,
      }),
    });
    chatThreadId = payload.thread_id;
    thinking.remove();
    appendChatBubble("assistant", payload.reply?.message || "No reply.");
    renderChatPatch(payload.reply?.patch || []);
  } catch (err) {
    thinking.remove();
    appendChatBubble("assistant", String(err.message || err));
  } finally {
    chatBusy = false;
    chatSendBtn.disabled = false;
    chatInput.disabled = false;
    chatInput.focus();
  }
}

async function resetChat() {
  if (chatThreadId) {
    apiFetch(`/schedule/import/${chatThreadId}`, { method: "DELETE" }).catch(() => {});
  }
  chatThreadId = null;
  clearChatPatch();
  chatMessages.innerHTML = "";
  appendChatBubble("system", "Ask about this week, or request a schedule change.");
}

connectBtn.addEventListener("click", connect);
tokenInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    connect();
  }
});

document.getElementById("prev-week-btn").addEventListener("click", async () => {
  await loadWeekFromBackend(addDays(state.week_start, -7));
});

document.getElementById("next-week-btn").addEventListener("click", async () => {
  await loadWeekFromBackend(addDays(state.week_start, 7));
});

document.getElementById("save-btn").addEventListener("click", async () => {
  try {
    await saveScheduleToBackend();
  } catch (err) {
    alert(String(err.message || err));
  }
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

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = chatInput.value;
  chatInput.value = "";
  sendChatMessage(message);
});

chatClearBtn.addEventListener("click", () => {
  resetChat();
});

if (token) {
  apiFetch("/api/health")
    .then(async () => {
      showApp();
      await loadCurrentWeek();
      resetChat();
    })
    .catch(() => localStorage.removeItem(TOKEN_KEY));
} else {
  resetChat();
}

if (!state.slots?.length) {
  state.slots = emptySlots();
}
