// app.js
// ─────────────────────────────────────────────────────────────────────
// RESPONSIBILITY: Handle all user interactions and API communication.
// Tracks conversation history so follow-up questions work correctly.
// ─────────────────────────────────────────────────────────────────────

// ── DOM REFERENCES ────────────────────────────────────────────────────
const chatWindow       = document.getElementById("chatWindow");
const questionInput    = document.getElementById("questionInput");
const sendBtn          = document.getElementById("sendBtn");
const fileInput        = document.getElementById("fileInput");
const uploadZone       = document.getElementById("uploadZone");
const uploadIdle       = document.getElementById("uploadIdle");
const uploadProcessing = document.getElementById("uploadProcessing");
const uploadDone       = document.getElementById("uploadDone");
const statusPill       = document.getElementById("statusPill");
const statusText       = document.querySelector(".status-text");
const welcomeState     = document.getElementById("welcomeState");
const suggestedQs      = document.getElementById("suggestedQuestions");
const processingStep   = document.getElementById("processingStep");


// ── STATE ─────────────────────────────────────────────────────────────
let pdfReady = false;

let conversationHistory = [];
// Stores all previous messages so LLM has memory across questions
// Format: [{role: "user", content: "..."}, {role: "assistant", content: "..."}]
// Matches OpenAI's messages format exactly — passed directly to the API


// ── UPLOAD: DRAG AND DROP ─────────────────────────────────────────────
uploadZone.addEventListener("dragover", (e) => {
e.preventDefault();
uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", () => {
uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", (e) => {
e.preventDefault();
uploadZone.classList.remove("drag-over");
const file = e.dataTransfer.files[0];
if (file) handleFile(file);
});

uploadZone.addEventListener("click", (e) => {
if (e.target === uploadZone || e.target.closest(".upload-idle")) {
fileInput.click();
}
});

fileInput.addEventListener("change", () => {
if (fileInput.files[0]) handleFile(fileInput.files[0]);
});


// ── HANDLE FILE UPLOAD ────────────────────────────────────────────────
async function handleFile(file) {
if (!file.name.endsWith(".pdf")) {
showError("Please upload a PDF file.");
return;
}

showUploadState("processing");
processingStep.textContent = "Extracting text...";

const formData = new FormData();
formData.append("file", file);
// "file" must match FastAPI parameter name: upload_pdf(file: UploadFile)

try {
// Simulate step labels for UX — real steps happen server-side
setTimeout(() => { processingStep.textContent = "Creating embeddings..."; }, 1500);
setTimeout(() => { processingStep.textContent = "Building search index..."; }, 3000);

const response = await fetch("/upload", {
    method: "POST",
    body: formData
    // No Content-Type header — browser sets it automatically for FormData
});

if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Upload failed");
}

const data = await response.json();

showUploadState("done");
document.getElementById("fileName").textContent  = file.name;
document.getElementById("chunkCount").textContent = `${data.chunks_count} chunks indexed`;

statusPill.classList.add("ready");
statusText.textContent = "Document ready";

pdfReady = true;
questionInput.disabled = false;
sendBtn.disabled = false;
questionInput.focus();

suggestedQs.style.display = "block";

} catch (err) {
showUploadState("idle");
showError(err.message);
}
}


// ── UPLOAD STATE SWITCHER ─────────────────────────────────────────────
function showUploadState(state) {
uploadIdle.style.display       = state === "idle"       ? "flex" : "none";
uploadProcessing.style.display = state === "processing" ? "flex" : "none";
uploadDone.style.display       = state === "done"       ? "flex" : "none";
}

function resetUpload() {
showUploadState("idle");
fileInput.value        = "";
pdfReady               = false;
conversationHistory    = [];
// Clear history when replacing document — new doc = fresh conversation

questionInput.disabled = true;
sendBtn.disabled       = true;
statusPill.classList.remove("ready");
statusText.textContent = "No document loaded";

chatWindow.innerHTML = "";
chatWindow.appendChild(welcomeState);
welcomeState.style.display = "block";
suggestedQs.style.display  = "none";
}


// ── KEYBOARD SHORTCUTS ────────────────────────────────────────────────
questionInput.addEventListener("keydown", (e) => {
if (e.key === "Enter" && !e.shiftKey) {
e.preventDefault();
sendQuestion();
}
});

questionInput.addEventListener("input", () => {
// Auto-resize textarea as user types
questionInput.style.height = "auto";
questionInput.style.height = questionInput.scrollHeight + "px";
});


// ── SEND QUESTION ─────────────────────────────────────────────────────
async function sendQuestion() {
const question = questionInput.value.trim();
if (!question || !pdfReady) return;

appendMessage(question, "user");
questionInput.value = "";
questionInput.style.height = "auto";
setLoading(true);

const typingEl = appendTyping();

try {
const response = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
    question,
    history: conversationHistory
    // Send full history so LLM remembers previous exchanges
    // This is what makes "more" and follow-up questions work
    })
});

typingEl.remove();

if (!response.ok) {
    const error = await response.json();
    appendMessage(error.detail || "Something went wrong.", "bot");
    return;
}

const data = await response.json();

// Add this exchange to history AFTER getting the response
conversationHistory.push({ role: "user",      content: question });
conversationHistory.push({ role: "assistant", content: data.answer });

// Keep last 12 messages (6 exchanges) — prevents prompt getting too long
// Too much history = slower responses + higher token cost
if (conversationHistory.length > 12) {
    conversationHistory = conversationHistory.slice(-12);
}

appendMessage(data.answer, "bot");

} catch (err) {
typingEl.remove();
appendMessage("Could not reach the server. Is it running?", "bot");
console.error(err);
} finally {
setLoading(false);
questionInput.focus();
}
}


// ── APPEND MESSAGE ────────────────────────────────────────────────────
function appendMessage(text, role) {
if (welcomeState.parentNode === chatWindow) {
welcomeState.style.display = "none";
}

const now = new Date().toLocaleTimeString([], {
hour: "2-digit", minute: "2-digit"
});

const wrapper = document.createElement("div");
wrapper.className = `message ${role}`;

const avatar = document.createElement("div");
avatar.className = "msg-avatar";
avatar.textContent = role === "user" ? "U" : "◈";

const content = document.createElement("div");
content.className = "msg-content";

const bubble = document.createElement("div");
bubble.className = "msg-bubble";
bubble.textContent = text;
// textContent not innerHTML — prevents XSS attacks
// Never inject raw HTML from server responses

const time = document.createElement("span");
time.className = "msg-time";
time.textContent = now;

content.appendChild(bubble);
content.appendChild(time);
wrapper.appendChild(avatar);
wrapper.appendChild(content);
chatWindow.appendChild(wrapper);

chatWindow.scrollTop = chatWindow.scrollHeight;
return wrapper;
}


// ── TYPING INDICATOR ──────────────────────────────────────────────────
function appendTyping() {
const wrapper = document.createElement("div");
wrapper.className = "message bot";

const avatar = document.createElement("div");
avatar.className = "msg-avatar";
avatar.textContent = "◈";

const content = document.createElement("div");
content.className = "msg-content";

const bubble = document.createElement("div");
bubble.className = "msg-bubble typing-bubble";

for (let i = 0; i < 3; i++) {
const dot = document.createElement("div");
dot.className = "typing-dot";
bubble.appendChild(dot);
}

content.appendChild(bubble);
wrapper.appendChild(avatar);
wrapper.appendChild(content);
chatWindow.appendChild(wrapper);
chatWindow.scrollTop = chatWindow.scrollHeight;

return wrapper;
}


// ── HELPERS ───────────────────────────────────────────────────────────
function setLoading(isLoading) {
sendBtn.disabled       = isLoading;
questionInput.disabled = isLoading;
}

function showError(message) {
appendMessage(`⚠ ${message}`, "bot");
}

function fillQuestion(btn) {
questionInput.value = btn.textContent;
questionInput.focus();
questionInput.setSelectionRange(
questionInput.value.length,
questionInput.value.length
);
}