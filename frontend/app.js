// app.js
// ─────────────────────────────────────────────────────────────────────
// RESPONSIBILITY: Handle all user interactions and API communication.
// No page reloads — everything is dynamic via fetch() API.
// ─────────────────────────────────────────────────────────────────────

// ── DOM REFERENCES ────────────────────────────────────────────────────
const chatWindow      = document.getElementById("chatWindow");
const questionInput   = document.getElementById("questionInput");
const sendBtn         = document.getElementById("sendBtn");
const fileInput       = document.getElementById("fileInput");
const uploadZone      = document.getElementById("uploadZone");
const uploadIdle      = document.getElementById("uploadIdle");
const uploadProcessing = document.getElementById("uploadProcessing");
const uploadDone      = document.getElementById("uploadDone");
const statusPill      = document.getElementById("statusPill");
const statusText      = document.querySelector(".status-text");
const welcomeState    = document.getElementById("welcomeState");
const suggestedQs     = document.getElementById("suggestedQuestions");
const processingStep  = document.getElementById("processingStep");


// ── STATE ─────────────────────────────────────────────────────────────
// Track whether a PDF has been loaded so we can enable/disable chat
let pdfReady = false;


// ── UPLOAD: DRAG AND DROP ─────────────────────────────────────────────
// Drag over — visual feedback
uploadZone.addEventListener("dragover", (e) => {
e.preventDefault();
// preventDefault() stops browser from opening the file
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

// Click to browse
uploadZone.addEventListener("click", (e) => {
// Only trigger if clicking the zone itself, not the browse button
// The browse button has its own onclick handler
if (e.target === uploadZone || e.target.closest(".upload-idle")) {
fileInput.click();
}
});

fileInput.addEventListener("change", () => {
if (fileInput.files[0]) handleFile(fileInput.files[0]);
});


// ── HANDLE FILE UPLOAD ────────────────────────────────────────────────
async function handleFile(file) {
// Validate — only PDFs
if (!file.name.endsWith(".pdf")) {
showError("Please upload a PDF file.");
return;
}

// Show processing state
showUploadState("processing");
processingStep.textContent = "Extracting text...";

// Build form data — multipart/form-data for file uploads
// This is what the browser sends when you submit an HTML form with a file
const formData = new FormData();
formData.append("file", file);
// "file" must match the parameter name in FastAPI: upload_pdf(file: UploadFile)

try {
// Simulate step progress for UX — shows user something is happening
// Real steps happen server-side but we give visual feedback
setTimeout(() => { processingStep.textContent = "Creating embeddings..."; }, 1500);
setTimeout(() => { processingStep.textContent = "Building search index..."; }, 3000);

const response = await fetch("/upload", {
    method: "POST",
    body: formData
    // No Content-Type header — browser sets it automatically for FormData
    // including the boundary string that separates fields
});

if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Upload failed");
}

const data = await response.json();
// data.chunks_count = number of chunks created (from our FastAPI response)
// data.message = success message

// Show done state
showUploadState("done");
document.getElementById("fileName").textContent = file.name;
document.getElementById("chunkCount").textContent = `${data.chunks_count} chunks indexed`;

// Update status pill
statusPill.classList.add("ready");
statusText.textContent = "Document ready";

// Enable chat
pdfReady = true;
questionInput.disabled = false;
sendBtn.disabled = false;
questionInput.focus();

// Show suggested questions
suggestedQs.style.display = "block";

} catch (err) {
showUploadState("idle");
showError(err.message);
}
}


// ── UPLOAD STATE SWITCHER ─────────────────────────────────────────────
function showUploadState(state) {
// Hide all states then show the right one
// Clean pattern — no risk of two states showing at once
uploadIdle.style.display       = state === "idle"       ? "flex" : "none";
uploadProcessing.style.display = state === "processing" ? "flex" : "none";
uploadDone.style.display       = state === "done"       ? "flex" : "none";
}

function resetUpload() {
// Let user upload a new document
showUploadState("idle");
fileInput.value = "";
pdfReady = false;
questionInput.disabled = true;
sendBtn.disabled = true;
statusPill.classList.remove("ready");
statusText.textContent = "No document loaded";

// Clear chat
chatWindow.innerHTML = "";
chatWindow.appendChild(welcomeState);
welcomeState.style.display = "block";
suggestedQs.style.display = "none";
}


// ── KEYBOARD SHORTCUTS ────────────────────────────────────────────────
questionInput.addEventListener("keydown", (e) => {
if (e.key === "Enter" && !e.shiftKey) {
// Enter = send, Shift+Enter = new line (standard chat behavior)
e.preventDefault();
sendQuestion();
}
});

// Auto-resize textarea as user types
// This makes the input grow naturally instead of scrolling inside
questionInput.addEventListener("input", () => {
questionInput.style.height = "auto";
questionInput.style.height = questionInput.scrollHeight + "px";
});


// ── SEND QUESTION ─────────────────────────────────────────────────────
async function sendQuestion() {
const question = questionInput.value.trim();

// Guard — don't send empty questions or if no PDF loaded
if (!question || !pdfReady) return;

// Show user message immediately — don't wait for server
// This makes the UI feel instant and responsive
appendMessage(question, "user");

// Clear and reset input
questionInput.value = "";
questionInput.style.height = "auto";
setLoading(true);

// Show typing indicator — gives visual feedback while waiting
const typingEl = appendTyping();

try {
const response = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // JSON.stringify converts JS object to JSON string
    // FastAPI's Pydantic model reads this and validates it
    body: JSON.stringify({ question })
});

// Remove typing indicator before showing answer
typingEl.remove();

if (!response.ok) {
    const error = await response.json();
    appendMessage(error.detail || "Something went wrong.", "bot");
    return;
}

const data = await response.json();
// data.answer = the LLM's generated answer
// data.question = echoed back (from our AnswerResponse model)

appendMessage(data.answer, "bot");

} catch (err) {
typingEl.remove();
appendMessage("Could not reach the server. Is it running?", "bot");
console.error(err);
} finally {
// finally always runs — re-enable input whether success or error
setLoading(false);
questionInput.focus();
}
}


// ── APPEND MESSAGE ────────────────────────────────────────────────────
function appendMessage(text, role) {
// Hide welcome state on first message
if (welcomeState.parentNode === chatWindow) {
welcomeState.style.display = "none";
}

const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

const wrapper = document.createElement("div");
wrapper.className = `message ${role}`;

// Avatar — "U" for user, ◈ for bot
const avatar = document.createElement("div");
avatar.className = "msg-avatar";
avatar.textContent = role === "user" ? "U" : "◈";

// Content wrapper
const content = document.createElement("div");
content.className = "msg-content";

// Bubble
const bubble = document.createElement("div");
bubble.className = "msg-bubble";
bubble.textContent = text;
// textContent (not innerHTML) — never inject raw HTML from server
// This prevents XSS (cross-site scripting) attacks

// Timestamp
const time = document.createElement("span");
time.className = "msg-time";
time.textContent = now;

content.appendChild(bubble);
content.appendChild(time);
wrapper.appendChild(avatar);
wrapper.appendChild(content);
chatWindow.appendChild(wrapper);

// Scroll to latest message
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

// Three animated dots — CSS handles the animation
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
// Disable both input and button together — they're one unit
sendBtn.disabled       = isLoading;
questionInput.disabled = isLoading;
}

function showError(message) {
// Flash a bot message with the error
appendMessage(`⚠ ${message}`, "bot");
}

function fillQuestion(btn) {
// Fill input with suggested question text
questionInput.value = btn.textContent;
questionInput.focus();
// Move cursor to end
questionInput.setSelectionRange(
questionInput.value.length,
questionInput.value.length
);
}