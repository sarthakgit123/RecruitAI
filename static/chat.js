const transcript = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");
const resetBtn = document.getElementById("reset-btn");

const WELCOME_HTML = `
  <div class="entry entry--bot">
    <span class="entry__tag">SUBJECT FILE</span>
    <div class="entry__bubble">
      Ask me anything about the uploaded resumes — for example
      “who knows Python”, “who has 2+ years experience”, or
      “who would be a good fit for a backend role”.
    </div>
  </div>
`;

function addEntry(text, sender, options = {}) {
  const entry = document.createElement("div");
  entry.className = `entry entry--${sender}`;

  const tag = document.createElement("span");
  tag.className = "entry__tag";
  tag.textContent = sender === "user" ? "YOU ASK" : "SUBJECT FILE";

  const bubble = document.createElement("div");
  bubble.className = "entry__bubble";
  if (options.loading) bubble.classList.add("is-loading");
  if (options.error) bubble.classList.add("is-error");
  bubble.textContent = text;

  entry.appendChild(tag);
  entry.appendChild(bubble);
  transcript.appendChild(entry);
  transcript.scrollTop = transcript.scrollHeight;

  return bubble;
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) return;

  addEntry(question, "user");
  questionInput.value = "";
  sendBtn.disabled = true;

  const loadingBubble = addEntry("Reviewing the file...", "bot", { loading: true });

  try {
    const response = await fetch("/chat-api", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ question }),
    });

    const data = await response.json();

    loadingBubble.classList.remove("is-loading");

    if (data.status === "success") {
      loadingBubble.textContent = data.answer;
    } else {
      loadingBubble.classList.add("is-error");
      loadingBubble.textContent = "Something went wrong: " + (data.message || "unknown error");
    }
  } catch (err) {
    loadingBubble.classList.remove("is-loading");
    loadingBubble.classList.add("is-error");
    loadingBubble.textContent = "Could not reach the server. Please try again.";
  } finally {
    sendBtn.disabled = false;
    questionInput.focus();
  }
});

resetBtn.addEventListener("click", async () => {
  try {
    await fetch("/chat-reset", { method: "POST" });
  } catch (err) {
    // Non-critical if this fails - history just won't reset server-side
  }

  transcript.innerHTML = WELCOME_HTML;
});