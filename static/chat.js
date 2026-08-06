const transcript = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");
const resetBtn = document.getElementById("reset-btn");
const chatChips = document.querySelectorAll(".chat-chip");

const WELCOME_HTML = `
  <div class="entry entry--bot">
    <span class="entry__tag">RECRUITAI ASSISTANT</span>
    <div class="entry__bubble">
      Ask me anything about the uploaded candidate resumes — for example:
      "Who knows Python?", "Who has 2+ years experience?", or
      "Who worked on machine learning projects?".
    </div>
  </div>
`;

function addEntry(text, sender, options = {}) {
  const entry = document.createElement("div");
  entry.className = `entry entry--${sender}`;

  const tag = document.createElement("span");
  tag.className = "entry__tag";
  tag.textContent = sender === "user" ? "YOU ASK" : "RECRUITAI ASSISTANT";

  const bubble = document.createElement("div");
  bubble.className = "entry__bubble";

  if (options.loading) {
    bubble.classList.add("is-loading");
    bubble.innerHTML = `Analyzing candidates <span class="typing-dots"><span></span><span></span><span></span></span>`;
  } else {
    if (options.error) bubble.classList.add("is-error");
    bubble.textContent = text;
  }

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

  const loadingBubble = addEntry("", "bot", { loading: true });

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
      loadingBubble.textContent = "Error: " + (data.message || "Could not generate answer.");
    }
  } catch (err) {
    loadingBubble.classList.remove("is-loading");
    loadingBubble.classList.add("is-error");
    loadingBubble.textContent = "Could not reach the server. Please check your connection.";
  } finally {
    sendBtn.disabled = false;
    questionInput.focus();
    transcript.scrollTop = transcript.scrollHeight;
  }
});

chatChips.forEach(chip => {
  chip.addEventListener("click", () => {
    questionInput.value = chip.getAttribute("data-q");
    chatForm.requestSubmit();
  });
});

resetBtn.addEventListener("click", async () => {
  try {
    await fetch("/chat-reset", { method: "POST" });
  } catch (err) {
    // Non-critical if reset endpoint fails
  }

  transcript.innerHTML = WELCOME_HTML;
});