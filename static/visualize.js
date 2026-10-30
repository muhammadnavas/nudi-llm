const $ = (id) => document.getElementById(id);
const form = $("visualize-form");
const runButton = $("run-btn");
const statusLine = $("status");
const results = $("results");
let traceData = null;
let playbackTimer = null;

const sleep = (ms) => new Promise(resolve => { playbackTimer = setTimeout(resolve, ms); });
const formatNumber = (value) => Number(value).toLocaleString("en-US");
const escapeHTML = (value) => String(value).replace(/[&<>'"]/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
}[char]));

function setActiveStage(id) {
  document.querySelectorAll(".stage").forEach(stage => stage.classList.toggle("active", stage.id === id));
}

function renderTokens(tokens, target, className) {
  target.innerHTML = "";
  tokens.forEach(token => {
    const chip = document.createElement("span");
    chip.className = className;
    chip.textContent = token.piece;
    chip.title = token.text || token.piece;
    target.appendChild(chip);
  });
}

function renderNetwork(model) {
  const svg = $("network");
  const ns = "http://www.w3.org/2000/svg";
  svg.innerHTML = "";
  const columns = [
    { label: "Input", x: 70, count: 5 },
    { label: "Embeddings", x: 205, count: 5 },
    ...Array.from({length: Math.min(model.layers, 4)}, (_, i) => ({
      label: i === Math.min(model.layers, 4) - 1 && model.layers > 4 ? "Layer " + model.layers : "Layer " + (i + 1),
      x: 320 + i * 85, count: 5
    })),
    { label: "Logits", x: 700, count: 5 }
  ];
  const positions = columns.map((column, colIndex) =>
    Array.from({length: column.count}, (_, row) => ({
      x: column.x, y: 42 + row * 39, colIndex, row
    }))
  );
  const add = (tag, attrs, text) => {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    svg.appendChild(node);
    return node;
  };
  for (let col = 0; col < positions.length - 1; col++) {
    positions[col].forEach((from, i) => positions[col + 1].forEach((to, j) => {
      add("line", {x1:from.x+7,y1:from.y,x2:to.x-7,y2:to.y,stroke:"#526985","stroke-opacity":".24","stroke-width":"1"});
    }));
  }
  positions.forEach((column, colIndex) => {
    column.forEach(point => add("circle", {
      cx:point.x,cy:point.y,r:6.5,fill:colIndex === 0 ? "#56a2ff" : colIndex === positions.length-1 ? "#3bd4a5" : "#6c83a2",
      stroke:"#b4c7df","stroke-opacity":".35","stroke-width":"1"
    }));
    add("text",{x:column[0].x,y:244,fill:"#91a2bb","font-size":"10","text-anchor":"middle"},columns[colIndex].label);
  });
}
function renderCandidates(candidates, selectedId) {
  const target = $("candidate-list");
  target.innerHTML = "";
  candidates.forEach(candidate => {
    const row = document.createElement("div");
    row.className = "candidate" + (candidate.id === selectedId ? " picked" : "");
    const name = document.createElement("span");
    name.className = "candidate-name";
    name.textContent = candidate.text || candidate.piece;
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("span");
    fill.style.width = Math.max(0, Math.min(100, candidate.probability * 100)) + "%";
    bar.appendChild(fill);
    const value = document.createElement("span");
    value.className = "candidate-value";
    value.textContent = (candidate.probability * 100).toFixed(1) + "%";
    row.append(name, bar, value);
    target.appendChild(row);
  });
}

function showSummary(data) {
  $("model-summary").textContent = data.model.layers + " layers · " + data.model.attention_heads + " heads";
  $("prompt-count").textContent = data.tokenizer.prompt_token_count;
  $("generated-count").textContent = data.steps.length;
  $("parameter-count").textContent = (data.model.parameters / 1e6).toFixed(2) + "M";
  $("network-meta").innerHTML =
    "<span><b>" + data.model.embedding_dimension + "</b> embedding dim</span>" +
    "<span><b>" + data.model.attention_heads + "</b> attention heads</span>" +
    "<span><b>" + data.model.layers + "</b> decoder blocks</span>" +
    "<span><b>" + data.model.vocabulary_size.toLocaleString() + "</b> vocabulary</span>";
  renderTokens(data.tokenizer.prompt_tokens, $("token-chips"), "token-chip");
  $("token-ids").innerHTML = "";
  data.tokenizer.prompt_tokens.forEach(token => {
    const chip = document.createElement("span");
    chip.className = "id-chip";
    chip.textContent = token.id;
    chip.title = token.piece;
    $("token-ids").appendChild(chip);
  });
  renderNetwork(data.model);
}

async function playTrace(data) {
  if (playbackTimer) clearTimeout(playbackTimer);
  const ids = $("token-ids");
  const response = $("response-text");
  const promptIds = data.tokenizer.prompt_tokens;
  ids.innerHTML = "";
  promptIds.forEach(token => {
    const chip = document.createElement("span");
    chip.className = "id-chip";
    chip.textContent = token.id;
    ids.appendChild(chip);
  });
  response.textContent = "";
  $("decode-id").textContent = "ID —";
  $("decode-piece").textContent = "Waiting…";
  $("completion-note").textContent = "";
  $("candidate-list").innerHTML = '<div class="empty-state">Waiting for the first prediction…</div>';
  $("progress-fill").style.width = "0%";
  $("step-count").textContent = "0 / " + data.steps.length;
  $("playback-label").textContent = "Generation trace";
  setActiveStage("stage-tokenizer");
  await sleep(450);
  setActiveStage("stage-ids");
  await sleep(450);
  setActiveStage("stage-transformer");
  await sleep(450);

  const generatedIds = [];
  for (let index = 0; index < data.steps.length; index++) {
    const step = data.steps[index];
    setActiveStage("stage-transformer");
    $("playback-label").textContent = "Forward pass · token " + step.index;
    await sleep(220);
    setActiveStage("stage-softmax");
    renderCandidates(step.candidates, step.selected.id);
    await sleep(430);
    setActiveStage("stage-decode");
    $("decode-id").textContent = "ID " + step.selected.id;
    $("decode-piece").textContent = step.selected.is_eos ? "<end>" : (step.selected.text || step.selected.piece || " ");
    await sleep(280);
    generatedIds.push(step.selected.id);
    // Use the exact decoded prefix from the backend; individual SentencePiece pieces
    // are not always independently decodable into the same concatenated string.
    response.textContent = step.cumulative_text || data.generated_text;
    setActiveStage("stage-response");
    const pct = ((index + 1) / Math.max(1, data.steps.length)) * 100;
    $("progress-fill").style.width = pct + "%";
    $("step-count").textContent = (index + 1) + " / " + data.steps.length;
    $("generated-count").textContent = index + 1;
    const chip = document.createElement("span");
    chip.className = "id-chip generated-chip";
    chip.textContent = step.selected.id;
    ids.appendChild(chip);
    await sleep(480);
  }
  setActiveStage("stage-response");
  $("playback-label").textContent = "Generation complete";
  $("completion-note").textContent = data.stopped_on_eos
    ? "The model emitted its end-of-sequence token and stopped."
    : "Reached the requested generation limit. The model predicts one token per forward pass.";
  $("progress-fill").style.width = "100%";
}

async function runVisualization() {
  const prompt = $("prompt").value.trim();
  if (!prompt) {
    statusLine.textContent = "Please enter a prompt first.";
    return;
  }
  if (playbackTimer) clearTimeout(playbackTimer);
  runButton.disabled = true;
  runButton.textContent = "Running model…";
  statusLine.textContent = "Calling NudiLLM and collecting its actual next-token distributions…";
  results.hidden = true;
  try {
    const response = await fetch("/api/visualize", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        prompt,
        use_chat_format: $("mode").value === "chat",
        max_new_tokens: Number($("max-tokens").value),
        temperature: Number($("temperature").value),
        top_k: 20,
        top_p: 0.9,
        repetition_penalty: 1.3
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Generation failed.");
    traceData = data;
    results.hidden = false;
    showSummary(data);
    statusLine.textContent = "Trace ready. Watch each real prediction step below.";
    await playTrace(data);
  } catch (error) {
    statusLine.textContent = "Unable to visualize: " + error.message;
  } finally {
    runButton.disabled = false;
    runButton.innerHTML = "<span>▶</span> Visualize generation";
  }
}

form.addEventListener("submit", event => {
  event.preventDefault();
  runVisualization();
});
$("replay-btn").addEventListener("click", async () => {
  if (!traceData) return;
  await playTrace(traceData);
});
