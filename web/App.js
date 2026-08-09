/*
  app.js — client-side logic for the greenglasses GUI.

  Connects to the Python "bridge" object over QWebChannel (Qt's built-in
  Python<->JS messaging, loaded via qwebchannel.js in index.html). From
  here on, it's normal DOM/JS: bridge.runStage('train') calls into Python,
  and bridge.someSignal.connect(fn) listens for events Python sends back.

  Available from Python (see gui.py's Bridge class):
    Signals (things Python tells JS):
      logLine(text)            new log output to append
      statusChanged(text)        status line text
      progressChanged(value)        current training step
      progressMax(value)              total training steps
      pipelineStarted()                  a run just began
      pipelineFinished()                    a run completed or was stopped
      pipelineFailed(stageLabel)               a run failed

    Slots (things JS can call):
      runStage(name)   name is 'prepare' | 'train' | 'sample' | 'all'
      stopStage()
*/

let bridge = null;
let maxIters = 0;

const els = {
  status: document.getElementById("status"),
  progressFill: document.getElementById("progress-fill"),
  progressLabel: document.getElementById("progress-label"),
  log: document.getElementById("log"),
  devLog: document.getElementById("dev-log"),
  modelList: document.getElementById("model-list"),
  modelDetails: document.getElementById("model-details"),
  btnPrepare: document.getElementById("btn-prepare"),
  btnTrain: document.getElementById("btn-train"),
  btnSample: document.getElementById("btn-sample"),
  btnAll: document.getElementById("btn-all"),
  btnRefreshModels: document.getElementById("btn-refresh-models"),
  btnStop: document.getElementById("btn-stop"),
};

const stageButtons = [els.btnPrepare, els.btnTrain, els.btnSample, els.btnAll];

function appendLog(text) {
  els.log.textContent += text;
  els.log.scrollTop = els.log.scrollHeight;
}

function setStatus(text) {
  els.status.textContent = text;

  // Only the "Training model" stage has numbered steps to show real
  // progress for. Prepare/Sample stages don't — show a pulsing
  // "working" animation instead so it's clear something is happening
  // rather than looking frozen at 0.
  if (text.startsWith("Training model")) {
    setIndeterminate(false);
  } else if (text.endsWith("...")) {
    setIndeterminate(true);
  } else {
    // Idle / Stopped / Failed
    setIndeterminate(false);
    setProgress(0);
  }
}

function setIndeterminate(on) {
  els.progressFill.classList.toggle("indeterminate", on);
  if (on) {
    els.progressLabel.textContent = "working…";
  }
}

function setProgress(value) {
  const pct = maxIters > 0 ? Math.min(100, (value / maxIters) * 100) : 0;
  els.progressFill.style.width = pct + "%";
  els.progressLabel.textContent = `${value} / ${maxIters}`;
}

function setStageButtonsEnabled(enabled) {
  stageButtons.forEach((btn) => (btn.disabled = !enabled));
}

function appendDevLog(text) {
  els.devLog.textContent += text;
  els.devLog.scrollTop = els.devLog.scrollHeight;
}

function setRunningState(isRunning) {
  setStageButtonsEnabled(!isRunning);
  els.btnStop.disabled = !isRunning;
}

function renderModelList(text) {
  const models = JSON.parse(text || "[]");
  els.modelList.innerHTML = "";
  if (models.length === 0) {
    els.modelList.textContent = "No trained models found.";
    return;
  }
  models.forEach((model) => {
    const button = document.createElement("button");
    button.className = "btn btn-secondary model-item";
    const sizeTag = model.size_tag ? ` · ${model.size_tag} params` : "";
    button.textContent = `${model.name} (iter ${model.iter ?? "?"}${sizeTag})`;
    button.addEventListener("click", () => {
      bridge.requestModelDetails(model.name);
    });
    els.modelList.appendChild(button);
  });
}

function renderModelDetails(text) {
  const details = JSON.parse(text || "{}");
  if (details.error) {
    els.modelDetails.textContent = details.error;
    return;
  }
  const lines = [];
  lines.push(`Folder: models/${details.folder}`);
  lines.push(`Name: ${details.name}  (version ${details.version})`);
  if (details.description) {
    lines.push(`\n${details.description}`);
  }
  lines.push("");
  lines.push(`Iteration: ${details.iteration}${details.max_iters ? ` / ${details.max_iters}` : ""}`);
  lines.push(`Parameters: ${details.num_params?.toLocaleString() || "unknown"} (${details.size_tag || "?"})`);
  if (details.created_at) {
    lines.push(`Created: ${new Date(details.created_at).toLocaleString()}`);
  }
  if (details.updated_at) {
    lines.push(`Last updated: ${new Date(details.updated_at).toLocaleString()}`);
  }
  lines.push(`\nArchitecture:`);
  Object.entries(details.config || {}).forEach(([key, value]) => {
    lines.push(`  ${key}: ${value}`);
  });
  els.modelDetails.textContent = lines.join("\n");
}

new QWebChannel(qt.webChannelTransport, function (channel) {
  bridge = channel.objects.bridge;

  bridge.logLine.connect(appendLog);
  bridge.statusChanged.connect(setStatus);
  bridge.progressChanged.connect(function (value) {
    // Any real progress value we receive (a training step count, or a
    // download percent) means we have genuine data to show — drop the
    // pulsing "working" animation in favor of the real number.
    setIndeterminate(false);
    setProgress(value);
  });
  bridge.progressMax.connect(function (value) {
    maxIters = value;
    setProgress(0);
  });
  bridge.pipelineStarted.connect(function () {
    appendDevLog("[ui] Pipeline started\n");
    setRunningState(true);
  });
  bridge.pipelineFinished.connect(function () {
    setRunningState(false);
  });
  bridge.pipelineFailed.connect(function (stageLabel) {
    appendDevLog(`[ui] Pipeline failed during ${stageLabel}\n`);
    setRunningState(false);
  });

  bridge.modelListChanged.connect(renderModelList);
  bridge.modelDetailsChanged.connect(renderModelDetails);

  // fetch initial max iters so the progress bar has a real scale at rest
  bridge.getMaxIters(function (value) {
    maxIters = value;
    setProgress(0);
  });

  // refresh model list at startup
  bridge.refreshModelList();

  bridge.internalLogLine.connect(function (text) {
    appendDevLog(text);
  });
});

els.btnPrepare.addEventListener("click", () => bridge.runStage("prepare"));
els.btnTrain.addEventListener("click", () => bridge.runStage("train"));
els.btnSample.addEventListener("click", () => bridge.runStage("sample"));
els.btnAll.addEventListener("click", () => bridge.runStage("all"));
els.btnRefreshModels.addEventListener("click", () => bridge.refreshModelList());
els.btnStop.addEventListener("click", () => bridge.stopStage());