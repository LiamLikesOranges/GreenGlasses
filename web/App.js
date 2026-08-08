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
  btnPrepare: document.getElementById("btn-prepare"),
  btnTrain: document.getElementById("btn-train"),
  btnSample: document.getElementById("btn-sample"),
  btnAll: document.getElementById("btn-all"),
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

function setRunningState(isRunning) {
  setStageButtonsEnabled(!isRunning);
  els.btnStop.disabled = !isRunning;
}

new QWebChannel(qt.webChannelTransport, function (channel) {
  bridge = channel.objects.bridge;

  bridge.logLine.connect(appendLog);
  bridge.statusChanged.connect(setStatus);
  bridge.progressChanged.connect(setProgress);
  bridge.progressMax.connect(function (value) {
    maxIters = value;
    setProgress(0);
  });
  bridge.pipelineStarted.connect(function () {
    setRunningState(true);
  });
  bridge.pipelineFinished.connect(function () {
    setRunningState(false);
  });
  bridge.pipelineFailed.connect(function (stageLabel) {
    setRunningState(false);
  });

  // fetch initial max iters so the progress bar has a real scale at rest
  bridge.getMaxIters(function (value) {
    maxIters = value;
    setProgress(0);
  });
});

els.btnPrepare.addEventListener("click", () => bridge.runStage("prepare"));
els.btnTrain.addEventListener("click", () => bridge.runStage("train"));
els.btnSample.addEventListener("click", () => bridge.runStage("sample"));
els.btnAll.addEventListener("click", () => bridge.runStage("all"));
els.btnStop.addEventListener("click", () => bridge.stopStage());