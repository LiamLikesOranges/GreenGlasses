"""
gui.py — greenglasses desktop app, HTML/CSS/JS front end.

This runs a real web page (web/index.html, web/style.css, web/app.js)
inside a Qt window using QWebEngineView (Chromium under the hood), so all
the styling is 100% normal CSS — no Qt-specific styling language involved.
Python only does two things: runs the training pipeline as subprocesses,
and exposes a small "bridge" object that JavaScript can call into and
receive live signals from.

Architecture:
    gui.py (this file)   <-- runs subprocesses, owns the Bridge object
        |
        | QWebChannel (Python <-> JS messaging, built into Qt)
        v
    web/index.html + style.css + app.js   <-- 100% normal web tech

To restyle the app: edit web/style.css only. It's real CSS — flexbox,
gradients, animations, custom fonts, all of it works. GPT (or anyone) can
edit that file without touching this one.

To change what buttons do or add new ones: edit web/index.html (structure)
and web/app.js (behavior — it calls bridge.runStage('train') etc., which
this file listens for). New backend capabilities go here, as new @Slot
methods on Bridge.

Run it with:
    python gui.py

If the window doesn't appear, run this from a terminal (not by
double-clicking) so you can see the error — a startup crash before
window.show() closes instantly otherwise.
"""

import os
import re
import sys
import traceback

from PySide6.QtCore import QObject, QProcess, QUrl, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication, QMainWindow

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(PROJECT_ROOT, "web")
INDEX_HTML = os.path.join(WEB_DIR, "index.html")

# Try to read MAX_ITERS from config.py so the progress bar knows the total.
try:
    sys.path.insert(0, PROJECT_ROOT)
    import config as gg_config
    DEFAULT_MAX_ITERS = gg_config.MAX_ITERS
except Exception:
    DEFAULT_MAX_ITERS = 3000

STEP_LINE_RE = re.compile(r"step\s+(\d+)\s*\|")

STAGE_INFO = {
    "prepare": ("Preparing data", [os.path.join(PROJECT_ROOT, "data", "prepare.py")]),
    "train": ("Training model", [os.path.join(PROJECT_ROOT, "train.py")]),
    "sample": ("Generating sample", [os.path.join(PROJECT_ROOT, "sample.py")]),
}
STAGE_GROUPS = {
    "prepare": ["prepare"],
    "train": ["train"],
    "sample": ["sample"],
    "all": ["prepare", "train", "sample"],
}


class Bridge(QObject):
    """The Python-side half of the JS bridge. Every Signal here becomes an
    event JS can listen to (bridge.logLine.connect(...) in app.js). Every
    @Slot here becomes a function JS can call (bridge.runStage(...))."""

    logLine = Signal(str)
    statusChanged = Signal(str)
    progressChanged = Signal(int)
    progressMax = Signal(int)
    pipelineStarted = Signal()
    pipelineFinished = Signal()
    pipelineFailed = Signal(str)

    def __init__(self):
        super().__init__()
        self.process = None
        self.queue = []
        self.current_label = None

    @Slot(str)
    def runStage(self, stage_group):
        if self.process is not None:
            self.logLine.emit("[gui] A stage is already running.\n")
            return

        names = STAGE_GROUPS.get(stage_group)
        if not names:
            self.logLine.emit(f"[gui] Unknown stage: {stage_group}\n")
            return

        self.queue = [STAGE_INFO[n] for n in names]
        self.progressMax.emit(DEFAULT_MAX_ITERS)
        self.progressChanged.emit(0)
        self.pipelineStarted.emit()
        self._run_next()

    @Slot()
    def stopStage(self):
        self.queue = []
        if self.process is not None:
            self.process.kill()
        self.statusChanged.emit("Stopped")
        self.logLine.emit("\n[gui] Stopped by user.\n")
        self.pipelineFinished.emit()

    @Slot(result=int)
    def getMaxIters(self):
        return DEFAULT_MAX_ITERS

    def _run_next(self):
        if not self.queue:
            self.statusChanged.emit("Idle")
            self.logLine.emit("\n[gui] Pipeline finished.\n")
            self.pipelineFinished.emit()
            return

        label, args = self.queue.pop(0)
        self.current_label = label
        self.statusChanged.emit(f"{label} ...")
        self.logLine.emit(f"\n=== {label} ===\n")

        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.setWorkingDirectory(PROJECT_ROOT)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._stage_finished)
        self.process.start(sys.executable, args)

    def _read_output(self):
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.logLine.emit(data)

        if self.current_label == "Training model":
            match = None
            for match in STEP_LINE_RE.finditer(data):
                pass
            if match:
                step = int(match.group(1))
                self.progressChanged.emit(min(step, DEFAULT_MAX_ITERS))

    def _stage_finished(self, exit_code, exit_status):
        proc = self.process
        self.process = None
        if exit_code != 0:
            self.logLine.emit(f"\n[gui] Stage failed (exit code {exit_code}). Stopping pipeline.\n")
            self.queue = []
            self.statusChanged.emit("Failed")
            self.pipelineFailed.emit(self.current_label or "stage")
            return
        proc.deleteLater()
        self._run_next()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("greenglasses")
        self.resize(920, 660)

        self.view = QWebEngineView()
        self.setCentralWidget(self.view)

        self.bridge = Bridge()
        self.channel = QWebChannel()
        self.channel.registerObject("bridge", self.bridge)
        self.view.page().setWebChannel(self.channel)

        if not os.path.exists(INDEX_HTML):
            raise FileNotFoundError(
                f"Couldn't find {INDEX_HTML} — make sure the web/ folder "
                "(index.html, style.css, app.js) sits next to gui.py."
            )
        self.view.load(QUrl.fromLocalFile(INDEX_HTML))


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Print the full error and wait for a keypress so a double-clicked
        # .py file doesn't just flash a console window and vanish.
        traceback.print_exc()
        input("\n[gui] Crashed on startup (see error above). Press Enter to close...")