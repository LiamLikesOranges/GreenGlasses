"""
main.py — the single entry point for greenglasses.

Default behavior: opens the GUI window (gui.py), which loads the web/
folder (index.html, style.css, app.js) and lets you run the pipeline
with buttons and a live log/progress view.

    python main.py

If you'd rather run the pipeline headlessly from a terminal (no window),
the old CLI flags still work — pass any of them and main.py skips the GUI
entirely and runs the stages directly, printing logs to your terminal:

    python main.py --cli                        # run everything, no window
    python main.py --cli --prompt "ROMEO:"        # customize final sample prompt
    python main.py --cli --skip-download           # use your own data/input.txt as-is
    python main.py --cli --only train                # run just one stage
    python main.py --cli --only prepare,train           # run a subset, in order

Stages: prepare, train, sample
"""

import argparse
import os
import subprocess
import sys

STAGES = ["prepare", "train", "sample"]
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\n[greenglasses] Step failed: {' '.join(cmd)}")
        sys.exit(result.returncode)


def launch_gui():
    """Imports and runs gui.py's main() directly, so it's the same Python
    process and the window pops up immediately."""
    web_dir = os.path.join(PROJECT_ROOT, "web")
    if not os.path.exists(web_dir):
        print(
            f"[greenglasses] Couldn't find {web_dir}.\n"
            "Make sure the 'web' folder (index.html, style.css, app.js) "
            "sits directly inside this project folder, next to main.py."
        )
        sys.exit(1)

    sys.path.insert(0, PROJECT_ROOT)
    try:
        import gui
    except ImportError as e:
        print(
            f"[greenglasses] Couldn't import gui.py ({e}).\n"
            "Make sure gui.py is in this same folder, and PySide6 is installed:\n"
            "    pip install -r requirements.txt"
        )
        sys.exit(1)

    gui.main()


def run_cli_pipeline(args):
    if args.only:
        stages = [s.strip() for s in args.only.split(",")]
        bad = [s for s in stages if s not in STAGES]
        if bad:
            print(f"[greenglasses] Unknown stage(s): {bad}. Choices are: {STAGES}")
            sys.exit(1)
    else:
        stages = STAGES

    py = sys.executable  # use whatever python is currently running this script

    if "prepare" in stages:
        cmd = [py, "data/prepare.py"]
        if args.skip_download:
            cmd.append("--skip-download")
        run(cmd)

    if "train" in stages:
        run([py, "train.py"])

    if "sample" in stages:
        cmd = [py, "sample.py", "--prompt", args.prompt]
        if args.max_new_tokens is not None:
            cmd += ["--max_new_tokens", str(args.max_new_tokens)]
        if args.temperature is not None:
            cmd += ["--temperature", str(args.temperature)]
        run(cmd)

    print("\n[greenglasses] All done.")


def main():
    parser = argparse.ArgumentParser(description="Run greenglasses: GUI by default, CLI with --cli")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run the pipeline headlessly in this terminal instead of opening the GUI window.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=f"(--cli only) Comma-separated subset of stages to run, in order. "
             f"Choices: {', '.join(STAGES)}. Default: run all of them.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="(--cli only) Data stage: use existing data/input.txt instead of downloading.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="\n",
        help="(--cli only) Sample stage: text to start generation from.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="(--cli only) Sample stage: how many characters to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="(--cli only) Sample stage: randomness of generation.",
    )
    args = parser.parse_args()

    if args.cli:
        run_cli_pipeline(args)
    else:
        launch_gui()


if __name__ == "__main__":
    main()