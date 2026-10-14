"""Headless command line for RawRefinery.

    rawrefinery-cli INPUT [INPUT ...] --out DIR [--model NAME] [--iso auto|N] [--blend 0-100]
                    [--device mps|cuda|cpu] [--json]

Each INPUT is a raw file or a folder of raw files. Results are written to DIR as NAME_denoised.dng (all metadata
copied from the original). Existing outputs and the originals are never overwritten.

With --json one JSON object per line is written to stdout (everything else goes to stderr):
    {"event": "start",    "file", "index", "total", "output"}
    {"event": "progress", "file", "fraction"}
    {"event": "done",     "file", "output", "seconds", "message"}
    {"event": "skipped",  "file", "output"?, "reason"}
    {"event": "error",    "file", "message"}
    {"event": "finished", "total", "done", "skipped", "failed"}
Exit status: 0 = every file done or skipped, 1 = at least one file failed, 2 = bad usage.
"""
import argparse
import json
import os
import sys
from pathlib import Path

RAW_EXTENSIONS = (".cr2", ".cr3", ".nef", ".arw", ".dng")


def collect_inputs(paths):
    files = []
    for entry in paths:
        path = Path(entry).expanduser()
        if path.is_dir():
            files += sorted(p for p in path.iterdir() if p.suffix.lower() in RAW_EXTENSIONS and not p.name.startswith("."))
        else:
            files.append(path)
    return files


def explain(error):
    text = str(error)
    if "Unsupported file format" in text:
        return "not an original raw file (e.g. a DNG already developed by Lightroom)"
    return text


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="rawrefinery-cli", description="Denoise raw files with RawRefinery, without a window.")
    parser.add_argument("inputs", nargs="+", help="raw files or folders")
    parser.add_argument("--out", required=True, help="output folder (created if missing)")
    parser.add_argument("--model", default="Tree Net Denoise", help="model name (default: %(default)s)")
    parser.add_argument("--iso", default="auto", help="noise level: 'auto' (each photo's ISO) or a number")
    parser.add_argument("--blend", type=int, default=0, help="percent of the original mixed back in, 0-100 (default 0)")
    parser.add_argument("--device", default=None, help="mps, cuda or cpu (default: best available)")
    parser.add_argument("--json", action="store_true", help="machine readable progress, one JSON object per line")
    args = parser.parse_args(argv)
    if not 0 <= args.blend <= 100:
        parser.error("--blend must be between 0 and 100")
    if args.iso != "auto":
        try:
            args.iso = int(args.iso)
        except ValueError:
            parser.error("--iso must be 'auto' or a number")
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    # Library code prints to stdout: keep stdout clean for the JSON events.
    events_out = sys.stdout
    sys.stdout = sys.stderr

    def emit(event, **data):
        if args.json:
            events_out.write(json.dumps({"event": event, **data}) + "\n")
            events_out.flush()
        elif event == "start":
            print(f"[{data['index']}/{data['total']}] {Path(data['file']).name} ...", file=sys.stderr)
        elif event in ("done", "skipped", "error"):
            print(f"    {event}: {data.get('message') or data.get('reason') or data.get('output', '')}", file=sys.stderr)

    from PySide6.QtCore import QCoreApplication, QObject, QTimer
    from RawRefinery.application.ModelHandler import ModelController, MODEL_REGISTRY

    if args.model not in MODEL_REGISTRY:
        print(f"unknown model {args.model!r}. Available: {', '.join(MODEL_REGISTRY)}", file=sys.stderr)
        return 2
    files = collect_inputs(args.inputs)
    if not files:
        print("no input files", file=sys.stderr)
        return 2
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    app = QCoreApplication(sys.argv[:1])
    controller = ModelController()
    if args.device:
        if args.device not in controller.devices:
            print(f"device {args.device!r} not available. Available: {', '.join(controller.devices)}", file=sys.stderr)
            return 2
        controller.set_device(args.device)
    controller.load_model(args.model)
    if controller.model is None:
        print(f"could not load model {args.model!r}", file=sys.stderr)
        return 1

    class Runner(QObject):
        def __init__(self):
            super().__init__()
            self.index = 0
            self.current = None
            self.counts = {"done": 0, "skipped": 0, "failed": 0}
            self.last_progress = -1.0
            controller.save_finished.connect(self.on_saved)
            controller.error_occurred.connect(self.on_error)
            controller.progress_update.connect(self.on_progress)

        def next(self):
            if self.index >= len(files):
                emit("finished", total=len(files), **self.counts)
                app.quit()
                return
            source = files[self.index]
            self.index += 1
            self.current = source
            output = out_dir / f"{source.stem}_denoised.dng"
            if not source.is_file():
                self.counts["failed"] += 1
                emit("error", file=str(source), message="file not found")
                return QTimer.singleShot(0, self.next)
            emit("start", file=str(source), index=self.index, total=len(files), output=str(output))
            if output.exists():
                self.counts["skipped"] += 1
                emit("skipped", file=str(source), output=str(output), reason="output already exists")
                return QTimer.singleShot(0, self.next)
            try:
                iso = controller.load_rh(str(source))
            except Exception as e:
                self.counts["skipped"] += 1
                emit("skipped", file=str(source), reason=explain(e))
                return QTimer.singleShot(0, self.next)
            self.last_progress = -1.0
            controller.save_image(str(output), [iso if args.iso == "auto" else args.iso, args.blend], save_cfa=True)

        def on_progress(self, fraction):
            if fraction - self.last_progress >= 0.05 or fraction >= 1.0:
                self.last_progress = fraction
                if args.json and self.current is not None:
                    emit("progress", file=str(self.current), fraction=round(float(fraction), 3))

        def on_saved(self, message):
            self.counts["done"] += 1
            output = out_dir / f"{self.current.stem}_denoised.dng"
            seconds = None
            if message.startswith("Done in "):
                try:
                    seconds = float(message.split()[2])
                except ValueError:
                    pass
            emit("done", file=str(self.current), output=str(output), seconds=seconds, message=message)
            QTimer.singleShot(200, self.next)  # let the finished worker thread clean up first

        def on_error(self, message):
            self.counts["failed"] += 1
            emit("error", file=str(self.current) if self.current else None, message=message)
            QTimer.singleShot(200, self.next)

    runner = Runner()
    QTimer.singleShot(0, runner.next)
    app.exec()
    return 1 if runner.counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
